"""
data_pipeline.py
================

Stack (industry standard):
  - pymupdf4llm  : PDF → per-page Markdown chunks with headings, bbox, page numbers
  - SemanticChunker (langchain_experimental) : topic-boundary splitting using
    your existing all-mpnet-base-v2 embeddings
  - Chunk-level SHA-256 dedup on top of existing document-level dedup

Public API (called by app.py):
    from data_pipeline import ingest_pdf, save_chunks_to_file, merge_context
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pymupdf4llm
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

from config import settings


# ---------------------------------------------------------------------------
# Embeddings — reuse the same model as ChromaDB so no second model is loaded
# ---------------------------------------------------------------------------

_embeddings: Optional[HuggingFaceEmbeddings] = None


def _get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


# ---------------------------------------------------------------------------
# Step 1 — Extract: PDF → list of LangChain Documents (one per page)
# ---------------------------------------------------------------------------

def _extract_pages(pdf_path: str) -> List[Document]:
    """
    Use pymupdf4llm to extract per-page Markdown chunks.

    Each returned Document has:
      page_content : Markdown text (headings as ## / ###, tables preserved)
      metadata     : page_number, source_file, headings (list), extraction_method
    """
    path = Path(pdf_path)
    raw_chunks = pymupdf4llm.to_markdown(str(path), page_chunks=True)

    documents: List[Document] = []
    for chunk in raw_chunks:
        text: str = chunk.get("text", "").strip()
        if not text:
            continue

        page_boxes = chunk.get("page_boxes", [])

        # Extract heading texts from page_boxes (class == 'section-header')
        headings: List[str] = []
        for box in page_boxes:
            if isinstance(box, dict) and box.get("class") == "section-header":
                # Get the heading text from the Markdown output (## lines)
                pass  # headings are already in the Markdown text as ## lines

        # Parse ## headings directly from the Markdown text — reliable
        heading_lines = [
            ln.lstrip("#").strip()
            for ln in text.splitlines()
            if ln.startswith("#")
        ]

        # Page number: pymupdf4llm page_chunks are in order, 0-indexed
        page_num = len(documents) + 1   # 1-based

        documents.append(Document(
            page_content=text,
            metadata={
                "page_number":        page_num,
                "source_file":        path.name,
                "headings":           heading_lines,
                "extraction_method":  "pymupdf4llm",
            }
        ))

    print(f"[pipeline] extracted {len(documents)} pages from '{path.name}'")
    return documents


# ---------------------------------------------------------------------------
# Step 2 — Chunk: LangChain Documents → semantic chunks
# ---------------------------------------------------------------------------

def _semantic_chunk(documents: List[Document]) -> List[Document]:
    """
    Split each page Document semantically using LangChain SemanticChunker.

    SemanticChunker splits on cosine-similarity drops between consecutive
    sentences using your embedding model — no regex, no hard character limits.

    breakpoint_threshold_type options:
      "percentile"        — split where similarity < Xth percentile (default 95)
      "standard_deviation"— split where similarity drops > X std devs
      "interquartile"     — split on IQR

    We use "percentile" — industry default, most stable across doc types.
    Metadata from the parent page Document is preserved on every chunk.
    """
    embeddings = _get_embeddings()
    chunker = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile",
    )

    chunks = chunker.split_documents(documents)
    print(f"[pipeline] {len(documents)} pages → {len(chunks)} semantic chunks")
    return chunks


# ---------------------------------------------------------------------------
# Step 3 — Build dataset: chunks → ContextualVectorDB-compatible format
# ---------------------------------------------------------------------------

def _build_dataset(
    chunks: List[Document],
    doc_id: int,
    original_uuid: str,
    source_file: str,
    full_text: str,
    existing_chunk_hashes: set,
) -> tuple[List[Dict[str, Any]], List[str]]:
    """
    Convert LangChain Documents into the dataset format expected by
    ContextualVectorDB.load_data() / .append_data().

    Adds chunk-level SHA-256 dedup: chunks whose hash already exists in
    ChromaDB are silently skipped.

    Returns:
        chunk_list       : list of chunk dicts
        new_hashes       : hashes of chunks actually included (for caller to record)
    """
    chunk_list: List[Dict[str, Any]] = []
    new_hashes: List[str] = []
    skipped = 0

    for i, doc in enumerate(chunks):
        content = doc.page_content.strip()
        if not content:
            continue

        chunk_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

        # Chunk-level dedup check
        if chunk_hash in existing_chunk_hashes:
            skipped += 1
            continue

        chunk_list.append({
            "chunk_id":       f"doc_{doc_id}_chunk_{i}",
            "original_index": i,
            "content":        content,
            # Rich metadata for citations (Phase 5)
            "page_number":    doc.metadata.get("page_number", 0),
            "section_heading": (
                doc.metadata.get("headings", [""])[0]
                if doc.metadata.get("headings") else ""
            ),
            "source_file":    doc.metadata.get("source_file", source_file),
            "chunk_hash":     chunk_hash,
            "char_count":     len(content),
            "extraction_method": doc.metadata.get("extraction_method", "pymupdf4llm"),
        })
        new_hashes.append(chunk_hash)

    if skipped:
        print(f"[pipeline] skipped {skipped} duplicate chunks (chunk-level dedup)")

    return chunk_list, new_hashes


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ingest_pdf(
    db: Any,
    file: Any,
    folder: str,
) -> List[Dict[str, Any]]:
    """
    Full ingest pipeline for an uploaded PDF.

    Steps:
      1. Save file to disk
      2. Document-level dedup check (existing hash in ChromaDB)
      3. pymupdf4llm extraction → per-page LangChain Documents
      4. SemanticChunker → topic-boundary chunks
      5. Chunk-level dedup
      6. Build ContextualVectorDB-compatible dataset

    Returns a list with a single document dict (same shape as codebase_chunks.json).
    Raises ValueError on duplicate document.
    """
    # 1. Save to disk
    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, file.filename)
    with open(file_path, "wb") as f:
        f.write(file.file.read())
    print(f"[pipeline] saved '{file.filename}' → {file_path}")

    # 2. Extract full text for doc-level hash + contextual enrichment
    page_docs = _extract_pages(file_path)
    if not page_docs:
        raise ValueError(f"No text could be extracted from '{file.filename}'.")

    full_text = "\n\n".join(d.page_content for d in page_docs)
    original_uuid = hashlib.sha256(full_text.encode("utf-8")).hexdigest()

    # 3. Document-level dedup
    known_hashes = db.get_known_hashes()
    if original_uuid in known_hashes:
        raise ValueError(
            f"'{file.filename}' is already in the database "
            "(duplicate document detected). Skipping ingestion."
        )

    # 4. Semantic chunking
    chunks = _semantic_chunk(page_docs)

    # 5. Chunk-level dedup — collect hashes already in DB
    existing_chunk_hashes: set = set()
    for meta in db.metadata.get("metadatas", []):
        ch = meta.get("chunk_hash", "")
        if ch:
            existing_chunk_hashes.add(ch)

    # 6. Build dataset
    doc_id = db.collection.count() + 1
    chunk_list, _ = _build_dataset(
        chunks,
        doc_id=doc_id,
        original_uuid=original_uuid,
        source_file=file.filename,
        full_text=full_text,
        existing_chunk_hashes=existing_chunk_hashes,
    )

    if not chunk_list:
        raise ValueError(
            f"All chunks from '{file.filename}' already exist in the database."
        )

    dataset = [{
        "doc_id":        f"doc_{doc_id}",
        "original_uuid": original_uuid,
        "source_file":   file.filename,
        "content":       full_text,          # full text for contextual enrichment
        "chunks":        chunk_list,
    }]

    print(
        f"[pipeline] '{file.filename}' → {len(chunk_list)} chunks ready for embedding"
    )
    return dataset


def save_chunks_to_file(
    chunks: List[Dict[str, Any]],
    output_file_name: str,
    output_path: str = "data/",
) -> None:
    """Persist chunked dataset as a JSON file (same as before)."""
    os.makedirs(output_path, exist_ok=True)
    out = os.path.join(output_path, output_file_name)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"[pipeline] saved chunks → {out}")


def merge_context(response: List[Dict[str, Any]]) -> str:
    """
    Merge retrieved chunks into a single context string for the LLM.
    Same interface as the old ocr_and_chunking.merge_context().
    """
    parts = []
    for doc in response:
        content = doc["chunk"].get("original_content", "").strip()
        if content:
            parts.append(content)
    return "\n\n".join(parts)
