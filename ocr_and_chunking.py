from pdf2image import convert_from_path
import pytesseract
import os
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
import hashlib
from typing import Any, List, Dict


def upload_ocr(file: Any, folder: str) -> str:
    """Save an uploaded PDF to disk and extract text via OCR."""
    file_path = os.path.join(folder, file.filename)
    with open(file_path, "wb") as buffer:
        buffer.write(file.file.read())

    images = convert_from_path(file_path)
    text = ""
    for image in images:
        raw_text = pytesseract.image_to_string(image, lang="eng")
        text += raw_text
    return text


def upload_create_chunks(
    db: Any,
    file: Any,
    folder: str,
    chunk_size: int = 512,
    overlap: int = 32,
) -> List[Dict[str, Any]]:
    """OCR a PDF, chunk it, and return a dataset list ready for ContextualVectorDB."""

    doc_id = db.collection.count() + 1
    text = upload_ocr(file, folder)
    original_uuid = hashlib.sha256(text.encode("utf-8")).hexdigest()

    # --- Duplicate detection ---
    known_hashes = db.get_known_hashes()
    if original_uuid in known_hashes:
        raise ValueError(
            f"'{file.filename}' is already in the database (duplicate content detected). "
            "Skipping ingestion to avoid duplicate chunks."
        )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap
    )
    docs = text_splitter.create_documents([text])

    chunks_data = []
    for i, doc in enumerate(docs):
        chunks_data.append(
            {
                "chunk_id": f"doc_{doc_id}_chunk_{i}",
                "original_index": i,
                "content": doc.page_content,
            }
        )

    doc_data = {
        "doc_id": f"doc_{doc_id}",
        "original_uuid": original_uuid,
        "source_file": file.filename,       # ← stored for citations
        "content": text,
        "chunks": chunks_data,
    }

    print(f"Created {len(chunks_data)} chunks for doc_id=doc_{doc_id}")
    return [doc_data]


def save_chunks_to_file(
    chunks: List[Dict[str, Any]],
    output_file_name: str,
    output_path: str = "data/",
) -> None:
    """Persist chunked data as a JSON file."""
    os.makedirs(output_path, exist_ok=True)
    output_file_path = os.path.join(output_path, output_file_name)
    with open(output_file_path, "w", encoding="utf-8") as json_file:
        json.dump(chunks, json_file, ensure_ascii=False, indent=2)
    print(f"Saved chunks to: {output_file_path}")


def merge_context(response: List[Dict[str, Any]]) -> str:
    """Concatenate original_content from all retrieved chunks into a single context string."""
    parts = []
    for doc in response:
        content = doc["chunk"].get("original_content", "").strip()
        if content:
            parts.append(content)
    return "\n\n".join(parts)
