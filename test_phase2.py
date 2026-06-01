"""
test_phase2.py
==============
Phase 2 checkpoint tests — Better Ingestion and Chunking.

Tests (all offline, no Ollama needed):
  1. pymupdf4llm extracts pages with headings and page numbers
  2. SemanticChunker import + instantiation works
  3. data_pipeline.ingest_pdf produces correct dataset shape
  4. Chunk metadata contains page_number, section_heading, chunk_hash, source_file
  5. Document-level dedup: uploading the same file twice raises ValueError
  6. Chunk-level dedup: duplicate chunks from different docs are skipped
  7. merge_context returns correct string from retrieved chunks

Run:
    python test_phase2.py
"""

import sys, json, hashlib, tempfile, os
from pathlib import Path
from typing import List, Tuple

PASS = "✓"; FAIL = "✗"; SKIP = "○"
results: List[Tuple[str, str, str]] = []

def record(status, name, detail=""):
    results.append((status, name, detail))
    icon = {"pass": PASS, "fail": FAIL, "skip": SKIP}[status]
    print(f"  {icon}  {name}" + (f"  →  {detail}" if detail else ""))

def section(title):
    print(f"\n{'─'*55}\n  {title}\n{'─'*55}")

# ── helpers ────────────────────────────────────────────────────────────────

def make_test_pdf(path: str, text_pages: List[str]):
    """Create a minimal multi-page PDF with real text using PyMuPDF."""
    import fitz
    doc = fitz.open()
    for text in text_pages:
        page = doc.new_page()
        y = 72
        for line in text.splitlines():
            size = 14 if line.startswith("##") else 11
            clean = line.lstrip("#").strip()
            if clean:
                page.insert_text((50, y), clean, fontsize=size)
                y += size + 6
    doc.save(path)
    doc.close()


class MockFile:
    """Mimics a FastAPI UploadFile for testing."""
    def __init__(self, path: str):
        self.filename = Path(path).name
        self._path = path
        self.file = open(path, "rb")
        self.file.seek(0)   # ensure at start
    def close(self):
        self.file.close()


class MockDB:
    """Minimal mock of ContextualVectorDB for pipeline tests."""
    def __init__(self):
        self._hashes: set = set()
        self._chunk_hashes: set = set()
        self.metadata = {"metadatas": []}
        self._count = 0

    def get_known_hashes(self):
        return self._hashes

    def collection_count(self):
        return self._count

    class _Col:
        def __init__(self, parent): self._p = parent
        def count(self): return self._p._count

    @property
    def collection(self): return self._Col(self)


# ── Test 1: pymupdf4llm extraction ─────────────────────────────────────────

def test_extraction():
    section("1. pymupdf4llm extraction")
    try:
        import pymupdf4llm
        record("pass", "pymupdf4llm imported", f"version={pymupdf4llm.__version__}")
    except ImportError as e:
        record("fail", "pymupdf4llm import", str(e)); return

    with tempfile.TemporaryDirectory() as tmp:
        pdf = os.path.join(tmp, "test.pdf")
        make_test_pdf(pdf, [
            "## Section 1: Leave Policy\nEmployees get 20 days of annual leave.",
            "## Section 2: Benefits\nHealth insurance provided to all staff.",
        ])
        try:
            chunks = pymupdf4llm.to_markdown(pdf, page_chunks=True)
            assert len(chunks) >= 1
            record("pass", "to_markdown(page_chunks=True) returns list", f"{len(chunks)} page(s)")
        except Exception as e:
            record("fail", "pymupdf4llm.to_markdown", str(e)); return

        # Check heading detection
        found_heading = any(
            "##" in c.get("text", "") for c in chunks
        )
        if found_heading:
            record("pass", "Headings detected as ## in Markdown text")
        else:
            record("fail", "Heading detection", "no ## found in output")

        # Check page_boxes for section-header class
        found_header_box = any(
            any(b.get("class") == "section-header" for b in c.get("page_boxes", []) if isinstance(b, dict))
            for c in chunks
        )
        if found_header_box:
            record("pass", "page_boxes contains section-header entries")
        else:
            record("skip", "page_boxes section-header", "not present in this PDF type (ok for simple test PDFs)")


# ── Test 2: SemanticChunker import ─────────────────────────────────────────

def test_semantic_chunker_import():
    section("2. SemanticChunker import")
    try:
        from langchain_experimental.text_splitter import SemanticChunker
        record("pass", "SemanticChunker imported from langchain_experimental")
    except ImportError as e:
        record("fail", "SemanticChunker import", str(e))
        return

    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        record("pass", "HuggingFaceEmbeddings imported from langchain_huggingface")
    except ImportError as e:
        record("fail", "HuggingFaceEmbeddings import", str(e))


# ── Test 3 & 4: data_pipeline.ingest_pdf shape + metadata ──────────────────

def test_ingest_shape_and_metadata():
    section("3 & 4. ingest_pdf — dataset shape + chunk metadata")
    try:
        from data_pipeline import ingest_pdf
    except ImportError as e:
        record("fail", "data_pipeline import", str(e)); return

    with tempfile.TemporaryDirectory() as tmp:
        pdf = os.path.join(tmp, "policy.pdf")
        upload_folder = os.path.join(tmp, "uploads")   # separate folder prevents collision
        os.makedirs(upload_folder, exist_ok=True)
        make_test_pdf(pdf, [
            "## Annual Leave Policy\nEmployees receive 20 days leave per year.\nRequests must be submitted 2 weeks in advance.",
            "## Health Benefits\nAll employees are entitled to health insurance from day one.",
        ])

        db = MockDB()
        mf = MockFile(pdf)
        try:
            dataset = ingest_pdf(db, mf, upload_folder)
            mf.close()
        except OSError as e:
            if "huggingface.co" in str(e) or "cached files" in str(e):
                record("skip", "ingest_pdf() — embedding model not in local cache (run on your machine)", str(e)[:80])
                return
            record("fail", "ingest_pdf() execution", str(e))
            import traceback; traceback.print_exc()
            return
        except Exception as e:
            record("fail", "ingest_pdf() execution", str(e))
            import traceback; traceback.print_exc()
            return

        # Shape
        assert isinstance(dataset, list) and len(dataset) == 1, "Expected list with 1 doc"
        doc = dataset[0]
        record("pass", "ingest_pdf returns list with 1 doc")

        required_doc_keys = {"doc_id", "original_uuid", "source_file", "content", "chunks"}
        missing = required_doc_keys - set(doc.keys())
        if not missing:
            record("pass", "doc has all required top-level keys")
        else:
            record("fail", "doc missing keys", str(missing))

        chunks = doc["chunks"]
        assert len(chunks) > 0, "Expected at least 1 chunk"
        record("pass", f"doc contains {len(chunks)} chunk(s)")

        # Metadata per chunk
        required_chunk_keys = {"chunk_id", "original_index", "content", "page_number",
                                "section_heading", "source_file", "chunk_hash", "char_count"}
        sample = chunks[0]
        missing_chunk = required_chunk_keys - set(sample.keys())
        if not missing_chunk:
            record("pass", "chunk has all required metadata keys")
        else:
            record("fail", "chunk missing metadata keys", str(missing_chunk))

        # page_number is int >= 1
        if isinstance(sample.get("page_number"), int) and sample["page_number"] >= 1:
            record("pass", f"page_number is int ≥ 1", f"= {sample['page_number']}")
        else:
            record("fail", "page_number", f"got {sample.get('page_number')!r}")

        # chunk_hash is a 64-char hex string
        ch = sample.get("chunk_hash", "")
        if isinstance(ch, str) and len(ch) == 64:
            record("pass", "chunk_hash is 64-char SHA-256 hex")
        else:
            record("fail", "chunk_hash format", repr(ch[:20]))

        # source_file matches
        if sample.get("source_file") == "policy.pdf":
            record("pass", "source_file correctly set on chunk")
        else:
            record("fail", "source_file on chunk", repr(sample.get("source_file")))


# ── Test 5: Document-level dedup ───────────────────────────────────────────

def test_doc_dedup():
    section("5. Document-level dedup")
    try:
        from data_pipeline import ingest_pdf
    except ImportError as e:
        record("fail", "data_pipeline import", str(e)); return

    with tempfile.TemporaryDirectory() as tmp:
        pdf = os.path.join(tmp, "dup.pdf")
        make_test_pdf(pdf, ["## Policy\nThis document will be uploaded twice."])

        upload_folder = os.path.join(tmp, "uploads")
        os.makedirs(upload_folder, exist_ok=True)
        db = MockDB()

        # First upload — succeeds
        mf1 = MockFile(pdf)
        try:
            dataset = ingest_pdf(db, mf1, upload_folder)
            mf1.close()
            db._hashes.add(dataset[0]["original_uuid"])
            record("pass", "First upload succeeds")
        except OSError as e:
            if "huggingface.co" in str(e) or "cached files" in str(e):
                record("skip", "Doc-dedup test — embedding model not in local cache (run on your machine)")
                return
            record("fail", "First upload", str(e)); return
        except Exception as e:
            record("fail", "First upload", str(e)); return

        # Second upload of identical file — must raise ValueError
        import io
        with open(pdf, "rb") as f:
            data = f.read()

        class MockFile2:
            filename = "dup.pdf"
            file = io.BytesIO(data)

        try:
            ingest_pdf(db, MockFile2(), upload_folder)
            record("fail", "Second upload should raise ValueError", "no error raised")
        except ValueError as e:
            record("pass", "Second upload raises ValueError (doc-level dedup)", str(e)[:70])
        except Exception as e:
            record("fail", "Second upload raised wrong exception", str(e))


# ── Test 6: Chunk-level dedup ──────────────────────────────────────────────

def test_chunk_dedup():
    section("6. Chunk-level dedup")
    try:
        from data_pipeline import _build_dataset
        from langchain_core.documents import Document
    except ImportError as e:
        record("fail", "data_pipeline._build_dataset import", str(e)); return

    content = "Employees receive 20 days of annual leave per year."
    chunk_hash = hashlib.sha256(content.encode()).hexdigest()

    docs = [Document(page_content=content, metadata={"page_number": 1, "source_file": "p.pdf", "headings": []})]

    # Without existing hashes — chunk should be included
    result, new_hashes = _build_dataset(docs, 1, "uuid1", "p.pdf", content, existing_chunk_hashes=set())
    if len(result) == 1:
        record("pass", "Chunk included when not already in DB")
    else:
        record("fail", "Chunk inclusion", f"got {len(result)} chunks")

    # With existing hash — chunk should be skipped
    result2, _ = _build_dataset(docs, 2, "uuid2", "p.pdf", content, existing_chunk_hashes={chunk_hash})
    if len(result2) == 0:
        record("pass", "Duplicate chunk skipped (chunk-level dedup)")
    else:
        record("fail", "Chunk dedup", f"expected 0 chunks, got {len(result2)}")


# ── Test 7: merge_context ──────────────────────────────────────────────────

def test_merge_context():
    section("7. merge_context")
    try:
        from data_pipeline import merge_context
    except ImportError as e:
        record("fail", "merge_context import", str(e)); return

    retrieved = [
        {"chunk": {"original_content": "Employees get 20 days leave.", "chunk_id": "c1"}},
        {"chunk": {"original_content": "Health insurance is provided.", "chunk_id": "c2"}},
        {"chunk": {"original_content": "", "chunk_id": "c3"}},  # empty — should be skipped
    ]
    result = merge_context(retrieved)
    assert "20 days" in result
    assert "Health insurance" in result
    assert result.count("\n\n") == 1  # two non-empty chunks joined with \n\n
    record("pass", "merge_context joins non-empty chunks with \\n\\n", repr(result[:60]))


# ── Summary ────────────────────────────────────────────────────────────────

def print_summary() -> bool:
    print(f"\n{'='*55}")
    print("  Summary")
    print(f"{'='*55}")
    passed  = sum(1 for s,_,_ in results if s=="pass")
    failed  = sum(1 for s,_,_ in results if s=="fail")
    skipped = sum(1 for s,_,_ in results if s=="skip")
    print(f"  {PASS} Passed : {passed}")
    if skipped: print(f"  {SKIP} Skipped: {skipped}")
    if failed:
        print(f"  {FAIL} Failed : {failed}")
        for s,n,d in results:
            if s=="fail":
                print(f"    {FAIL}  {n}")
                if d: print(f"       {d}")
    print(f"{'='*55}")
    if failed == 0:
        print("\n  ✅  Phase 2 checkpoint PASSED.")
    else:
        print("\n  ❌  Fix failures above before moving to Phase 3.")
    return failed == 0


if __name__ == "__main__":
    print("="*55)
    print("  Phase 2 — Ingestion & Chunking Tests")
    print("="*55)

    test_extraction()
    test_semantic_chunker_import()
    test_ingest_shape_and_metadata()
    test_doc_dedup()
    test_chunk_dedup()
    test_merge_context()

    ok = print_summary()
    sys.exit(0 if ok else 1)
