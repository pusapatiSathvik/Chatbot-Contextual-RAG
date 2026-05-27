"""
config.py
=========
Central configuration loader for the RAG chatbot.
Phase 2 update: adds PDF extraction and semantic chunking settings.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


def _load_dotenv() -> None:
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv
        load_dotenv(env_path, override=False)
    except ImportError:
        with open(env_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value


_load_dotenv()

BackendType = Literal["ollama", "openai", "anthropic", "gemini"]


@dataclass
class Settings:
    # Core model
    model_backend: BackendType = field(
        default_factory=lambda: os.environ.get("MODEL_BACKEND", "ollama").lower()
    )
    model_id: str = field(
        default_factory=lambda: os.environ.get("MODEL_ID", "llama3.1")
    )

    # API keys
    openai_api_key: str = field(
        default_factory=lambda: os.environ.get("OPENAI_API_KEY", "")
    )
    anthropic_api_key: str = field(
        default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY", "")
    )
    google_api_key: str = field(
        default_factory=lambda: os.environ.get("GOOGLE_API_KEY", "")
    )

    # Ollama
    ollama_base_url: str = field(
        default_factory=lambda: os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    )

    # Embedding
    embedding_model: str = field(
        default_factory=lambda: os.environ.get("EMBEDDING_MODEL", "all-mpnet-base-v2")
    )

    # Generation
    temperature: float = field(
        default_factory=lambda: float(os.environ.get("TEMPERATURE", "0.0"))
    )
    max_tokens: int = field(
        default_factory=lambda: int(os.environ.get("MAX_TOKENS", "1024"))
    )

    # Features
    enable_contextual_enrichment: bool = field(
        default_factory=lambda: os.environ.get(
            "ENABLE_CONTEXTUAL_ENRICHMENT", "true"
        ).lower() == "true"
    )

    # Phase 2 — Extraction
    # "pymupdf" = native text (fast, accurate); "ocr" = always Tesseract
    pdf_extractor: str = field(
        default_factory=lambda: os.environ.get("PDF_EXTRACTOR", "pymupdf")
    )
    # Font-size ratio above median body text to be classified as a heading
    heading_font_ratio: float = field(
        default_factory=lambda: float(os.environ.get("HEADING_FONT_RATIO", "1.2"))
    )

    # Phase 2 — Semantic chunking
    # Cosine-similarity drop below this threshold = topic boundary
    semantic_split_threshold: float = field(
        default_factory=lambda: float(os.environ.get("SEMANTIC_SPLIT_THRESHOLD", "0.3"))
    )
    min_chunk_chars: int = field(
        default_factory=lambda: int(os.environ.get("MIN_CHUNK_CHARS", "200"))
    )
    max_chunk_chars: int = field(
        default_factory=lambda: int(os.environ.get("MAX_CHUNK_CHARS", "1500"))
    )
    # Sentences of overlap between consecutive chunks for continuity
    chunk_sentence_overlap: int = field(
        default_factory=lambda: int(os.environ.get("CHUNK_SENTENCE_OVERLAP", "1"))
    )

    def __post_init__(self) -> None:
        valid = ("ollama", "openai", "anthropic", "gemini")
        if self.model_backend not in valid:
            raise ValueError(f"MODEL_BACKEND='{self.model_backend}' not in {valid}")
        if self.model_backend == "openai" and not self.openai_api_key:
            raise ValueError("MODEL_BACKEND=openai requires OPENAI_API_KEY.")
        if self.model_backend == "anthropic" and not self.anthropic_api_key:
            raise ValueError("MODEL_BACKEND=anthropic requires ANTHROPIC_API_KEY.")
        if self.model_backend == "gemini" and not self.google_api_key:
            raise ValueError("MODEL_BACKEND=gemini requires GOOGLE_API_KEY.")

    @property
    def is_local(self) -> bool:
        return self.model_backend == "ollama"

    @property
    def active_api_key(self) -> str:
        return {
            "openai": self.openai_api_key,
            "anthropic": self.anthropic_api_key,
            "gemini": self.google_api_key,
            "ollama": "",
        }.get(self.model_backend, "")

    def summary(self) -> str:
        key_hint = ""
        if not self.is_local:
            k = self.active_api_key
            key_hint = f"  API key   : {'*'*8}{k[-4:] if len(k) > 4 else '(not set)'}\n"
        return (
            f"  Backend   : {self.model_backend}\n"
            f"  Model     : {self.model_id}\n"
            f"  Embed     : {self.embedding_model}\n"
            f"  Temp      : {self.temperature}\n"
            f"  MaxTok    : {self.max_tokens}\n"
            f"{key_hint}"
            f"  Ctx enrich: {self.enable_contextual_enrichment}\n"
            f"  Extractor : {self.pdf_extractor}\n"
            f"  SemThresh : {self.semantic_split_threshold}\n"
            f"  ChunkRange: {self.min_chunk_chars}-{self.max_chunk_chars} chars"
        )


settings = Settings()

if __name__ == "__main__":
    print("Current configuration:")
    print(settings.summary())
