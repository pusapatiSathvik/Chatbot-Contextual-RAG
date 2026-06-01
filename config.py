"""
config.py — Central configuration loader.
Phase 1: model backend + embedding.
Phase 2: PDF extraction + chunking settings.
Phase 3: single flag to enable all advanced retrieval strategies.
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

    # ── Phase 1: Core model ─────────────────────────────────────────────────
    model_backend: BackendType = field(
        default_factory=lambda: os.environ.get("MODEL_BACKEND", "ollama").lower()
    )
    model_id: str = field(
        default_factory=lambda: os.environ.get("MODEL_ID", "llama3.1")
    )
    openai_api_key: str = field(
        default_factory=lambda: os.environ.get("OPENAI_API_KEY", "")
    )
    anthropic_api_key: str = field(
        default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY", "")
    )
    google_api_key: str = field(
        default_factory=lambda: os.environ.get("GOOGLE_API_KEY", "")
    )
    ollama_base_url: str = field(
        default_factory=lambda: os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    )
    embedding_model: str = field(
        default_factory=lambda: os.environ.get("EMBEDDING_MODEL", "all-mpnet-base-v2")
    )
    temperature: float = field(
        default_factory=lambda: float(os.environ.get("TEMPERATURE", "0.0"))
    )
    max_tokens: int = field(
        default_factory=lambda: int(os.environ.get("MAX_TOKENS", "1024"))
    )
    enable_contextual_enrichment: bool = field(
        default_factory=lambda: os.environ.get(
            "ENABLE_CONTEXTUAL_ENRICHMENT", "true"
        ).lower() == "true"
    )

    # ── Phase 2: PDF extraction ─────────────────────────────────────────────
    pdf_extractor: str = field(
        default_factory=lambda: os.environ.get("PDF_EXTRACTOR", "pymupdf")
    )

    # ── Phase 2: Semantic chunking ──────────────────────────────────────────
    min_chunk_chars: int = field(
        default_factory=lambda: int(os.environ.get("MIN_CHUNK_CHARS", "200"))
    )
    max_chunk_chars: int = field(
        default_factory=lambda: int(os.environ.get("MAX_CHUNK_CHARS", "1500"))
    )

    # ── Phase 3: Advanced retrieval ─────────────────────────────────────────
    # Single flag — true enables ALL strategies at once:
    #   query rewriting, multi-query, HyDE, MMR, adaptive-k
    enable_advanced_retrieval: bool = field(
        default_factory=lambda: os.environ.get(
            "ENABLE_ADVANCED_RETRIEVAL", "false"
        ).lower() == "true"
    )

    # Tuning knobs (only matter when enable_advanced_retrieval=true)
    # Number of query variations for multi-query
    multi_query_count: int = field(
        default_factory=lambda: int(os.environ.get("MULTI_QUERY_COUNT", "3"))
    )
    # MMR balance: 1.0 = pure relevance, 0.0 = pure diversity
    mmr_lambda: float = field(
        default_factory=lambda: float(os.environ.get("MMR_LAMBDA", "0.5"))
    )
    # Adaptive-k: if best reranker score < threshold, expand candidate pool
    adaptive_k_threshold: float = field(
        default_factory=lambda: float(os.environ.get("ADAPTIVE_K_THRESHOLD", "0.5"))
    )
    adaptive_k_max: int = field(
        default_factory=lambda: int(os.environ.get("ADAPTIVE_K_MAX", "20"))
    )

    # ── Validation ──────────────────────────────────────────────────────────
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
            key_hint = f"  API key      : {'*'*8}{k[-4:] if len(k) > 4 else '(not set)'}\n"
        return (
            f"  Backend      : {self.model_backend}\n"
            f"  Model        : {self.model_id}\n"
            f"  Embed        : {self.embedding_model}\n"
            f"  Temp         : {self.temperature}\n"
            f"  MaxTok       : {self.max_tokens}\n"
            f"{key_hint}"
            f"  Ctx enrich   : {self.enable_contextual_enrichment}\n"
            f"  Extractor    : {self.pdf_extractor}\n"
            f"  ChunkRange   : {self.min_chunk_chars}-{self.max_chunk_chars} chars\n"
            f"  Adv retrieval: {'ON  (rewrite + multi-query + HyDE + MMR + adaptive-k)' if self.enable_advanced_retrieval else 'off (baseline hybrid)'}"
        )


settings = Settings()

if __name__ == "__main__":
    print("Current configuration:")
    print(settings.summary())
