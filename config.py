"""
config.py
=========
Central configuration loader for the RAG chatbot.

Reads from environment variables (or a .env file in the project root).
All other modules import from here — never read os.environ directly.

Usage:
    from config import settings
    print(settings.model_backend)   # "ollama"
    print(settings.model_id)        # "llama3.1"
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

# ---------------------------------------------------------------------------
# Load .env file if present (pure stdlib — no python-dotenv required,
# but python-dotenv is used when available for richer parsing)
# ---------------------------------------------------------------------------

def _load_dotenv() -> None:
    """Load .env from project root into os.environ (best-effort)."""
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv          # type: ignore
        load_dotenv(env_path, override=False)   # existing env vars take priority
    except ImportError:
        # Fallback: manual parser for simple KEY=VALUE lines
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

# ---------------------------------------------------------------------------
# Supported backends
# ---------------------------------------------------------------------------

BackendType = Literal["ollama", "openai", "anthropic", "gemini"]

# ---------------------------------------------------------------------------
# Settings dataclass
# ---------------------------------------------------------------------------

@dataclass
class Settings:
    # --- Core model config ---
    model_backend: BackendType = field(
        default_factory=lambda: os.environ.get("MODEL_BACKEND", "ollama").lower()  # type: ignore
    )
    model_id: str = field(
        default_factory=lambda: os.environ.get("MODEL_ID", "llama3.1")
    )

    # --- API keys ---
    openai_api_key: str = field(
        default_factory=lambda: os.environ.get("OPENAI_API_KEY", "")
    )
    anthropic_api_key: str = field(
        default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY", "")
    )
    google_api_key: str = field(
        default_factory=lambda: os.environ.get("GOOGLE_API_KEY", "")
    )

    # --- Ollama ---
    ollama_base_url: str = field(
        default_factory=lambda: os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    )

    # --- Embedding ---
    embedding_model: str = field(
        default_factory=lambda: os.environ.get("EMBEDDING_MODEL", "all-mpnet-base-v2")
    )

    # --- Generation ---
    temperature: float = field(
        default_factory=lambda: float(os.environ.get("TEMPERATURE", "0.0"))
    )
    max_tokens: int = field(
        default_factory=lambda: int(os.environ.get("MAX_TOKENS", "1024"))
    )

    # --- Features ---
    enable_contextual_enrichment: bool = field(
        default_factory=lambda: os.environ.get(
            "ENABLE_CONTEXTUAL_ENRICHMENT", "true"
        ).lower() == "true"
    )

    # ---------------------------------------------------------------------------
    # Validation
    # ---------------------------------------------------------------------------

    def __post_init__(self) -> None:
        valid_backends: tuple = ("ollama", "openai", "anthropic", "gemini")
        if self.model_backend not in valid_backends:
            raise ValueError(
                f"MODEL_BACKEND='{self.model_backend}' is not valid. "
                f"Choose one of: {valid_backends}"
            )

        if self.model_backend == "openai" and not self.openai_api_key:
            raise ValueError(
                "MODEL_BACKEND=openai requires OPENAI_API_KEY to be set."
            )
        if self.model_backend == "anthropic" and not self.anthropic_api_key:
            raise ValueError(
                "MODEL_BACKEND=anthropic requires ANTHROPIC_API_KEY to be set."
            )
        if self.model_backend == "gemini" and not self.google_api_key:
            raise ValueError(
                "MODEL_BACKEND=gemini requires GOOGLE_API_KEY to be set."
            )

    # ---------------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------------

    @property
    def is_local(self) -> bool:
        """True when running a local model (no API cost)."""
        return self.model_backend == "ollama"

    @property
    def active_api_key(self) -> str:
        """Returns whichever API key is relevant for the current backend."""
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
            key_hint = f"  API key  : {'*' * 8}{k[-4:] if len(k) > 4 else '(not set)'}\n"
        return (
            f"  Backend  : {self.model_backend}\n"
            f"  Model    : {self.model_id}\n"
            f"  Embed    : {self.embedding_model}\n"
            f"  Temp     : {self.temperature}\n"
            f"  MaxTok   : {self.max_tokens}\n"
            f"{key_hint}"
            f"  Ctx enrich: {self.enable_contextual_enrichment}"
        )


# ---------------------------------------------------------------------------
# Singleton — import this everywhere
# ---------------------------------------------------------------------------

settings = Settings()


if __name__ == "__main__":
    print("Current configuration:")
    print(settings.summary())
