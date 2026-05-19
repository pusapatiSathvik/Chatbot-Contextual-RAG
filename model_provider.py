"""
model_provider.py
=================
Unified LLM provider that wraps Ollama, OpenAI, Anthropic, and Gemini
behind a single interface.

Every other module in this project imports `get_llm()` or
`ModelProvider` from here — never a vendor SDK directly.

Public API
----------
    provider = ModelProvider()          # uses settings from config.py
    text     = provider.invoke("Hello") # returns str
    texts    = provider.batch(["Hello", "World"])  # returns List[str]

    # Or get a raw LangChain runnable (for chains, LCEL, etc.)
    llm = get_llm()
"""

from __future__ import annotations

import time
from typing import List, Optional

from config import settings, BackendType


# ---------------------------------------------------------------------------
# Lazy imports — only the SDK for the active backend is imported.
# This prevents import errors when a package isn't installed.
# ---------------------------------------------------------------------------

def _build_ollama(model_id: str, temperature: float):
    from langchain_ollama import OllamaLLM          # type: ignore
    return OllamaLLM(
        model=model_id,
        temperature=temperature,
        base_url=settings.ollama_base_url,
    )


def _build_openai(model_id: str, temperature: float, max_tokens: int):
    import openai as _openai_sdk                    # type: ignore  # noqa: F401
    from langchain_openai import ChatOpenAI          # type: ignore
    return ChatOpenAI(
        model=model_id,
        temperature=temperature,
        max_tokens=max_tokens,
        openai_api_key=settings.openai_api_key,
    )


def _build_anthropic(model_id: str, temperature: float, max_tokens: int):
    from langchain_anthropic import ChatAnthropic    # type: ignore
    return ChatAnthropic(
        model=model_id,
        temperature=temperature,
        max_tokens=max_tokens,
        anthropic_api_key=settings.anthropic_api_key,
    )


def _build_gemini(model_id: str, temperature: float, max_tokens: int):
    from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore
    return ChatGoogleGenerativeAI(
        model=model_id,
        temperature=temperature,
        max_output_tokens=max_tokens,
        google_api_key=settings.google_api_key,
    )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_llm(
    backend: Optional[BackendType] = None,
    model_id: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
):
    """
    Return a LangChain-compatible LLM/Chat model for the configured backend.

    Parameters override config.py values when supplied.
    The returned object supports `.invoke(str) -> str|AIMessage`.
    """
    backend     = backend     or settings.model_backend
    model_id    = model_id    or settings.model_id
    temperature = temperature if temperature is not None else settings.temperature
    max_tokens  = max_tokens  or settings.max_tokens

    builders = {
        "ollama":    lambda: _build_ollama(model_id, temperature),
        "openai":    lambda: _build_openai(model_id, temperature, max_tokens),
        "anthropic": lambda: _build_anthropic(model_id, temperature, max_tokens),
        "gemini":    lambda: _build_gemini(model_id, temperature, max_tokens),
    }

    if backend not in builders:
        raise ValueError(f"Unknown backend: {backend!r}")

    return builders[backend]()


# ---------------------------------------------------------------------------
# ModelProvider  —  the main class every module uses
# ---------------------------------------------------------------------------

class ModelProvider:
    """
    Wraps any LangChain LLM/Chat model and exposes a uniform interface:

        provider.invoke(prompt: str) -> str
        provider.batch(prompts: List[str]) -> List[str]
        provider.generate(prompts: List[str]) -> List[str]   # alias for batch
    """

    def __init__(
        self,
        backend: Optional[BackendType] = None,
        model_id: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> None:
        self._backend  = backend  or settings.model_backend
        self._model_id = model_id or settings.model_id
        self._llm = get_llm(backend, model_id, temperature, max_tokens)
        print(
            f"[ModelProvider] backend={self._backend!r}  "
            f"model={self._model_id!r}"
        )

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def invoke(self, prompt: str) -> str:
        """
        Send a single prompt and return the response as a plain string.
        Works with both LLM (OllamaLLM) and Chat (ChatOpenAI etc.) models.
        """
        t0 = time.time()
        raw = self._llm.invoke(prompt)
        elapsed = time.time() - t0
        text = self._extract_text(raw)
        print(f"[ModelProvider] invoke took {elapsed:.1f}s  ({len(text)} chars)")
        return text

    def batch(self, prompts: List[str]) -> List[str]:
        """
        Send multiple prompts in one call (uses LangChain batch for efficiency).
        Returns a list of plain strings in the same order.
        """
        t0 = time.time()
        raws = self._llm.batch(prompts)
        elapsed = time.time() - t0
        texts = [self._extract_text(r) for r in raws]
        print(f"[ModelProvider] batch({len(prompts)}) took {elapsed:.1f}s")
        return texts

    # Alias used by legacy code that called ollama.generate()
    def generate(self, prompts: List[str]) -> List[str]:
        return self.batch(prompts)

    # ------------------------------------------------------------------
    # Compatibility shim — lets ModelProvider be passed where a raw
    # LangChain LLM is expected (e.g. RAGAS LangchainLLMWrapper)
    # ------------------------------------------------------------------

    @property
    def langchain_llm(self):
        """Return the underlying LangChain model object."""
        return self._llm

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_text(raw) -> str:
        """
        Normalise output from any LangChain model type to a plain string.

        - OllamaLLM         → str directly
        - ChatOpenAI        → AIMessage  → .content  (str)
        - ChatAnthropic     → AIMessage  → .content  (str or list)
        - ChatGoogleGenAI   → AIMessage  → .content  (str)
        """
        if isinstance(raw, str):
            return raw.strip()
        # AIMessage and subclasses
        if hasattr(raw, "content"):
            content = raw.content
            if isinstance(content, list):
                # Anthropic sometimes returns list of content blocks
                parts = [
                    block["text"] if isinstance(block, dict) else str(block)
                    for block in content
                ]
                return "".join(parts).strip()
            return str(content).strip()
        return str(raw).strip()

    def __repr__(self) -> str:
        return f"ModelProvider(backend={self._backend!r}, model={self._model_id!r})"


# ---------------------------------------------------------------------------
# Module-level singleton — import this for the default provider
# ---------------------------------------------------------------------------

_default_provider: Optional[ModelProvider] = None


def get_provider(
    backend: Optional[BackendType] = None,
    model_id: Optional[str] = None,
) -> ModelProvider:
    """
    Return the default ModelProvider (cached after first call).
    Pass backend/model_id to override the cached instance.
    """
    global _default_provider
    if _default_provider is None or backend is not None or model_id is not None:
        _default_provider = ModelProvider(backend=backend, model_id=model_id)
    return _default_provider


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== ModelProvider smoke test ===")
    print(settings.summary())
    print()

    provider = ModelProvider()
    prompt = "Say exactly: 'Model provider is working.' and nothing else."
    print(f"Prompt : {prompt}")
    response = provider.invoke(prompt)
    print(f"Response: {response}")
