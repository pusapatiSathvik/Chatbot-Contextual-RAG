"""
evaluate_ragas.py
=================
Evaluates your RAG pipeline using the RAGAS framework.

PHASE 1 UPDATE:
- Judge LLM for RAGAS metrics now uses ModelProvider (any backend).
- Ollama requires ChatOllama for RAGAS; API backends use their Chat models.
- Completion LLM for question/answer generation also uses ModelProvider.

Usage:
    python evaluate_ragas.py --sample 10
    python evaluate_ragas.py --sample 20 --full
    python evaluate_ragas.py --backend anthropic --model claude-haiku-4-5 --sample 5
"""

import json
import math
import random
import asyncio
import argparse
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from typing import List, Dict, Any

from chromadb import PersistentClient

from ragas import SingleTurnSample
from ragas.metrics import Faithfulness, ResponseRelevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_community.embeddings import SentenceTransformerEmbeddings

from config import settings, BackendType
from model_provider import ModelProvider


# ---------------------------------------------------------------------------
# Build the RAGAS judge LLM
# RAGAS requires a Chat model (not a completion model) for its internal
# prompt parsing. Ollama uses ChatOllama; API backends use their Chat wrappers.
# ---------------------------------------------------------------------------

def build_ragas_llm(backend: BackendType, model_id: str) -> LangchainLLMWrapper:
    """Return a LangchainLLMWrapper-wrapped Chat model for RAGAS metrics."""
    if backend == "ollama":
        from langchain_ollama import ChatOllama         # type: ignore
        chat_llm = ChatOllama(
            model=model_id,
            temperature=0.0,
            base_url=settings.ollama_base_url,
        )
    elif backend == "openai":
        from langchain_openai import ChatOpenAI         # type: ignore
        chat_llm = ChatOpenAI(
            model=model_id,
            temperature=0.0,
            openai_api_key=settings.openai_api_key,
        )
    elif backend == "anthropic":
        from langchain_anthropic import ChatAnthropic   # type: ignore
        chat_llm = ChatAnthropic(
            model=model_id,
            temperature=0.0,
            anthropic_api_key=settings.anthropic_api_key,
        )
    elif backend == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore
        chat_llm = ChatGoogleGenerativeAI(
            model=model_id,
            temperature=0.0,
            google_api_key=settings.google_api_key,
        )
    else:
        raise ValueError(f"Unknown backend: {backend!r}")

    return LangchainLLMWrapper(chat_llm)


def build_ragas_embeddings() -> LangchainEmbeddingsWrapper:
    """Same SentenceTransformer your RAG already uses — no extra download."""
    return LangchainEmbeddingsWrapper(
        SentenceTransformerEmbeddings(model_name=settings.embedding_model)
    )


# ---------------------------------------------------------------------------
# Load chunks from ChromaDB
# ---------------------------------------------------------------------------

def load_chunks_from_db(db_name: str) -> List[Dict[str, Any]]:
    db_path = f"./data/{db_name}/chromadb"
    if not Path(db_path).exists():
        raise FileNotFoundError(
            f"ChromaDB not found at {db_path}\n"
            "Upload PDFs via the app first."
        )
    client     = PersistentClient(path=db_path)
    collection = client.get_collection(name=db_name)
    data       = collection.get(include=["metadatas"])
    metadatas  = data.get("metadatas", [])
    if not metadatas:
        raise ValueError("ChromaDB is empty. Upload PDFs via the app first.")
    print(f"Loaded {len(metadatas)} chunks from ChromaDB.")
    return metadatas


# ---------------------------------------------------------------------------
# Retrieval (uses your real pipeline)
# ---------------------------------------------------------------------------

def retrieve_context(query: str, db_name: str, k: int = 5) -> List[str]:
    """Retrieve context chunks using the real pipeline. Caches DB + BM25."""
    from contextual_vector_db import ContextualVectorDB
    from bm25 import create_bm25_index
    from retrieval import retrieve_advanced

    if not hasattr(retrieve_context, "_cache"):
        db   = ContextualVectorDB(db_name)
        db.load_data([])
        bm25 = create_bm25_index(db)
        retrieve_context._cache = (db, bm25)

    db, bm25 = retrieve_context._cache
    results, _, _ = retrieve_advanced(query, db, bm25, k=k)
    return [
        r["chunk"].get("original_content", "").strip()
        for r in results
        if r["chunk"].get("original_content", "").strip()
    ]


# ---------------------------------------------------------------------------
# LLM calls for test data generation
# ---------------------------------------------------------------------------

def generate_question(chunk_text: str, provider: ModelProvider) -> str:
    prompt = (
        "Read the following text and generate ONE specific question "
        "answerable ONLY from this text. Output only the question.\n\n"
        f"Text:\n{chunk_text}"
    )
    return provider.invoke(prompt).strip().strip('"')


def generate_answer(
    question: str, context_chunks: List[str], provider: ModelProvider
) -> str:
    context = "\n\n".join(context_chunks)
    prompt  = (
        "You are a helpful assistant. Answer ONLY from the context below.\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\nAnswer:"
    )
    return provider.invoke(prompt).strip()


def generate_reference(
    question: str, chunk_text: str, provider: ModelProvider
) -> str:
    """Ground-truth answer direct from source chunk — for full metrics."""
    prompt = (
        "Read the following text and answer the question accurately "
        "using only the information provided.\n\n"
        f"Text:\n{chunk_text}\n\n"
        f"Question: {question}\nAnswer:"
    )
    return provider.invoke(prompt).strip()


# ---------------------------------------------------------------------------
# Score one sample
# ---------------------------------------------------------------------------

async def score_sample(metric, sample: SingleTurnSample) -> float:
    try:
        score = await metric.single_turn_ascore(sample)
        return float(score)
    except Exception as e:
        print(f"    WARN: scoring failed ({type(e).__name__}: {str(e)[:80]})")
        return float("nan")


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def run_ragas_evaluation(
    db_name: str       = "my_contextual_db",
    sample: int        = 10,
    full_metrics: bool = False,
    output_path: str   = "data/ragas_results.json",
    backend: BackendType = None,   # type: ignore
    model_id: str      = None,     # type: ignore
) -> Dict[str, Any]:

    # Resolve backend / model — CLI args override config
    backend  = backend  or settings.model_backend
    model_id = model_id or settings.model_id

    print(f"Setting up RAGAS with backend={backend!r}  model={model_id!r}")
    ragas_llm    = build_ragas_llm(backend, model_id)
    ragas_embeds = build_ragas_embeddings()

    # Completion provider for question/answer generation
    gen_provider = ModelProvider(backend=backend, model_id=model_id)

    faithfulness_metric = Faithfulness(llm=ragas_llm)
    relevancy_metric    = ResponseRelevancy(llm=ragas_llm, embeddings=ragas_embeds)

    if full_metrics:
        from ragas.metrics import LLMContextRecall, LLMContextPrecisionWithReference
        metrics = {
            "faithfulness":       faithfulness_metric,
            "response_relevancy": relevancy_metric,
            "context_recall":     LLMContextRecall(llm=ragas_llm),
            "context_precision":  LLMContextPrecisionWithReference(llm=ragas_llm),
        }
        print("Mode: FULL — 4 metrics")
    else:
        metrics = {
            "faithfulness":       faithfulness_metric,
            "response_relevancy": relevancy_metric,
        }
        print("Mode: FAST — faithfulness + response_relevancy")

    print(f"\nLoading ChromaDB '{db_name}'…")
    metadatas = load_chunks_from_db(db_name)

    usable = [m for m in metadatas if len(m.get("original_content", "")) > 100]
    if not usable:
        raise ValueError("No usable chunks (all too short).")
    random.seed(42)
    sampled = random.sample(usable, min(sample, len(usable)))
    print(f"\nBuilding {len(sampled)} test samples…\n")

    all_scores: Dict[str, List[float]] = {name: [] for name in metrics}

    for i, meta in enumerate(sampled):
        chunk_text  = meta.get("original_content", "").strip()
        source_file = meta.get("source_file", meta.get("doc_id", "unknown"))
        chunk_idx   = meta.get("original_index", 0)
        print(f"[{i+1}/{len(sampled)}] {source_file}  chunk {chunk_idx}")

        try:
            question = generate_question(chunk_text, gen_provider)
            print(f"  Q: {question[:80]}")
        except Exception as e:
            print(f"  WARN: question generation failed: {e}")
            continue

        try:
            contexts = retrieve_context(question, db_name, k=5)
        except Exception as e:
            print(f"  WARN: retrieval failed: {e}")
            continue
        if not contexts:
            print("  WARN: no context retrieved — skipping.")
            continue

        try:
            answer = generate_answer(question, contexts, gen_provider)
        except Exception as e:
            print(f"  WARN: answer generation failed: {e}")
            continue

        reference = (
            generate_reference(question, chunk_text, gen_provider)
            if full_metrics else ""
        )

        sample_obj = SingleTurnSample(
            user_input         = question,
            response           = answer,
            retrieved_contexts = contexts,
            reference          = reference,
        )

        for metric_name, metric in metrics.items():
            score = asyncio.run(score_sample(metric, sample_obj))
            if not math.isnan(score):
                all_scores[metric_name].append(score)
                print(f"  {metric_name}: {score:.4f}")
            else:
                print(f"  {metric_name}: failed to score")
        print()

    # Aggregate
    final_scores: Dict[str, float] = {
        name: round(sum(scores) / len(scores), 4)
        for name, scores in all_scores.items()
        if scores
    }

    output = {
        "db_name":      db_name,
        "backend":      backend,
        "model_id":     model_id,
        "samples":      len(sampled),
        "metrics_used": list(metrics.keys()),
        "scores":       final_scores,
    }

    score_labels = {
        "faithfulness":       ("Faithfulness",        "Is the answer grounded in context?"),
        "response_relevancy": ("Response Relevancy",  "Does the answer address the question?"),
        "context_recall":     ("Context Recall",      "Did retrieval find all needed info?"),
        "context_precision":  ("Context Precision",   "Are retrieved chunks actually useful?"),
    }

    print(f"\n{'='*50}")
    print(f"  RAGAS Evaluation Results")
    print(f"{'='*50}")
    print(f"  Samples  : {len(sampled)}")
    print(f"  Backend  : {backend}  ({model_id})\n")

    if final_scores:
        for key, val in final_scores.items():
            label, desc = score_labels.get(key, (key, ""))
            bar = "█" * int(val * 20) + "░" * (20 - int(val * 20))
            print(f"  {label:<24} {val:.4f}  [{bar}]")
            print(f"  {'':24} {desc}\n")
    else:
        print("  No scores were produced.")

    print(f"{'='*50}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved → {output_path}")

    return output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate RAG pipeline with RAGAS."
    )
    parser.add_argument("--db_name",  default="my_contextual_db")
    parser.add_argument("--sample",   type=int, default=10)
    parser.add_argument("--full",     action="store_true")
    parser.add_argument("--output",   default="data/ragas_results.json")
    parser.add_argument(
        "--backend",
        choices=["ollama", "openai", "anthropic", "gemini"],
        default=None,
        help="Override MODEL_BACKEND from .env",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override MODEL_ID from .env",
    )
    args = parser.parse_args()

    run_ragas_evaluation(
        db_name      = args.db_name,
        sample       = args.sample,
        full_metrics = args.full,
        output_path  = args.output,
        backend      = args.backend,
        model_id     = args.model,
    )
