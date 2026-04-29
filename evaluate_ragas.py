"""
evaluate_ragas.py
=================
Evaluates your RAG pipeline using the RAGAS framework with fully local models.

Uses SingleTurnSample + metric.single_turn_score() instead of evaluate()
because the high-level evaluate() function fails with small local LLMs
(llama3.1 8B) due to JSON parsing issues in its internal executor.

Install:
    pip install ragas datasets langchain-ollama langchain-community

Usage:
    python evaluate_ragas.py --sample 10
    python evaluate_ragas.py --sample 20 --full
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

# RAGAS — low-level API (works reliably with local LLMs)
from ragas import SingleTurnSample
from ragas.metrics import Faithfulness, ResponseRelevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from datasets import Dataset

# LangChain
from langchain_ollama import ChatOllama, OllamaLLM
from langchain_community.embeddings import SentenceTransformerEmbeddings


MODEL_ID    = "llama3.1"
EMBED_MODEL = "all-mpnet-base-v2"

# Shared completion LLM — for question/answer/reference generation
_ollama = OllamaLLM(model=MODEL_ID, temperature=0.0)


# ---------------------------------------------------------------------------
# RAGAS judge model setup
# ---------------------------------------------------------------------------

def build_ragas_llm():
    """
    ChatOllama required for RAGAS judge — it uses chat message format internally.
    No format="json" here — SingleTurnSample API handles parsing itself.
    """
    return LangchainLLMWrapper(ChatOllama(model=MODEL_ID, temperature=0.0))


def build_ragas_embeddings():
    """Same SentenceTransformer your RAG already uses — no extra download."""
    return LangchainEmbeddingsWrapper(
        SentenceTransformerEmbeddings(model_name=EMBED_MODEL)
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
# Retrieval  (uses your real pipeline)
# ---------------------------------------------------------------------------

def retrieve_context(query: str, db_name: str, k: int = 5) -> List[str]:
    """Caches DB + BM25 after first call."""
    from contextual_vector_db import ContextualVectorDB
    from bm25 import create_bm25_index
    from retrieval import retrieve_advanced

    if not hasattr(retrieve_context, "_cache"):
        db = ContextualVectorDB(db_name)
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
# LLM calls using OllamaLLM (same as your working inference_by_Ollama.py)
# ---------------------------------------------------------------------------

def generate_question(chunk_text: str) -> str:
    prompt = (
        "Read the following text and generate ONE specific question "
        "answerable ONLY from this text. Output only the question.\n\n"
        f"Text:\n{chunk_text}"
    )
    return _ollama.invoke(prompt).strip().strip('"')


def generate_answer(question: str, context_chunks: List[str]) -> str:
    context = "\n\n".join(context_chunks)
    prompt  = (
        "You are a helpful assistant. Answer ONLY from the context below.\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\nAnswer:"
    )
    return _ollama.invoke(prompt).strip()


def generate_reference(question: str, chunk_text: str) -> str:
    """Ground-truth answer direct from source chunk — for full metrics."""
    prompt = (
        "Read the following text and answer the question accurately "
        "using only the information provided.\n\n"
        f"Text:\n{chunk_text}\n\n"
        f"Question: {question}\nAnswer:"
    )
    return _ollama.invoke(prompt).strip()


# ---------------------------------------------------------------------------
# Score one sample with a metric  (async, with error handling)
# ---------------------------------------------------------------------------

async def score_sample(metric, sample: SingleTurnSample) -> float:
    """
    Score one sample. Returns float score or NaN on failure.
    Uses single_turn_ascore() — the low-level async API that avoids
    the executor/parser issues in evaluate().
    """
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
    db_name: str = "my_contextual_db",
    sample: int  = 10,
    full_metrics: bool = False,
    output_path: str   = "data/ragas_results.json",
) -> Dict[str, Any]:

    print("Setting up local models for RAGAS…")
    ragas_llm    = build_ragas_llm()
    ragas_embeds = build_ragas_embeddings()

    # Instantiate metrics with local models injected
    faithfulness_metric    = Faithfulness(llm=ragas_llm)
    relevancy_metric       = ResponseRelevancy(llm=ragas_llm, embeddings=ragas_embeds)

    if full_metrics:
        from ragas.metrics import LLMContextRecall, LLMContextPrecisionWithReference
        recall_metric    = LLMContextRecall(llm=ragas_llm)
        precision_metric = LLMContextPrecisionWithReference(llm=ragas_llm)
        metrics = {
            "faithfulness":       faithfulness_metric,
            "response_relevancy": relevancy_metric,
            "context_recall":     recall_metric,
            "context_precision":  precision_metric,
        }
        print("Mode: FULL — faithfulness + response_relevancy + context_recall + context_precision")
    else:
        metrics = {
            "faithfulness":       faithfulness_metric,
            "response_relevancy": relevancy_metric,
        }
        print("Mode: FAST — faithfulness + response_relevancy (no ground truth needed)")

    print(f"\nLoading ChromaDB '{db_name}'…")
    metadatas = load_chunks_from_db(db_name)

    # Sample chunks
    usable = [m for m in metadatas if len(m.get("original_content", "")) > 100]
    if not usable:
        raise ValueError("No usable chunks (all too short).")
    random.seed(42)
    sampled = random.sample(usable, min(sample, len(usable)))
    print(f"\nBuilding {len(sampled)} test samples and scoring each one…\n")

    # Per-metric score accumulators
    all_scores: Dict[str, List[float]] = {name: [] for name in metrics}

    for i, meta in enumerate(sampled):
        chunk_text  = meta.get("original_content", "").strip()
        source_file = meta.get("source_file", meta.get("doc_id", "unknown"))
        chunk_idx   = meta.get("original_index", 0)
        print(f"[{i+1}/{len(sampled)}] {source_file}  chunk {chunk_idx}")

        # Step 1: generate question
        try:
            question = generate_question(chunk_text)
            print(f"  Q: {question[:80]}")
        except Exception as e:
            print(f"  WARN: question generation failed: {e}")
            continue

        # Step 2: retrieve context via real pipeline
        try:
            contexts = retrieve_context(question, db_name, k=5)
        except Exception as e:
            print(f"  WARN: retrieval failed: {e}")
            continue
        if not contexts:
            print(f"  WARN: no context retrieved — skipping.")
            continue

        # Step 3: generate answer from retrieved context
        try:
            answer = generate_answer(question, contexts)
        except Exception as e:
            print(f"  WARN: answer generation failed: {e}")
            continue

        # Step 4: reference answer (only for full metrics)
        reference = generate_reference(question, chunk_text) if full_metrics else ""

        # Step 5: build RAGAS sample and score with each metric
        sample_obj = SingleTurnSample(
            user_input        = question,
            response          = answer,
            retrieved_contexts= contexts,
            reference         = reference,
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
    final_scores: Dict[str, float] = {}
    for name, scores in all_scores.items():
        if scores:
            final_scores[name] = round(sum(scores) / len(scores), 4)

    output = {
        "db_name":      db_name,
        "samples":      len(sampled),
        "metrics_used": list(metrics.keys()),
        "scores":       final_scores,
    }

    # Pretty print
    score_labels = {
        "faithfulness":       ("Faithfulness",        "Is the answer grounded in context?"),
        "response_relevancy": ("Response Relevancy",  "Does the answer address the question?"),
        "context_recall":     ("Context Recall",      "Did retrieval find all needed info?"),
        "context_precision":  ("Context Precision",   "Are retrieved chunks actually useful?"),
    }

    print(f"\n{'='*50}")
    print(f"  RAGAS Evaluation Results")
    print(f"{'='*50}")
    print(f"  Samples : {len(sampled)}   Model : {MODEL_ID}\n")

    if final_scores:
        for key, val in final_scores.items():
            label, desc = score_labels.get(key, (key, ""))
            bar = "█" * int(val * 20) + "░" * (20 - int(val * 20))
            print(f"  {label:<24} {val:.4f}  [{bar}]")
            print(f"  {'':24} {desc}\n")
    else:
        print("  No scores were produced.")
        print("  This usually means llama3.1 failed to follow RAGAS JSON prompts.")
        print("  Try: ollama pull mistral  and set MODEL_ID = 'mistral'")

    print(f"{'='*50}")
    print("\n  Score guide  (0.0 → 1.0)")
    print("  > 0.80  Excellent")
    print("  > 0.60  Good")
    print("  > 0.40  Needs improvement")
    print("  < 0.40  Something is broken\n")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"  Results saved → {output_path}")

    return output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate your RAG pipeline with RAGAS using local Ollama models."
    )
    parser.add_argument("--db_name", default="my_contextual_db")
    parser.add_argument("--sample",  type=int, default=10,
                        help="Number of chunks to sample (default: 10)")
    parser.add_argument("--full",    action="store_true",
                        help="Run all 4 metrics (slower, needs reference answers)")
    parser.add_argument("--output",  default="data/ragas_results.json")
    args = parser.parse_args()

    run_ragas_evaluation(
        db_name=args.db_name,
        sample=args.sample,
        full_metrics=args.full,
        output_path=args.output,
    )
