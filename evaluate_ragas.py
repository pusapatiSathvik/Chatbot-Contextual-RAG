"""
evaluate_ragas.py
=================
Evaluates your RAG pipeline using the RAGAS framework with fully local models.

Install first:
    pip install ragas datasets

Usage:
    # Quick test — 10 queries, no ground truth needed
    python evaluate_ragas.py --sample 10

    # Full run with all 4 metrics (needs ground truth — LLM generates it)
    python evaluate_ragas.py --sample 20 --full

    # Use a specific DB
    python evaluate_ragas.py --db_name my_contextual_db --sample 15
"""

import json
import random
import argparse
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from typing import List, Dict, Any

import requests
from chromadb import PersistentClient

# RAGAS imports
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from ragas.metrics import context_precision, context_recall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from datasets import Dataset

# LangChain wrappers for local models
from langchain_ollama import OllamaLLM
from langchain_community.embeddings import SentenceTransformerEmbeddings


OLLAMA_URL  = "http://localhost:11434/api/generate"
MODEL_ID    = "llama3.1"
EMBED_MODEL = "all-mpnet-base-v2"   # same model your RAG already uses


# ---------------------------------------------------------------------------
# Local model setup
# ---------------------------------------------------------------------------

def build_ragas_llm() -> LangchainLLMWrapper:
    """Wrap Ollama llama3.1 so RAGAS can use it as a judge."""
    llm = OllamaLLM(model=MODEL_ID, temperature=0.0)
    return LangchainLLMWrapper(llm)


def build_ragas_embeddings() -> LangchainEmbeddingsWrapper:
    """Wrap the same SentenceTransformer your RAG uses — no extra model needed."""
    embeddings = SentenceTransformerEmbeddings(model_name=EMBED_MODEL)
    return LangchainEmbeddingsWrapper(embeddings)


# ---------------------------------------------------------------------------
# Load chunks from ChromaDB
# ---------------------------------------------------------------------------

def load_chunks_from_db(db_name: str) -> List[Dict[str, Any]]:
    db_path = f"./data/{db_name}/chromadb"
    if not Path(db_path).exists():
        raise FileNotFoundError(
            f"ChromaDB not found at {db_path}\n"
            "Make sure you have uploaded PDFs via the app first."
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
# Retrieve context for a query  (mirrors your actual retrieval pipeline)
# ---------------------------------------------------------------------------

def retrieve_context_for_query(
    query: str,
    db_name: str,
    k: int = 5,
) -> List[str]:
    """
    Run your actual retrieval pipeline and return the top-k chunk texts.
    This means RAGAS evaluates the real system, not a mock.
    """
    from contextual_vector_db import ContextualVectorDB
    from bm25 import create_bm25_index
    from retrieval import retrieve_advanced

    # Lazy-load to avoid slow startup when not needed
    if not hasattr(retrieve_context_for_query, "_cache"):
        db = ContextualVectorDB(db_name)
        db.load_data([])   # loads existing data from disk only
        bm25 = create_bm25_index(db)
        retrieve_context_for_query._cache = (db, bm25)

    db, bm25 = retrieve_context_for_query._cache
    results, _, _ = retrieve_advanced(query, db, bm25, k=k)
    return [
        r["chunk"].get("original_content", "").strip()
        for r in results
        if r["chunk"].get("original_content", "").strip()
    ]


# ---------------------------------------------------------------------------
# Generate answer using Ollama  (mirrors your actual inference)
# ---------------------------------------------------------------------------

def generate_answer(query: str, context_chunks: List[str]) -> str:
    context = "\n\n".join(context_chunks)
    prompt  = (
        "You are a helpful assistant. Answer ONLY from the context below.\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\nAnswer:"
    )
    resp = requests.post(
        OLLAMA_URL,
        json={"model": MODEL_ID, "prompt": prompt, "stream": False},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["response"].strip()


# ---------------------------------------------------------------------------
# Generate a reference answer  (needed for context_precision + context_recall)
# ---------------------------------------------------------------------------

def generate_reference_answer(query: str, golden_chunk: str) -> str:
    """
    Ask the LLM to produce a reference answer from the golden chunk directly.
    This becomes the 'ground truth' for context_precision and context_recall.
    """
    prompt = (
        "Read the following text and answer the question as accurately as possible "
        "using only the information provided.\n\n"
        f"Text:\n{golden_chunk}\n\n"
        f"Question: {query}\n"
        "Answer concisely and factually:"
    )
    resp = requests.post(
        OLLAMA_URL,
        json={"model": MODEL_ID, "prompt": prompt, "stream": False},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["response"].strip()


# ---------------------------------------------------------------------------
# Build RAGAS dataset
# ---------------------------------------------------------------------------

def build_ragas_dataset(
    metadatas: List[Dict[str, Any]],
    db_name: str,
    sample: int,
    full_metrics: bool,
) -> Dataset:
    """
    For each sampled chunk:
      1. Generate a question from the chunk (LLM)
      2. Retrieve context using the real pipeline
      3. Generate an answer using the real pipeline
      4. Optionally generate a reference answer for full metrics

    Returns a HuggingFace Dataset ready for ragas.evaluate().
    """
    # Filter out very short chunks
    usable = [m for m in metadatas if len(m.get("original_content", "")) > 100]
    if not usable:
        raise ValueError("No usable chunks found (all too short).")

    random.seed(42)
    sample_meta = random.sample(usable, min(sample, len(usable)))
    print(f"Building evaluation dataset from {len(sample_meta)} chunks…")

    rows = {
        "question":        [],
        "answer":          [],
        "contexts":        [],
        "ground_truth":    [],  # required by RAGAS schema even if empty
    }

    for i, meta in enumerate(sample_meta):
        chunk_text  = meta.get("original_content", "").strip()
        source_file = meta.get("source_file", meta.get("doc_id", "unknown"))
        print(f"  [{i+1}/{len(sample_meta)}] {source_file}  chunk {meta.get('original_index',0)}")

        # Step 1: Generate a question from this chunk
        q_prompt = (
            "Read the following text and generate ONE specific question "
            "that can be answered using ONLY this text. "
            "Output only the question, nothing else.\n\n"
            f"Text:\n{chunk_text}"
        )
        try:
            resp = requests.post(
                OLLAMA_URL,
                json={"model": MODEL_ID, "prompt": q_prompt, "stream": False},
                timeout=120,
            )
            question = resp.json()["response"].strip().strip('"')
        except Exception as e:
            print(f"    WARN: question generation failed: {e}")
            continue

        # Step 2: Retrieve context via your actual pipeline
        try:
            context_chunks = retrieve_context_for_query(question, db_name, k=5)
        except Exception as e:
            print(f"    WARN: retrieval failed: {e}")
            continue

        if not context_chunks:
            print(f"    WARN: no context retrieved, skipping.")
            continue

        # Step 3: Generate answer via your actual pipeline
        try:
            answer = generate_answer(question, context_chunks)
        except Exception as e:
            print(f"    WARN: answer generation failed: {e}")
            continue

        # Step 4: Reference answer (only needed for full metrics)
        if full_metrics:
            try:
                reference = generate_reference_answer(question, chunk_text)
            except Exception as e:
                print(f"    WARN: reference answer failed: {e}")
                reference = chunk_text[:300]   # fall back to chunk text
        else:
            reference = ""   # faithfulness + answer_relevancy don't need this

        rows["question"].append(question)
        rows["answer"].append(answer)
        rows["contexts"].append(context_chunks)
        rows["ground_truth"].append(reference)

    if not rows["question"]:
        raise ValueError("Failed to build any evaluation rows. Check Ollama is running.")

    print(f"\nBuilt {len(rows['question'])} evaluation samples.")
    return Dataset.from_dict(rows)


# ---------------------------------------------------------------------------
# Run RAGAS evaluation
# ---------------------------------------------------------------------------

def run_ragas_evaluation(
    db_name: str = "my_contextual_db",
    sample: int = 10,
    full_metrics: bool = False,
    output_path: str = "data/ragas_results.json",
) -> Dict[str, Any]:

    print("Setting up local models for RAGAS…")
    ragas_llm    = build_ragas_llm()
    ragas_embeds = build_ragas_embeddings()

    # Pick which metrics to run
    if full_metrics:
        metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
        print("Running FULL evaluation: faithfulness + answer_relevancy + context_precision + context_recall")
        print("NOTE: context_precision and context_recall use LLM-generated reference answers.")
    else:
        metrics = [faithfulness, answer_relevancy]
        print("Running FAST evaluation: faithfulness + answer_relevancy (no ground truth needed)")

    # Attach local models to each metric
    for metric in metrics:
        metric.llm = ragas_llm
        if hasattr(metric, "embeddings"):
            metric.embeddings = ragas_embeds

    # Load data and build dataset
    print(f"\nLoading ChromaDB '{db_name}'…")
    metadatas = load_chunks_from_db(db_name)

    print(f"\nGenerating {sample} test cases (this calls Ollama for each chunk)…")
    dataset = build_ragas_dataset(metadatas, db_name, sample, full_metrics)

    # Run RAGAS
    print("\nRunning RAGAS evaluation…")
    result = evaluate(dataset, metrics=metrics)

    # Format output
    scores = {}
    for metric in metrics:
        key = metric.name
        val = result.get(key)
        if val is not None:
            scores[key] = round(float(val), 4)

    output = {
        "db_name":       db_name,
        "samples":       len(dataset),
        "metrics_used":  [m.name for m in metrics],
        "scores":        scores,
    }

    # Pretty print
    print(f"\n{'='*45}")
    print(f"  RAGAS Evaluation Results")
    print(f"{'='*45}")
    print(f"  Samples evaluated : {len(dataset)}")
    print(f"  Model used        : {MODEL_ID}")
    print()

    score_labels = {
        "faithfulness":     ("Faithfulness",      "Is the answer grounded in retrieved context?"),
        "answer_relevancy": ("Answer Relevancy",   "Does the answer address the question?"),
        "context_precision":("Context Precision",  "Are retrieved chunks actually useful?"),
        "context_recall":   ("Context Recall",     "Did retrieval find all the needed info?"),
    }

    for key, val in scores.items():
        label, desc = score_labels.get(key, (key, ""))
        bar_len = int(val * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(f"  {label:<22} {val:.4f}  [{bar}]")
        print(f"  {'':22} {desc}")
        print()

    print(f"{'='*45}")
    print()

    # Interpretation guide
    print("  Score guide (all metrics are 0.0 → 1.0):")
    print("  > 0.80  Excellent")
    print("  > 0.60  Good")
    print("  > 0.40  Needs improvement")
    print("  < 0.40  Something is broken")
    print()

    # Save to file
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
        description="Evaluate your RAG pipeline with RAGAS using fully local models."
    )
    parser.add_argument("--db_name", default="my_contextual_db",
                        help="ChromaDB collection name (default: my_contextual_db)")
    parser.add_argument("--sample",  type=int, default=10,
                        help="Number of chunks to sample for evaluation (default: 10)")
    parser.add_argument("--full",    action="store_true",
                        help="Run all 4 metrics including context_precision and context_recall "
                             "(slower, uses LLM-generated reference answers)")
    parser.add_argument("--output",  default="data/ragas_results.json",
                        help="Where to save results JSON (default: data/ragas_results.json)")
    args = parser.parse_args()

    run_ragas_evaluation(
        db_name=args.db_name,
        sample=args.sample,
        full_metrics=args.full,
        output_path=args.output,
    )
