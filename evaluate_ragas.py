"""
evaluate_ragas.py
=================
Evaluates your RAG pipeline using the RAGAS framework with fully local models.

Install first:
    pip install ragas datasets langchain-ollama langchain-community

Usage:
    # Quick test — 10 queries, no ground truth needed
    python evaluate_ragas.py --sample 10

    # Full run with all 4 metrics (LLM generates reference answers)
    python evaluate_ragas.py --sample 20 --full

    # Custom DB name
    python evaluate_ragas.py --db_name my_contextual_db --sample 15
"""

import json
import random
import argparse
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from typing import List, Dict, Any

from chromadb import PersistentClient

# RAGAS imports
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from ragas.metrics import context_precision, context_recall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
from datasets import Dataset

# LangChain wrappers
# ChatOllama is required for RAGAS — it uses chat-style prompt templates internally.
# OllamaLLM (completion style) causes template errors inside RAGAS metrics.
from langchain_ollama import ChatOllama, OllamaLLM
from langchain_community.embeddings import SentenceTransformerEmbeddings


MODEL_ID    = "llama3.1"
EMBED_MODEL = "all-mpnet-base-v2"   # same model your RAG already uses

# Shared OllamaLLM instance — used for question/answer/reference generation.
# Same pattern as inference_by_Ollama.py which is confirmed working.
_ollama = OllamaLLM(model=MODEL_ID, temperature=0.0)


# ---------------------------------------------------------------------------
# Local model setup
# ---------------------------------------------------------------------------

def build_ragas_llm():
    """
    Wrap Ollama llama3.1 so RAGAS can use it as a judge.
    Uses ChatOllama (not OllamaLLM) because RAGAS metrics use
    chat-style message templates internally.
    """
    # format="json" forces llama3.1 to return strict JSON
    # which is required for RAGAS prompt parsers to work correctly
    chat_model = ChatOllama(model=MODEL_ID, temperature=0.0, format="json")
    return LangchainLLMWrapper(chat_model)


def build_ragas_embeddings():
    """
    Wrap the same SentenceTransformer your RAG already uses.
    No extra model download needed.
    """
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
# Retrieve context  (uses your actual pipeline — not a mock)
# ---------------------------------------------------------------------------

def retrieve_context_for_query(
    query: str,
    db_name: str,
    k: int = 5,
) -> List[str]:
    """
    Runs the real hybrid retrieval pipeline so RAGAS scores
    reflect actual system performance.
    Caches the DB + BM25 index after first call.
    """
    from contextual_vector_db import ContextualVectorDB
    from bm25 import create_bm25_index
    from retrieval import retrieve_advanced

    if not hasattr(retrieve_context_for_query, "_cache"):
        db = ContextualVectorDB(db_name)
        db.load_data([])   # loads existing data from disk, no re-embedding
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
# Answer generation  (mirrors your actual /chat endpoint)
# ---------------------------------------------------------------------------

def generate_answer(query: str, context_chunks: List[str]) -> str:
    context = "\n\n".join(context_chunks)
    prompt  = (
        "You are a helpful assistant. Answer ONLY from the context below.\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\nAnswer:"
    )
    return _ollama.invoke(prompt).strip()


# ---------------------------------------------------------------------------
# Reference answer generation  (ground truth for full metrics)
# ---------------------------------------------------------------------------

def generate_reference_answer(query: str, golden_chunk: str) -> str:
    """
    Generates a reference answer directly from the golden chunk.
    Used as ground truth for context_precision and context_recall.
    The LLM answers from the chunk itself — not from retrieved context —
    so this is the most faithful possible answer for that question.
    """
    prompt = (
        "Read the following text and answer the question as accurately as possible "
        "using only the information provided.\n\n"
        f"Text:\n{golden_chunk}\n\n"
        f"Question: {query}\n"
        "Answer concisely and factually:"
    )
    return _ollama.invoke(prompt).strip()


# ---------------------------------------------------------------------------
# Build RAGAS-compatible dataset
# ---------------------------------------------------------------------------

def build_ragas_dataset(
    metadatas: List[Dict[str, Any]],
    db_name: str,
    sample: int,
    full_metrics: bool,
) -> Dataset:
    """
    For each sampled chunk:
      1. LLM generates a question answerable from that chunk
      2. Real retrieval pipeline fetches context
      3. LLM generates an answer from retrieved context
      4. (full mode only) LLM generates a reference answer from the chunk itself

    Returns a HuggingFace Dataset with columns:
      question, answer, contexts, ground_truth
    """
    # Skip chunks that are too short to generate a meaningful question
    usable = [m for m in metadatas if len(m.get("original_content", "")) > 100]
    if not usable:
        raise ValueError("No usable chunks found (all too short).")

    random.seed(42)
    sample_meta = random.sample(usable, min(sample, len(usable)))
    print(f"Building evaluation dataset from {len(sample_meta)} chunks…")

    rows: Dict[str, List] = {
        "question":     [],
        "answer":       [],
        "contexts":     [],
        "ground_truth": [],
    }

    for i, meta in enumerate(sample_meta):
        chunk_text  = meta.get("original_content", "").strip()
        source_file = meta.get("source_file", meta.get("doc_id", "unknown"))
        chunk_idx   = meta.get("original_index", 0)
        print(f"  [{i+1}/{len(sample_meta)}] {source_file}  chunk {chunk_idx}")

        # Step 1: generate a question from this chunk
        q_prompt = (
            "Read the following text and generate ONE specific question "
            "that can be answered using ONLY this text. "
            "Output only the question, nothing else.\n\n"
            f"Text:\n{chunk_text}"
        )
        try:
            question = _ollama.invoke(q_prompt).strip().strip('"')
        except Exception as e:
            print(f"    WARN: question generation failed: {e}")
            continue

        # Step 2: retrieve context via the real pipeline
        try:
            context_chunks = retrieve_context_for_query(question, db_name, k=5)
        except Exception as e:
            print(f"    WARN: retrieval failed: {e}")
            continue

        if not context_chunks:
            print(f"    WARN: no context retrieved — skipping.")
            continue

        # Step 3: generate answer from retrieved context
        try:
            answer = generate_answer(question, context_chunks)
        except Exception as e:
            print(f"    WARN: answer generation failed: {e}")
            continue

        # Step 4: reference answer (only for full metrics)
        if full_metrics:
            try:
                reference = generate_reference_answer(question, chunk_text)
            except Exception as e:
                print(f"    WARN: reference answer failed — using chunk text: {e}")
                reference = chunk_text[:300]
        else:
            # faithfulness + answer_relevancy do not need ground_truth
            reference = ""

        rows["question"].append(question)
        rows["answer"].append(answer)
        rows["contexts"].append(context_chunks)
        rows["ground_truth"].append(reference)

    if not rows["question"]:
        raise ValueError(
            "No evaluation rows were built. "
            "Check that Ollama is running and the DB has documents."
        )

    print(f"\nBuilt {len(rows['question'])} evaluation samples.")
    return Dataset.from_dict(rows)


# ---------------------------------------------------------------------------
# Main evaluation runner
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

    if full_metrics:
        metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
        print("Mode: FULL — faithfulness + answer_relevancy + context_precision + context_recall")
        print("      context_precision and context_recall use LLM-generated reference answers.")
    else:
        metrics = [faithfulness, answer_relevancy]
        print("Mode: FAST — faithfulness + answer_relevancy (no ground truth needed)")


    print(f"\nLoading ChromaDB '{db_name}'…")
    metadatas = load_chunks_from_db(db_name)

    print(f"\nGenerating {sample} test cases…")
    dataset = build_ragas_dataset(metadatas, db_name, sample, full_metrics)

    print("\nRunning RAGAS evaluation…")
    result = evaluate(
        dataset,
        metrics=metrics,
        llm=ragas_llm,
        embeddings=ragas_embeds,
        raise_exceptions=False,
        run_config=RunConfig(max_workers=1, timeout=600),
    )

    # Collect scores
    # EvaluationResult is accessed via index in newer RAGAS versions
    # NaN means the LLM judge failed to parse — we detect and report it
    import math
    scores: Dict[str, float] = {}
    for metric in metrics:
        try:
            val = result[metric.name]
            if val is None:
                print(f"  WARN: {metric.name} returned None — skipping.")
                continue
            fval = float(val)
            if math.isnan(fval):
                print(f"  WARN: {metric.name} returned NaN — LLM failed to parse RAGAS output.")
                continue
            scores[metric.name] = round(fval, 4)
        except Exception as e:
            print(f"  WARN: could not read score for {metric.name}: {e}")

    output = {
        "db_name":      db_name,
        "samples":      len(dataset),
        "metrics_used": [m.name for m in metrics],
        "scores":       scores,
    }

    # Pretty print results
    score_labels = {
        "faithfulness":      ("Faithfulness",       "Is the answer grounded in retrieved context?"),
        "answer_relevancy":  ("Answer Relevancy",    "Does the answer address the question?"),
        "context_precision": ("Context Precision",   "Are retrieved chunks actually useful?"),
        "context_recall":    ("Context Recall",      "Did retrieval find all the needed info?"),
    }

    print(f"\n{'='*48}")
    print(f"  RAGAS Evaluation Results")
    print(f"{'='*48}")
    print(f"  Samples : {len(dataset)}   Model : {MODEL_ID}")
    print()
    for key, val in scores.items():
        label, desc = score_labels.get(key, (key, ""))
        bar = "█" * int(val * 20) + "░" * (20 - int(val * 20))
        print(f"  {label:<22} {val:.4f}  [{bar}]")
        print(f"  {'':22} {desc}")
        print()
    print(f"{'='*48}")
    print()
    print("  Score guide  (all metrics: 0.0 → 1.0)")
    print("  > 0.80  Excellent")
    print("  > 0.60  Good")
    print("  > 0.40  Needs improvement")
    print("  < 0.40  Something is broken")
    print()

    # Save
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
    parser.add_argument(
        "--db_name", default="my_contextual_db",
        help="ChromaDB collection name (default: my_contextual_db)"
    )
    parser.add_argument(
        "--sample", type=int, default=10,
        help="Number of chunks to sample (default: 10)"
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Run all 4 metrics — slower, uses LLM-generated reference answers"
    )
    parser.add_argument(
        "--output", default="data/ragas_results.json",
        help="Output path for results JSON (default: data/ragas_results.json)"
    )
    args = parser.parse_args()

    run_ragas_evaluation(
        db_name=args.db_name,
        sample=args.sample,
        full_metrics=args.full,
        output_path=args.output,
    )
