"""
retrieval.py
============
Hybrid retrieval pipeline with optional Phase 3 advanced strategies.

ENABLE_ADVANCED_RETRIEVAL=false  →  baseline (BM25 + semantic + RRF + FlashRank)
ENABLE_ADVANCED_RETRIEVAL=true   →  all strategies active:
    1. Query rewriting  — LLM expands the query before retrieval
    2. Multi-query      — 3 query variations, results merged + deduped
    3. HyDE             — hypothetical answer embedded instead of raw query
    4. MMR              — diversity filter on final top-k
    5. Adaptive-k       — expands candidate pool if top scores are weak
"""

from __future__ import annotations

import numpy as np
from flashrank import Ranker, RerankRequest
from contextual_vector_db import ContextualVectorDB
from bm25 import BM25Retriever
from typing import List, Dict, Any, Tuple

from config import settings
from model_provider import get_provider


# ── Strategy 1: Query rewriting ─────────────────────────────────────────────

def rewrite_query(query: str) -> str:
    """Rewrite the query to be more retrieval-friendly."""
    prompt = (
        "You are a search query optimisation assistant.\n"
        "Rewrite the following query to improve document retrieval. "
        "Expand abbreviations, add relevant synonyms, and make implicit "
        "context explicit. Output ONLY the rewritten query — no explanation, "
        "no preamble, no quotes.\n\n"
        f"Original query: {query}\n"
        "Rewritten query:"
    )
    try:
        rewritten = get_provider().invoke(prompt).strip()
        if not rewritten or len(rewritten) > 500:
            return query
        print(f"[rewrite] '{query[:60]}' → '{rewritten[:80]}'")
        return rewritten
    except Exception as e:
        print(f"[rewrite] failed ({e}), using original")
        return query


# ── Strategy 2: Multi-query ──────────────────────────────────────────────────

def generate_query_variations(query: str, n: int) -> List[str]:
    """
    Generate n alternative phrasings of the query.
    Returns the orignal + n variations (deduped).
    """
    prompt = (
        f"Generate {n} different ways to ask the following question for "
        f"document retrieval. Each variation should approach the topic "
        f"from a slightly different angle. "
        f"Output ONLY the variations, one per line, no numbering, no explanation.\n\n"
        f"Question: {query}\n"
        f"Variations:"
    )
    try:
        raw = get_provider().invoke(prompt).strip()
        variations = [ln.strip() for ln in raw.splitlines() if ln.strip()][:n]
        all_queries = list(dict.fromkeys([query] + variations))  # dedup, preserve order
        print(f"[multi-query] generated {len(all_queries)} queries")
        return all_queries
    except Exception as e:
        print(f"[multi-query] failed ({e}), using original only")
        return [query]


# ── Strategy 3: HyDE ────────────────────────────────────────────────────────

def generate_hypothetical_answer(query: str) -> str:
    """
    Generate a hypothetical ideal document passage that would answer the query.
    This is embedded instead of (or alongside) the raw query.
    The intuition: a hypothetical answer lives in the same vector space as
    real document chunks, so it retrieves closer neighbours than a short query.
    """
    prompt = (
        "Write a short document passage (2-4 sentences) that would perfectly "
        "answer the following question. Write it as factual content, not as an "
        "answer to someone. Output ONLY the passage.\n\n"
        f"Question: {query}\n"
        "Passage:"
    )
    try:
        passage = get_provider().invoke(prompt).strip()
        if not passage:
            return query
        print(f"[HyDE] generated hypothetical passage ({len(passage)} chars)")
        return passage
    except Exception as e:
        print(f"[HyDE] failed ({e}), using original query")
        return query


# ── Strategy 4: MMR ─────────────────────────────────────────────────────────

def apply_mmr(
    query_embedding: np.ndarray,
    candidate_ids: List[tuple],
    chunk_data: Dict[tuple, Dict[str, Any]],
    embedding_fn,
    k: int,
    lambda_param: float,
) -> List[tuple]:
    """
    Maximal Marginal Relevance — balances relevance vs diversity.

    lambda_param=1.0 → pure relevance (same as no MMR)
    lambda_param=0.0 → pure diversity
    lambda_param=0.5 → balanced (default)

    Iteratively picks the next chunk that maximises:
        lambda * similarity(chunk, query) - (1-lambda) * max_similarity(chunk, selected)
    """
    if not candidate_ids:
        return []

    # Build embedding matrix for all candidates
    texts = [
        chunk_data[c_id].get("original_content", "")
        for c_id in candidate_ids
        if c_id in chunk_data
    ]
    valid_ids = [c_id for c_id in candidate_ids if c_id in chunk_data]

    if not texts:
        return candidate_ids[:k]

    try:
        embeddings = np.array(embedding_fn(texts))  # (n, dim)
        # Normalise
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        embeddings = embeddings / norms

        query_norm = np.linalg.norm(query_embedding)
        q = query_embedding / (query_norm if query_norm > 0 else 1)

        relevance_scores = embeddings @ q          # (n,)
        selected_indices: List[int] = []
        remaining = list(range(len(valid_ids)))

        while len(selected_indices) < k and remaining:
            if not selected_indices:
                # First pick: highest relevance
                best = max(remaining, key=lambda i: relevance_scores[i])
            else:
                # MMR score for each remaining candidate
                selected_embs = embeddings[selected_indices]            # (s, dim)
                best_score = -float("inf")
                best = remaining[0]
                for i in remaining:
                    rel  = lambda_param * relevance_scores[i]
                    red  = (1 - lambda_param) * float(np.max(embeddings[i] @ selected_embs.T))
                    mmr  = rel - red
                    if mmr > best_score:
                        best_score = mmr
                        best = i
            selected_indices.append(best)
            remaining.remove(best)

        return [valid_ids[i] for i in selected_indices]
    except Exception as e:
        print(f"[MMR] failed ({e}), returning top-k by order")
        return valid_ids[:k]


# ── Shared helpers ───────────────────────────────────────────────────────────

def _run_hybrid_retrieval(
    query: str,
    db: ContextualVectorDB,
    bm25: BM25Retriever,
    num_to_recall: int,
    semantic_weight: float,
    bm25_weight: float,
) -> Tuple[List[tuple], List[tuple], Dict[tuple, Dict[str, Any]]]:
    """Run semantic + BM25 search, return ranked ID lists and chunk lookup."""
    semantic_results = db.search(query, k=num_to_recall)
    ranked_semantic = [
        (r["metadata"]["doc_id"], r["metadata"]["original_index"])
        for r in semantic_results
    ]

    bm25_results = bm25.search(query, k=num_to_recall)
    ranked_bm25 = [
        (r["doc_id"], r["original_index"])
        for r in bm25_results
    ]

    chunk_data: Dict[tuple, Dict[str, Any]] = {}
    for r in semantic_results:
        c_id = (r["metadata"]["doc_id"], r["metadata"]["original_index"])
        chunk_data[c_id] = r["metadata"]
    for r in bm25_results:
        c_id = (r["doc_id"], r["original_index"])
        if c_id not in chunk_data:
            match = next(
                (m for m in db.metadata["metadatas"]
                 if m["doc_id"] == c_id[0] and m["original_index"] == c_id[1]),
                None,
            )
            if match:
                chunk_data[c_id] = match

    return ranked_semantic, ranked_bm25, chunk_data


def _rrf_fuse(
    ranked_semantic: List[tuple],
    ranked_bm25: List[tuple],
    semantic_weight: float,
    bm25_weight: float,
    top_n: int,
) -> List[tuple]:
    """Reciprocal Rank Fusion."""
    all_ids = list(set(ranked_semantic + ranked_bm25))
    scores: Dict[tuple, float] = {}
    for c_id in all_ids:
        s = 0.0
        if c_id in ranked_semantic:
            s += semantic_weight * (1.0 / (ranked_semantic.index(c_id) + 1))
        if c_id in ranked_bm25:
            s += bm25_weight * (1.0 / (ranked_bm25.index(c_id) + 1))
        scores[c_id] = s
    return sorted(scores.keys(), key=lambda x: (scores[x], x[0], x[1]), reverse=True)[:top_n]


def _flashrank(
    query: str,
    top_chunk_ids: List[tuple],
    chunk_data: Dict[tuple, Dict[str, Any]],
) -> List[Dict]:
    """Run FlashRank cross-encoder reranking."""
    passages = []
    for idx, c_id in enumerate(top_chunk_ids):
        if c_id not in chunk_data:
            continue
        meta = chunk_data[c_id]
        text = (
            meta.get("original_content", "")
            + "\n\nContext: "
            + meta.get("contextualized_content", "")
        )
        passages.append({"id": idx, "text": text})

    if not passages:
        return []

    ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="/opt")
    reranked = ranker.rerank(RerankRequest(query=query, passages=passages))
    return sorted(reranked, key=lambda x: x["score"], reverse=True)


# ── Public entry point ───────────────────────────────────────────────────────

def retrieve_advanced(
    query: str,
    db: ContextualVectorDB,
    bm25: BM25Retriever,
    k: int,
    semantic_weight: float = 0.8,
    bm25_weight: float = 0.2,
) -> Tuple[List[Dict[str, Any]], float, float]:
    """
    Hybrid retrieval: semantic + BM25 + RRF + FlashRank reranking.

    When ENABLE_ADVANCED_RETRIEVAL=true, additionally runs:
        query rewriting → multi-query → HyDE → MMR → adaptive-k

    Returns:
        final_results  : top-k chunks with metadata and scores
        semantic_count : weighted count sourced from semantic search
        bm25_count     : weighted count sourced from BM25
    """
    adv = settings.enable_advanced_retrieval
    num_to_recall = k * 10

    # ── Advanced: query rewriting ─────────────────────────────────────────
    active_query = rewrite_query(query) if adv else query

    # ── Advanced: multi-query retrieval ──────────────────────────────────
    if adv:
        queries = generate_query_variations(active_query, settings.multi_query_count)
    else:
        queries = [active_query]

    # ── Advanced: HyDE — embed hypothetical answer alongside query ────────
    if adv:
        hyde_passage = generate_hypothetical_answer(active_query)
        # Use the HyDE passage as the semantic search query
        semantic_query = hyde_passage
    else:
        semantic_query = active_query

    # ── Hybrid retrieval across all queries ───────────────────────────────
    # Merge results from every query variation, deduplicate by chunk id
    all_ranked_semantic: List[tuple] = []
    all_ranked_bm25: List[tuple] = []
    merged_chunk_data: Dict[tuple, Dict[str, Any]] = {}

    for q in queries:
        # For semantic search, primary query uses HyDE passage (if adv);
        # variation queries use their own text
        sem_q = semantic_query if q == active_query else q
        ranked_sem, ranked_bm25, chunk_data = _run_hybrid_retrieval(
            sem_q, db, bm25, num_to_recall, semantic_weight, bm25_weight
        )
        # Merge: preserve first-seen rank position
        for c_id in ranked_sem:
            if c_id not in all_ranked_semantic:
                all_ranked_semantic.append(c_id)
        for c_id in ranked_bm25:
            if c_id not in all_ranked_bm25:
                all_ranked_bm25.append(c_id)
        merged_chunk_data.update(chunk_data)

    # ── RRF fusion ────────────────────────────────────────────────────────
    top_chunk_ids = _rrf_fuse(
        all_ranked_semantic, all_ranked_bm25,
        semantic_weight, bm25_weight, num_to_recall
    )

    # ── Advanced: adaptive-k — expand if top reranker scores are weak ─────
    if adv:
        # Quick pre-score on top-k*2 to check quality
        probe_ids = top_chunk_ids[:k * 2]
        probe_results = _flashrank(active_query, probe_ids, merged_chunk_data)
        if probe_results:
            best_score = probe_results[0]["score"]
            if best_score < settings.adaptive_k_threshold:
                expanded = min(settings.adaptive_k_max * 10, len(top_chunk_ids))
                top_chunk_ids = top_chunk_ids[:expanded]
                print(
                    f"[adaptive-k] best score {best_score:.3f} < "
                    f"{settings.adaptive_k_threshold}, expanded pool to {expanded}"
                )

    # ── FlashRank reranking ───────────────────────────────────────────────
    reranked_sorted = _flashrank(active_query, top_chunk_ids, merged_chunk_data)

    # ── Advanced: MMR diversity filter ───────────────────────────────────
    if adv and reranked_sorted:
        # Get candidate ids in reranked order
        reranked_ids = []
        for res in reranked_sorted:
            idx = res["id"]
            if idx < len(top_chunk_ids):
                reranked_ids.append(top_chunk_ids[idx])

        # Get query embedding for MMR
        query_embedding = np.array(db.embedding_fn([active_query])[0])

        top_k_ids = apply_mmr(
            query_embedding,
            reranked_ids,
            merged_chunk_data,
            db.embedding_fn,
            k,
            settings.mmr_lambda,
        )

        # Rebuild final_results from MMR-selected ids
        final_results: List[Dict[str, Any]] = []
        semantic_count = 0.0
        bm25_count = 0.0

        for c_id in top_k_ids:
            chunk_metadata = merged_chunk_data.get(c_id, {})
            is_sem = c_id in all_ranked_semantic
            is_bm25 = c_id in all_ranked_bm25
            if is_sem and not is_bm25:
                semantic_count += 1
            elif is_bm25 and not is_sem:
                bm25_count += 1
            else:
                semantic_count += 0.5
                bm25_count += 0.5
            final_results.append({
                "chunk":         chunk_metadata,
                "score":         1.0,   # MMR reorders, score is positional
                "from_semantic": is_sem,
                "from_bm25":     is_bm25,
                "query_used":    active_query,
            })
        return final_results, semantic_count, bm25_count

    # ── Baseline assembly (no MMR) ────────────────────────────────────────
    final_results = []
    semantic_count = 0.0
    bm25_count = 0.0

    for res in reranked_sorted[:k]:
        original_idx = res["id"]
        if original_idx >= len(top_chunk_ids):
            continue
        chunk_id = top_chunk_ids[original_idx]
        chunk_metadata = merged_chunk_data.get(chunk_id, {})

        is_sem  = chunk_id in all_ranked_semantic
        is_bm25 = chunk_id in all_ranked_bm25

        if is_sem and not is_bm25:
            semantic_count += 1
        elif is_bm25 and not is_sem:
            bm25_count += 1
        else:
            semantic_count += 0.5
            bm25_count += 0.5

        final_results.append({
            "chunk":         chunk_metadata,
            "score":         res["score"],
            "from_semantic": is_sem,
            "from_bm25":     is_bm25,
            "query_used":    active_query,
        })

    return final_results, semantic_count, bm25_count
