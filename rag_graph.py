"""
Step 3: Retrieval agent added. Router + Retrieval are real. 
        Grader / Answer / Reflection still stubs.
"""

from typing import TypedDict, List, Any
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END, START
from model_provider import get_provider


# ── State ────────────────────────────────────────────────────────────────────

class RAGState(TypedDict):
    query:          str
    query_type:     str
    retrieved:      List[dict]
    graded:         List[dict]
    answer:         str
    grounded:       bool
    iterations:     int
    semantic_count: float      # how many results came from vector search
    bm25_count:     float      # how many came from BM25


# ── Agent 1: Router (unchanged from Step 2) ──────────────────────────────────

ROUTER_PROMPT = """You are a query classifier for a document retrieval system.

Classify the following query into exactly ONE of these three categories:
  factual      — answerable from a single document passage
  multi_hop    — requires connecting information across multiple passages
  out_of_scope — cannot be answered from stored documents (weather, news, personal advice, etc.)

Reply with ONLY the single category word. No explanation, no punctuation.

Query: {query}
Category:"""


def router_node(state: RAGState) -> dict:
    prompt = ROUTER_PROMPT.format(query=state["query"])
    try:
        raw = get_provider().invoke(prompt).strip().lower()
        first_word = raw.split()[0] if raw else "factual"
        query_type = first_word if first_word in ("factual", "multi_hop", "out_of_scope") else "factual"
    except Exception as e:
        print(f"[router] error ({e}), defaulting to factual")
        query_type = "factual"
    print(f"[router] '{state['query'][:60]}' → {query_type}")
    return {"query_type": query_type, "iterations": 0}


# ── Agent 2: Retrieval ────────────────────────────────────────────────────────
#
# WHY a separate node:
#   The reflection agent can loop back here. On retry we expand the query
#   so the retriever surfaces different chunks instead of the same ones again.
#
# HOW db/bm25 are passed in:
#   LangGraph lets you pass runtime objects via config["configurable"].
#   This keeps the node a pure function — no global state, fully testable.

def retrieval_node(state: RAGState, config: RunnableConfig) -> dict:
    """
    Runs Phase 3 hybrid retrieval for the current query.
    On retry (iterations > 0) appends extra context hint to query
    so the retriever surfaces different chunks.
    """
    db   = config["configurable"]["db"]
    bm25 = config["configurable"]["bm25"]

    query     = state["query"]
    iteration = state.get("iterations", 0)

    # On retry: nudge query to surface different chunks
    if iteration > 0:
        query = query + " provide more detail and additional context"
        print(f"[retrieval] retry {iteration} with expanded query")

    from retrieval import retrieve_advanced
    results, sem_count, bm25_count = retrieve_advanced(query, db, bm25, k=10)

    print(f"[retrieval] {len(results)} chunks  sem={sem_count:.1f}  bm25={bm25_count:.1f}")
    return {
        "retrieved":      results,
        "semantic_count": sem_count,
        "bm25_count":     bm25_count,
        "iterations":     iteration + 1,
    }


# ── Stub nodes ────────────────────────────────────────────────────────────────

GRADER_PROMPT = """You are grading whether a document chunk is relevant to a question.

Question: {query}
Chunk: {chunk}

Is this chunk relevant and useful for answering the question?
Reply with ONLY the word YES or NO."""


def grader_node(state: RAGState) -> dict:
    """
    Scores each retrieved chunk as relevant (YES) or not (NO).

    WHY this exists:
      retrieve_advanced returns the top-k by similarity score, but similarity
      does not equal relevance. A chunk can score highly because it shares
      keywords without actually answering the question. The grader filters
      these out so the answer agent only sees genuinely useful context.

    HOW it works:
      For each chunk, we call the LLM with a tight YES/NO prompt.
      Only YES chunks pass through to the answer agent.
      If ALL chunks are graded NO, we pass them all through anyway —
      better to try than to return an empty answer.
    """
    query   = state["query"]
    chunks  = state["retrieved"]

    if not chunks:
        print("[grader] no chunks to grade")
        return {"graded": []}

    graded = []
    for item in chunks:
        content = item.get("chunk", {}).get("original_content", "").strip()
        if not content:
            continue
        prompt = GRADER_PROMPT.format(query=query, chunk=content[:600])
        try:
            raw = get_provider().invoke(prompt).strip().upper()
            passed = raw.startswith("YES")
        except Exception as e:
            print(f"[grader] LLM error ({e}), keeping chunk")
            passed = True   # safe default: keep on error

        if passed:
            graded.append(item)

    # Fallback: if nothing passed, keep all (avoid empty context)
    if not graded:
        print(f"[grader] all {len(chunks)} chunks failed — keeping all as fallback")
        graded = chunks

    print(f"[grader] {len(graded)}/{len(chunks)} chunks passed")
    return {"graded": graded}

ANSWER_PROMPT = """You are a helpful assistant. Answer the question using ONLY the context below.
If the context does not contain enough information to answer, say "I don't have enough information to answer this question."

Context:
{context}

Question: {query}
Answer:"""


def answer_node(state: RAGState) -> dict:
    """
    Generates the final answer from graded context only.

    WHY graded context only:
      The grader already filtered out irrelevant chunks.
      Passing only relevant chunks reduces hallucination and keeps
      the answer focused — the LLM won't pad with unrelated facts.

    The context is built by joining original_content from each graded chunk,
    separated by blank lines so the LLM can distinguish chunk boundaries.
    """
    graded = state["graded"]
    query  = state["query"]

    if not graded:
        return {"answer": "I don't have enough information to answer this question."}

    # Build context string from graded chunks
    context_parts = []
    for item in graded:
        content = item.get("chunk", {}).get("original_content", "").strip()
        if content:
            context_parts.append(content)
    context = "\n\n".join(context_parts)

    prompt = ANSWER_PROMPT.format(context=context, query=query)
    try:
        answer = get_provider().invoke(prompt).strip()
    except Exception as e:
        print(f"[answer] LLM error ({e})")
        answer = "I encountered an error generating the answer."

    print(f"[answer] generated {len(answer)} char answer")
    return {"answer": answer}

REFLECTION_PROMPT = """You are checking whether an answer is properly grounded in the provided context.

Context:
{context}

Question: {query}
Answer: {answer}

Is the answer directly supported by the context above?
A good answer only uses facts present in the context — it does not add outside knowledge.
Reply with ONLY the word YES or NO."""


def reflection_node(state: RAGState) -> dict:
    """
    Checks if the generated answer is grounded in the retrieved context.

    WHY this exists:
      LLMs can hallucinate — generating confident-sounding answers from
      training data rather than the actual context. Reflection catches this
      by asking a separate LLM call: "is this answer actually in the context?"

    WHAT happens on failure:
      grounded=False triggers route_after_reflection to loop back to
      retrieval_node with an expanded query (up to 2 retries).
      After 2 retries we return the best answer we have.

    WHY a separate prompt:
      The reflection call is intentionally a different prompt to the answer
      call — it's acting as an independent judge, not the same system
      that generated the answer.
    """
    answer  = state["answer"]
    query   = state["query"]
    graded  = state["graded"]

    if not answer or not graded:
        # Nothing to reflect on — treat as grounded to avoid infinite loop
        return {"grounded": True}

    context = "\n\n".join(
        item.get("chunk", {}).get("original_content", "")
        for item in graded
    ).strip()

    prompt = REFLECTION_PROMPT.format(context=context, query=query, answer=answer)
    try:
        raw = get_provider().invoke(prompt).strip().upper()
        grounded = raw.startswith("YES")
    except Exception as e:
        print(f"[reflection] LLM error ({e}), assuming grounded")
        grounded = True   # safe default: don't loop forever on error

    print(f"[reflection] grounded={grounded}  iterations={state.get('iterations', 0)}")
    return {"grounded": grounded}


# ── Routing functions ─────────────────────────────────────────────────────────

def route_after_router(state: RAGState) -> str:
    return "out_of_scope" if state["query_type"] == "out_of_scope" else "retrieval"

def route_after_reflection(state: RAGState) -> str:
    if state["grounded"] or state["iterations"] >= 2:
        return "end"
    return "retry"


# ── Graph assembly ────────────────────────────────────────────────────────────

def build_rag_graph():
    g = StateGraph(RAGState)
    g.add_node("router",     router_node)
    g.add_node("retrieval",  retrieval_node)
    g.add_node("grader",     grader_node)
    g.add_node("answer",     answer_node)
    g.add_node("reflection", reflection_node)

    g.add_edge(START, "router")
    g.add_conditional_edges("router", route_after_router,
        {"retrieval": "retrieval", "out_of_scope": END})
    g.add_edge("retrieval", "grader")
    g.add_edge("grader",    "answer")
    g.add_edge("answer",    "reflection")
    g.add_conditional_edges("reflection", route_after_reflection,
        {"end": END, "retry": "retrieval"})
    return g.compile()


_graph = None


def run_rag_graph(query: str, db: Any = None, bm25: Any = None) -> RAGState:
    """
    Run the full agent graph.
    db and bm25 passed via LangGraph configurable — standard pattern for
    injecting runtime dependencies without global variables.
    """
    global _graph
    if _graph is None:
        _graph = build_rag_graph()
    return _graph.invoke(
        {
            "query": query, "query_type": "", "retrieved": [],
            "graded": [], "answer": "", "grounded": False,
            "iterations": 0, "semantic_count": 0.0, "bm25_count": 0.0,
        },
        config={"configurable": {"db": db, "bm25": bm25}},
    )
