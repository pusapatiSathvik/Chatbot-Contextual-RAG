"""
inference_by_Ollama.py
======================
Inference module — generates answers and conversation summaries.

PHASE 1 UPDATE: All LLM calls now go through ModelProvider so the
backend (Ollama / OpenAI / Anthropic / Gemini) is controlled entirely
by config.py / .env — no code changes needed to switch models.

The module name is kept as-is for backward compatibility with app.py.
"""

import time
from typing import Dict

from model_provider import get_provider


def get_response(user_input: str, context: str) -> Dict[str, str]:
    """
    Generate an answer grounded in the retrieved context.

    Returns a dict with role="assistant" and the model's response.
    """
    t0 = time.time()

    prompt = (
        "System: You are a helpful assistant. "
        "Answer the question ONLY based on the context below. "
        "If the context does not contain enough information to answer, "
        "say so clearly rather than guessing.\n\n"
        f"Context:\n{context}\n\n"
        f"User: {user_input}\n"
        "Assistant:"
    )

    print("Generating answer …")
    provider = get_provider()
    response = provider.invoke(prompt)

    print(f"Generation took {time.time() - t0:.1f}s")
    print(f"Model response: {response[:120]}{'...' if len(response) > 120 else ''}")

    return {"role": "assistant", "content": response}


def summarize_dialog(user_question: str, assistant_answer: str) -> str:
    """
    Produce a two-sentence summary of the latest turn.
    Used as context for follow-up queries.
    """
    summary_prompt = (
        "Please summarize the following conversation in two sentences:\n\n"
        f"User: {user_question}\n"
        f"Assistant: {assistant_answer}\n\n"
        "Summary:"
    )

    provider = get_provider()
    summary_text = provider.invoke(summary_prompt)
    print(f"Conversation summary: {summary_text[:100]}...")
    return summary_text
