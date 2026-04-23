import time
from langchain_ollama import OllamaLLM
from typing import Dict


MODEL_ID = "llama3.1"  # must match exactly what `ollama list` shows (lowercase)
ollama = OllamaLLM(model=MODEL_ID, temperature=0.0)


def get_response(user_input: str, context: str) -> Dict[str, str]:
    """Generate an answer grounded in the retrieved context."""
    t0 = time.time()
    prompt = (
        f"System: You are a helpful assistant. Answer the question ONLY based on the context below.\n"
        f"Context:\n{context}\n\n"
        f"User: {user_input}\n"
        f"Assistant:"
    )

    print("Generating answer …")
    llm_result = ollama.generate(prompts=[prompt])
    response = llm_result.generations[0][0].text
    print(f"Generation took {time.time() - t0:.1f}s")
    print(f"Model response: {response}")
    return {"role": "assistant", "content": response.strip()}


def summarize_dialog(user_question: str, assistant_answer: str) -> str:
    """Produce a two-sentence summary of the latest turn for context-aware follow-up."""
    summary_prompt = (
        "Please summarize the following conversation in two sentences:\n\n"
        f"User: {user_question}\n"
        f"Assistant: {assistant_answer}\n\n"
        "Summary:"
    )
    summary_result = ollama.generate(prompts=[summary_prompt])
    summary_text = summary_result.generations[0][0].text.strip()
    print(f"Conversation summary: {summary_text}")
    return summary_text
