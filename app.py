from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import uvicorn
import json
from fastapi.responses import HTMLResponse
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator, Dict

from inference_by_Ollama import get_response, summarize_dialog
from contextual_vector_db import ContextualVectorDB
from bm25 import BM25Retriever, create_bm25_index
from data_pipeline import ingest_pdf, save_chunks_to_file, merge_context   # Phase 2
from retrieval import retrieve_advanced


UPLOAD_FOLDER = "Files_uploaded"
Path(UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)

db: ContextualVectorDB = None
bm25_index: BM25Retriever = None
last_summary: str = ""


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    global db, bm25_index, last_summary
    last_summary = ""
    db = ContextualVectorDB("my_contextual_db")

    seed_file = Path("data/codebase_chunks.json")
    if seed_file.exists():
        print(f"Loading seed data from {seed_file} ...")
        with open(seed_file, "r", encoding="utf-8") as f:
            transformed_dataset = json.load(f)
        db.load_data(transformed_dataset)
        print("Seed data loaded.")
    else:
        print("No seed file — starting with empty DB.")
        db.metadata = {"metadatas": []}

    bm25_index = create_bm25_index(db)
    print("App startup complete.")
    yield
    print("App shutting down.")


app = FastAPI(lifespan=lifespan)


class ChatRequest(BaseModel):
    input: str


@app.post("/chat")
def chat(query: ChatRequest) -> Dict:
    global last_summary, db, bm25_index

    if db.collection.count() == 0:
        return {"role": "assistant", "content": "No documents yet. Please upload a PDF first."}

    # Prepend conversation summary for context-aware retrieval (same as before)
    extended_query = f"{last_summary}\n\nUser question: {query.input}" if last_summary else query.input

    # Phase 4: multi-agent graph replaces the old single-pass pipeline
    from rag_graph import run_rag_graph
    final_state = run_rag_graph(extended_query, db=db, bm25=bm25_index)

    # Router said out_of_scope — return polite decline without hallucinating
    if final_state["query_type"] == "out_of_scope":
        return {
            "role": "assistant",
            "content": "I can only answer questions based on the uploaded documents. This question appears to be outside the scope of the available content.",
        }

    answer = final_state["answer"]

    # summarize_dialog still runs for conversation memory (unchanged)
    last_summary = summarize_dialog(query.input, answer)
    return {"role": "assistant", "content": answer}


@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)) -> Dict:
    global db, bm25_index
    if not file.filename.lower().endswith(".pdf"):
        return {"status": "error", "message": "Only PDF files are accepted."}

    try:
        # Phase 2: ingest_pdf handles extraction + semantic chunking + dedup
        dataset = ingest_pdf(db, file, UPLOAD_FOLDER)
    except ValueError as e:
        # Duplicate document or empty extraction
        return {"status": "error", "message": str(e)}

    json_name = file.filename.replace(".pdf", ".json")
    save_chunks_to_file(dataset, json_name)

    db.append_data(dataset)
    bm25_index = create_bm25_index(db)

    chunk_count = sum(len(d["chunks"]) for d in dataset)
    return {
        "status": "success",
        "message": f"'{file.filename}' processed: {chunk_count} chunks added.",
    }


@app.post("/reset_chat")
def reset_chat() -> Dict:
    global last_summary
    last_summary = ""
    return {"status": "success", "message": "New chat ready."}


@app.get("/", response_class=HTMLResponse)
def chat_page() -> str:
    return """
<!DOCTYPE html>
<html>
<head>
    <title>Contextual RAG Chatbot</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        #chatBox { border: 1px solid #ccc; padding: 10px; width: 500px; height: 300px; overflow-y: auto; }
        #userInput { width: 400px; }
        button { background-color: #007bff; color: white; padding: 5px 10px; border-radius: 5px; border: none; cursor: pointer; }
        button:hover { background-color: #0056b3; }
        label { background-color: #007bff; color: white; padding: 5px 10px; border-radius: 5px; cursor: pointer; display: inline-block; }
        label:hover { background-color: #0056b3; }
        input[type="file"] { display: none; }
        #uploadStatus { margin-top: 10px; font-weight: bold; color: #007bff; }
    </style>
</head>
<body>
    <h1>Chatbot — Contextual RAG with Hybrid Search</h1>
    <div id="chatBox"></div><br>
    <input type="text" id="userInput" placeholder="Type your message here..." onkeypress="handleKeyPress(event)">
    <button onclick="sendMessage()">Send</button>
    <button onclick="resetChat()">New Chat</button>
    <hr>
    <h2>Add a PDF file</h2>
    <label for="pdfFileInput">Choose a file</label>
    <input type="file" id="pdfFileInput" accept="application/pdf" onchange="displayFileName()" />
    <span id="fileName">No file selected</span>
    <button type="button" onclick="uploadPDF()">Upload PDF</button>
    <div id="uploadStatus"></div>
    <script>
        function displayFileName() {
            const f = document.getElementById("pdfFileInput").files[0];
            document.getElementById("fileName").textContent = f ? f.name : "No file selected";
        }
        async function sendMessage() {
            const userInput = document.getElementById("userInput").value.trim();
            if (!userInput) return;
            const chatBox = document.getElementById("chatBox");
            chatBox.innerHTML += `<p><strong>You:</strong> ${userInput}</p>`;
            document.getElementById("userInput").value = "";
            const response = await fetch("/chat", { method: "POST", headers: {"Content-Type":"application/json"}, body: JSON.stringify({input: userInput}) });
            const data = await response.json();
            chatBox.innerHTML += `<p><strong>Assistant:</strong> ${data.content}</p>`;
            chatBox.scrollTop = chatBox.scrollHeight;
        }
        function handleKeyPress(e) { if (e.key === "Enter") sendMessage(); }
        async function resetChat() {
            const r = await fetch("/reset_chat", {method:"POST"});
            const d = await r.json();
            if (d.status === "success") document.getElementById("chatBox").innerHTML = "";
        }
        async function uploadPDF() {
            const f = document.getElementById("pdfFileInput").files[0];
            const status = document.getElementById("uploadStatus");
            if (!f) { alert("No file selected!"); return; }
            const fd = new FormData(); fd.append("file", f);
            status.textContent = "Processing...";
            try {
                const r = await fetch("/upload_pdf", {method:"POST", body: fd});
                const d = await r.json();
                status.textContent = (d.status === "success" ? "✓ " : "✗ ") + d.message;
                setTimeout(() => status.textContent = "", 5000);
            } catch(e) { status.textContent = "Upload error."; }
        }
    </script>
</body>
</html>"""


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
