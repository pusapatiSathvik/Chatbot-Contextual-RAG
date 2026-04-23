from fastapi import FastAPI, File, UploadFile
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
from ocr_and_chunking import merge_context, upload_create_chunks, save_chunks_to_file
from retrieval import retrieve_advanced


UPLOAD_FOLDER = "Files_uploaded"
Path(UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)

# Global state
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
        print("No seed file found at data/codebase_chunks.json — starting with empty DB.")
        print("Use the Upload PDF button in the UI to add documents.")
        # Still initialise metadata so BM25 doesn't break on an empty DB
        db.metadata = {"metadatas": []}

    bm25_index = create_bm25_index(db)
    print("App startup complete — DB and BM25 index ready.")
    yield
    print("App shutting down.")


app = FastAPI(lifespan=lifespan)


class ChatRequest(BaseModel):
    input: str


@app.post("/chat")
def chat(query: ChatRequest) -> Dict:
    global last_summary, db, bm25_index
    print(f"Query: {query.input}")

    if db.collection.count() == 0:
        return {
            "role": "assistant",
            "content": "No documents in the database yet. Please upload a PDF first using the upload button.",
        }

    # Prepend conversation summary for context-aware retrieval
    extended_query = f"{last_summary}\n\nUser question: {query.input}" if last_summary else query.input
    print(f"Extended query: {extended_query}")

    retrieved_context, _, _ = retrieve_advanced(extended_query, db, bm25_index, k=10)
    context = merge_context(retrieved_context)
    response = get_response(extended_query, context)
    summary = summarize_dialog(query.input, response["content"])
    last_summary = summary
    return response


@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)) -> Dict:
    global db, bm25_index

    if not file.filename.lower().endswith(".pdf"):
        return {"status": "error", "message": "Only PDF files are accepted."}

    chunks = upload_create_chunks(db, file, UPLOAD_FOLDER)
    json_name = file.filename.replace(".pdf", ".json")
    save_chunks_to_file(chunks, json_name)

    with open(f"data/{json_name}", "r", encoding="utf-8") as f:
        uploaded_data = json.load(f)

    db.append_data(uploaded_data)
    # Rebuild BM25 index to include the newly added documents
    bm25_index = create_bm25_index(db)

    return {
        "status": "success",
        "message": f"File '{json_name}' converted and added to the database.",
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
        button {
            background-color: #007bff; color: white;
            padding: 5px 10px; border-radius: 5px;
            border: none; cursor: pointer;
        }
        button:hover { background-color: #0056b3; }
        label {
            background-color: #007bff; color: white;
            padding: 5px 10px; border-radius: 5px;
            cursor: pointer; display: inline-block;
        }
        label:hover { background-color: #0056b3; }
        input[type="file"] { display: none; }
        #fileName { margin-left: 10px; font-style: italic; }
        #uploadStatus { margin-top: 10px; font-weight: bold; color: #007bff; }
    </style>
</head>
<body>
    <h1>Chatbot — Contextual RAG with Hybrid Search</h1>

    <div id="chatBox"></div>
    <br>
    <input type="text" id="userInput" placeholder="Type your message here..."
           onkeypress="handleKeyPress(event)">
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
            const pdfFile = document.getElementById("pdfFileInput").files[0];
            document.getElementById("fileName").textContent = pdfFile ? pdfFile.name : "No file selected";
        }

        async function sendMessage() {
            const userInput = document.getElementById("userInput").value.trim();
            if (!userInput) return;
            const chatBox = document.getElementById("chatBox");
            chatBox.innerHTML += `<p><strong>You:</strong> ${userInput}</p>`;
            document.getElementById("userInput").value = "";

            const response = await fetch("/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ input: userInput })
            });
            const data = await response.json();
            chatBox.innerHTML += `<p><strong>Assistant:</strong> ${data.content}</p>`;
            chatBox.scrollTop = chatBox.scrollHeight;
        }

        function handleKeyPress(event) {
            if (event.key === "Enter") sendMessage();
        }

        async function resetChat() {
            const response = await fetch("/reset_chat", { method: "POST" });
            const data = await response.json();
            if (data.status === "success") {
                document.getElementById("chatBox").innerHTML = "";
            } else {
                alert("Failed to reset chat.");
            }
        }

        async function uploadPDF() {
            const pdfFile = document.getElementById("pdfFileInput").files[0];
            const uploadStatus = document.getElementById("uploadStatus");
            if (!pdfFile) { alert("No file selected!"); return; }

            const formData = new FormData();
            formData.append("file", pdfFile);
            uploadStatus.textContent = "Upload in progress...";

            try {
                const response = await fetch("/upload_pdf", { method: "POST", body: formData });
                const data = await response.json();
                uploadStatus.textContent = response.ok ? "Success: " + data.message : "Error: " + data.message;
                setTimeout(() => uploadStatus.textContent = "", 4000);
            } catch (error) {
                uploadStatus.textContent = "Error during file upload.";
            }
        }
    </script>
</body>
</html>
    """


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
