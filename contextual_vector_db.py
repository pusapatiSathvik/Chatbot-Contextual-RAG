from concurrent.futures import ThreadPoolExecutor, as_completed
from chromadb import PersistentClient
from chromadb.utils import embedding_functions
from langchain_ollama import OllamaLLM
import numpy as np
from typing import List, Dict, Any, Set
from tqdm import tqdm


class ContextualVectorDB:
    def __init__(self, name: str) -> None:
        self.name = name
        self.db_path = f"./data/{name}/chromadb"
        self.metadata: Dict[str, Any] = {}
        self.client = PersistentClient(path=self.db_path)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-mpnet-base-v2"
        )
        self.collection = self.client.get_or_create_collection(
            name=self.name,
            embedding_function=self.embedding_fn,
        )
        self.llm = OllamaLLM(model="llama3.1")

    # ------------------------------------------------------------------
    # Duplicate detection
    # ------------------------------------------------------------------

    def get_known_hashes(self) -> Set[str]:
        """
        Return the set of original_uuid (SHA-256 hashes) already stored.
        Used by upload_create_chunks() to reject duplicate documents.
        """
        if not self.metadata or "metadatas" not in self.metadata:
            return set()
        seen: Set[str] = set()
        for meta in self.metadata.get("metadatas", []):
            h = meta.get("original_uuid", "")
            if h:
                seen.add(h)
        return seen

    # ------------------------------------------------------------------
    # Context generation
    # ------------------------------------------------------------------

    def situate_context(self, doc: str, chunk: str) -> str:
        """Ask the LLM to produce a short context sentence for a chunk."""
        prompt = (
            f"<document>\n{doc}\n</document>\n\n"
            f"Here is the chunk we want to situate within the whole document:\n"
            f"<chunk>\n{chunk}\n</chunk>\n\n"
            f"Please give a short succinct context to situate this chunk within the overall "
            f"document for the purposes of improving search retrieval of the chunk. "
            f"Answer only with the succinct context and nothing else."
        )
        return self.llm.invoke(prompt)

    # ------------------------------------------------------------------
    # Initial data loading
    # ------------------------------------------------------------------

    def load_data(self, dataset: List[Dict[str, Any]], parallel_threads: int = 4) -> None:
        """Load a full dataset into ChromaDB. Skips if DB already populated."""
        if self.collection.count() > 0:
            print("Vector database already populated — skipping data loading.")
            self.metadata = self.collection.get(include=["metadatas"])
            return

        texts_to_embed: List[str] = []
        metadata: List[Dict[str, Any]] = []
        total_chunks = sum(len(doc["chunks"]) for doc in dataset)

        def process_chunk(doc: Dict[str, Any], chunk: Dict[str, Any]) -> Dict[str, Any]:
            contextualized_text = self.situate_context(doc["content"], chunk["content"])
            return {
                "text_to_embed": f"{chunk['content']}\n\n{contextualized_text}",
                "metadata": {
                    "doc_id": doc["doc_id"],
                    "chunk_id": chunk["chunk_id"],
                    "original_index": chunk["original_index"],
                    "original_content": chunk["content"],
                    "contextualized_content": contextualized_text,
                    # Store for citations and duplicate detection
                    "source_file": doc.get("source_file", doc["doc_id"]),
                    "original_uuid": doc.get("original_uuid", ""),
                },
            }

        print(f"Processing {total_chunks} chunks with {parallel_threads} threads …")
        with ThreadPoolExecutor(max_workers=parallel_threads) as executor:
            futures = [
                executor.submit(process_chunk, doc, chunk)
                for doc in dataset
                for chunk in doc["chunks"]
            ]
            for future in tqdm(as_completed(futures), total=total_chunks, desc="Processing chunks"):
                result = future.result()
                texts_to_embed.append(result["text_to_embed"])
                metadata.append(result["metadata"])

        self._embed_and_store(texts_to_embed, metadata)
        print(f"Database loaded. Total chunks: {len(texts_to_embed)}")

    def _embed_and_store(self, texts: List[str], data: List[Dict[str, Any]]) -> None:
        batch_size = 128
        with tqdm(total=len(texts), desc="Embedding & storing") as pbar:
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i: i + batch_size]
                batch_metadata = data[i: i + batch_size]
                self.collection.add(
                    ids=[f"doc_{idx}" for idx in range(i, i + len(batch_texts))],
                    documents=batch_texts,
                    metadatas=batch_metadata,
                )
                pbar.update(len(batch_texts))
        self.metadata = self.collection.get(include=["metadatas"])
        print(f"Total documents in database: {self.collection.count()}")

    # ------------------------------------------------------------------
    # Incremental append (for new PDF uploads)
    # ------------------------------------------------------------------

    def append_data(self, new_dataset: List[Dict[str, Any]], parallel_threads: int = 4) -> None:
        if not new_dataset:
            print("No data to add.")
            return

        existing_count = self.collection.count()
        print(f"Existing documents: {existing_count}")
        total_chunks = sum(len(doc["chunks"]) for doc in new_dataset)

        def process_chunk(doc: Dict[str, Any], chunk: Dict[str, Any]) -> Dict[str, Any]:
            contextualized_text = self.situate_context(doc["content"], chunk["content"])
            return {
                "text_to_embed": f"{chunk['content']}\n\n{contextualized_text}",
                "metadata": {
                    "doc_id": doc["doc_id"],
                    "chunk_id": chunk["chunk_id"],
                    "original_index": chunk["original_index"],
                    "original_content": chunk["content"],
                    "contextualized_content": contextualized_text,
                    "source_file": doc.get("source_file", doc["doc_id"]),
                    "original_uuid": doc.get("original_uuid", ""),
                },
            }

        texts_to_embed: List[str] = []
        metadata: List[Dict[str, Any]] = []

        print(f"Processing {total_chunks} new chunks with {parallel_threads} threads …")
        with ThreadPoolExecutor(max_workers=parallel_threads) as executor:
            futures = [
                executor.submit(process_chunk, doc, chunk)
                for doc in new_dataset
                for chunk in doc["chunks"]
            ]
            for future in tqdm(as_completed(futures), total=total_chunks, desc="Processing chunks"):
                result = future.result()
                texts_to_embed.append(result["text_to_embed"])
                metadata.append(result["metadata"])

        self._append_embed_and_store(texts_to_embed, metadata, offset=existing_count)
        print(f"Chunks added: {len(texts_to_embed)}")

    def _append_embed_and_store(
        self,
        texts: List[str],
        data: List[Dict[str, Any]],
        offset: int = 0,
    ) -> None:
        batch_size = 128
        with tqdm(total=len(texts), desc="Embedding & appending") as pbar:
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i: i + batch_size]
                batch_metadata = data[i: i + batch_size]
                batch_ids = [f"doc_{offset + i + j}" for j in range(len(batch_texts))]
                self.collection.add(
                    ids=batch_ids,
                    documents=batch_texts,
                    metadatas=batch_metadata,
                )
                pbar.update(len(batch_texts))
        self.metadata = self.collection.get(include=["metadatas"])
        print(f"Total documents in database: {self.collection.count()}")

    # ------------------------------------------------------------------
    # Semantic search
    # ------------------------------------------------------------------

    def search(self, query: str, k: int = 20) -> List[Dict[str, Any]]:
        """Return the top-k most semantically similar chunks for a query."""
        query_embedding = self.embedding_fn([query])[0]

        documents = self.collection.get(include=["metadatas", "embeddings"])
        doc_embeddings = documents.get("embeddings")
        doc_metadata = documents.get("metadatas")

        if doc_embeddings is None or doc_metadata is None:
            raise ValueError("Embeddings or metadata missing from the collection.")

        doc_embeddings = np.array(doc_embeddings)
        similarities = np.dot(doc_embeddings, query_embedding)
        top_indices = np.argsort(similarities)[::-1][:k]

        return [
            {
                "metadata": doc_metadata[idx],
                "original_content": doc_metadata[idx].get("original_content", ""),
                "contextualized_content": doc_metadata[idx].get("contextualized_content", ""),
                "similarity": float(similarities[idx]),
            }
            for idx in top_indices
        ]
