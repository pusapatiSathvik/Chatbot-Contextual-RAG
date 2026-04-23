from rank_bm25 import BM25Okapi
from typing import List, Dict, Any
import nltk
from nltk.tokenize import word_tokenize

# Download NLTK tokenizer data if not present
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')


class BM25Retriever:
    def __init__(self, documents: List[Dict[str, Any]]) -> None:
        self.documents = documents
        self._empty = len(documents) == 0
        if not self._empty:
            corpus = [
                word_tokenize(
                    (doc.get("original_content", "") + " " + doc.get("contextualized_content", "")).lower()
                )
                for doc in documents
            ]
            self.bm25 = BM25Okapi(corpus)

    def search(self, query: str, k: int = 20) -> List[Dict[str, Any]]:
        if self._empty:
            return []
        tokenized_query = word_tokenize(query.lower())
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [
            {
                "doc_id": self.documents[i]["doc_id"],
                "original_index": self.documents[i]["original_index"],
                "content": self.documents[i].get("original_content", ""),
                "contextualized_content": self.documents[i].get("contextualized_content", ""),
                "score": float(scores[i]),
            }
            for i in top_indices
        ]


def create_bm25_index(db: Any) -> "BM25Retriever":
    """Create a BM25Retriever from the metadata stored in a ContextualVectorDB instance."""
    metadatas = db.metadata.get("metadatas", []) if isinstance(db.metadata, dict) else []
    return BM25Retriever(metadatas)
