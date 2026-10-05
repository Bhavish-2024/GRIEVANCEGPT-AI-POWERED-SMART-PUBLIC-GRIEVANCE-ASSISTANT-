"""
Retriever module for GrievanceGPT RAG knowledge base.
Performs similarity search across chunked civic documents.
"""
import joblib
import numpy as np
from typing import List, Dict, Any
from sklearn.metrics.pairwise import cosine_similarity
from src.config import RAG_INDEX_PATH
from src.rag.ingest import build_knowledge_index

class KnowledgeRetriever:
    def __init__(self):
        self.index_data = None
        self._load_index()

    def _load_index(self):
        """Load vector index from disk, building it if missing."""
        if not RAG_INDEX_PATH.exists():
            print("[Retriever] Index not found, building now...")
            self.index_data = build_knowledge_index()
        else:
            try:
                self.index_data = joblib.load(RAG_INDEX_PATH)
            except Exception as e:
                print(f"[Retriever] Error loading index: {e}. Rebuilding...")
                self.index_data = build_knowledge_index()

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve top_k most relevant chunks for a user query.
        """
        if not self.index_data or "chunks" not in self.index_data:
            self._load_index()
            if not self.index_data:
                return []

        chunks = self.index_data["chunks"]
        tfidf = self.index_data["tfidf_vectorizer"]
        tfidf_matrix = self.index_data["tfidf_matrix"]

        query_vec = tfidf.transform([query])
        scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            score = float(scores[idx])
            chunk = chunks[idx]
            results.append({
                "doc_name": chunk["doc_name"],
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "score": round(score, 4)
            })

        return results

retriever = KnowledgeRetriever()
