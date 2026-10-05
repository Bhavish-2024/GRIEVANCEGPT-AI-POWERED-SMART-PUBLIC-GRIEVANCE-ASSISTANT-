"""
RAG Ingestion Pipeline for GrievanceGPT Knowledge Base.
Loads civic procedure guides, chunks documents, generates vector embeddings,
and builds a persistent search index.
"""
import joblib
import sys
from pathlib import Path

BASE_PATH = Path(__file__).resolve().parent.parent.parent
if str(BASE_PATH) not in sys.path:
    sys.path.insert(0, str(BASE_PATH))

from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from src.config import KNOWLEDGE_BASE_DIR, VECTOR_STORE_DIR, RAG_INDEX_PATH
from src.llm.ollama_client import ollama_client

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 80) -> List[str]:
    """
    Split document text into clean overlapping paragraphs/chunks.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    
    current_chunk = ""
    for p in paragraphs:
        if len(current_chunk) + len(p) <= chunk_size:
            current_chunk += ("\n\n" + p if current_chunk else p)
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = p
            
    if current_chunk:
        chunks.append(current_chunk)
        
    return chunks

def build_knowledge_index() -> Dict[str, Any]:
    """
    Index all documents in knowledge_base/ into vector store.
    """
    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    
    docs_chunks: List[Dict[str, Any]] = []
    texts_for_tfidf: List[str] = []
    
    doc_files = sorted(list(KNOWLEDGE_BASE_DIR.glob("*.txt")))
    print(f"[RAG Ingest] Found {len(doc_files)} knowledge base documents.")
    
    for doc_path in doc_files:
        try:
            content = doc_path.read_text(encoding="utf-8")
            chunks = chunk_text(content)
            for idx, c in enumerate(chunks):
                chunk_data = {
                    "doc_name": doc_path.name,
                    "chunk_id": f"{doc_path.stem}_{idx}",
                    "text": c
                }
                docs_chunks.append(chunk_data)
                texts_for_tfidf.append(c)
        except Exception as e:
            print(f"[RAG Ingest] Error reading {doc_path.name}: {e}")
            
    if not docs_chunks:
        print("[RAG Ingest] Warning: No knowledge documents indexed!")
        return {}

    # Build dual indexing: TF-IDF vectorizer + Ollama embeddings (if available)
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        sublinear_tf=True,
        stop_words="english"
    )
    tfidf_matrix = tfidf.fit_transform(texts_for_tfidf)
    
    # Try Ollama embeddings
    ollama_embeddings = []
    ollama_active = ollama_client.check_connection().get("connected", False)
    if ollama_active:
        print("[RAG Ingest] Generating Ollama dense embeddings...")
        for c in texts_for_tfidf:
            emb = ollama_client.get_embeddings(c)
            ollama_embeddings.append(emb)
    
    index_payload = {
        "chunks": docs_chunks,
        "tfidf_vectorizer": tfidf,
        "tfidf_matrix": tfidf_matrix,
        "ollama_embeddings": ollama_embeddings if any(ollama_embeddings) else None
    }
    
    joblib.dump(index_payload, RAG_INDEX_PATH)
    print(f"[RAG Ingest] Successfully indexed {len(docs_chunks)} chunks to {RAG_INDEX_PATH}")
    return index_payload

if __name__ == "__main__":
    build_knowledge_index()
