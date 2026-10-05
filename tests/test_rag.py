"""
Unit tests for RAG retriever and grounded FAQ search.
"""
from src.rag.retriever import retriever
from src.rag.qa import answer_civic_question

def test_retriever_water_query():
    results = retriever.retrieve("Which department handles drinking water supply?", top_k=2)
    assert len(results) > 0
    top_doc = results[0]["doc_name"]
    assert top_doc in ["water_supply.txt", "departments.txt", "faq.txt"]

def test_retriever_roads_query():
    results = retriever.retrieve("How are potholes and damaged roads repaired?", top_k=2)
    assert len(results) > 0
    docs = [r["doc_name"] for r in results]
    assert any(d in ["roads.txt", "departments.txt", "faq.txt"] for d in docs)

def test_rag_qa_response():
    resp = answer_civic_question("What information should I provide for a public grievance?")
    assert len(resp["answer"]) > 20
    assert len(resp["sources"]) > 0
