"""
Grounded Question-Answering module for Civic FAQs and Inquiries.
Retrieves authoritative civic guidance from the knowledge base and generates
answers with verified source citations.
"""
from typing import Dict, Any, List
from src.rag.retriever import retriever
from src.llm.ollama_client import ollama_client

RAG_QA_SYSTEM = """You are GrievanceGPT's official Civic Advisory Assistant.
Answer the citizen's question based strictly and accurately on the provided context excerpts.
Cite the source file names for the information provided.
Do not fabricate government policies. If the information is not in the context, state that clearly and suggest consulting municipal officers.
"""

def answer_civic_question(query: str) -> Dict[str, Any]:
    """
    Execute grounded RAG workflow:
    1. Retrieve relevant knowledge chunks.
    2. Format prompt with context.
    3. Query LLM / synthesize answer.
    4. Return answer with citations.
    """
    chunks = retriever.retrieve(query, top_k=3)
    
    if not chunks:
        return {
            "query": query,
            "answer": "No relevant civic knowledge documents found for this query.",
            "sources": [],
            "chunks": []
        }

    sources = list(dict.fromkeys(c["doc_name"] for c in chunks))
    
    # Check if Ollama is connected
    conn_info = ollama_client.check_connection()
    if conn_info.get("connected"):
        context_block = "\n\n---\n\n".join([f"Source: {c['doc_name']}\n{c['text']}" for c in chunks])
        prompt = f"User Question: {query}\n\nRelevant Civic Context:\n{context_block}\n\nPlease provide a clear, helpful, grounded answer."
        answer = ollama_client.generate(prompt, system=RAG_QA_SYSTEM)
    else:
        # Grounded extractive synthesis from top chunks
        top_chunk = chunks[0]
        summary_lines = []
        for line in top_chunk["text"].split("\n"):
            line = line.strip()
            if line and not line.startswith("#"):
                summary_lines.append(line)
                if len(summary_lines) >= 4:
                    break
        
        extracted_content = " ".join(summary_lines)
        answer = (
            f"Based on civic guidance from **{top_chunk['doc_name']}**:\n\n"
            f"{extracted_content}\n\n"
            f"For complete details or escalation, please refer to the corresponding department."
        )

    return {
        "query": query,
        "answer": answer,
        "sources": sources,
        "chunks": chunks
    }
