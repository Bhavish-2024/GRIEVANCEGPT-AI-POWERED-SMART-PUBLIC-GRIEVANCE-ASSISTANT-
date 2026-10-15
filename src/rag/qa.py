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

def check_substantive_overlap(query: str, chunks: List[Dict[str, Any]]) -> bool:
    """Verify that retrieved chunks contain substantive query keywords beyond generic civic stopwords."""
    import re
    from src.nlp.preprocessing import normalize_text
    stopwords = {
        'what', 'which', 'how', 'where', 'who', 'when', 'why', 'is', 'are', 'was', 'were',
        'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'and', 'or', 'do', 'does',
        'can', 'could', 'should', 'would', 'i', 'you', 'my', 'your', 'our', 'it', 'this',
        'that', 'there', 'please', 'tell', 'me', 'about', 'any', 'some'
    }
    q_words = [w for w in re.findall(r'\b[a-zA-Z\u0B80-\u0BFF]{3,}\b', normalize_text(query).lower()) if w not in stopwords]
    if not q_words:
        return False
        
    combined_text = ' '.join(c['text'].lower() for c in chunks)
    generic_words = {'municipal', 'municipality', 'civic', 'department', 'departments', 'government', 'govt', 'public'}
    specific_q_words = [w for w in q_words if w not in generic_words]
    
    if specific_q_words:
        specific_matches = [w for w in specific_q_words if w in combined_text]
        return len(specific_matches) > 0
    matches = [w for w in q_words if w in combined_text]
    return len(matches) > 0

def answer_civic_question(query: str) -> Dict[str, Any]:
    """
    Execute grounded RAG workflow:
    1. Retrieve relevant knowledge chunks.
    2. Validate substantive relevance of retrieved documents.
    3. If no relevant info exists, return honest limitation acknowledgment.
    4. Otherwise, generate grounded answer with source citations.
    """
    chunks = retriever.retrieve(query, top_k=3, min_score=0.02)
    
    if not chunks or not check_substantive_overlap(query, chunks):
        return {
            "query": query,
            "answer": "I could not find relevant information in the municipal civic knowledge base for your inquiry. Please consult your local municipal corporation office or ward administrative center for specific guidance.",
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
        sources_str = ", ".join(sources)
        answer = (
            f"Based on civic guidance from **{sources_str}**:\n\n"
            f"{extracted_content}\n\n"
            f"For complete details or escalation, please refer to the corresponding department."
        )

    return {
        "query": query,
        "answer": answer,
        "sources": sources,
        "chunks": chunks
    }
