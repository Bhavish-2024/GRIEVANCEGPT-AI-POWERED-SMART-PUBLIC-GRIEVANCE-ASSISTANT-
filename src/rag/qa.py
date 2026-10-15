"""
Grounded Question-Answering module for Civic FAQs and Inquiries.
Retrieves authoritative civic guidance from the knowledge base and generates
comprehensive answers with verified source citations.
"""
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
from src.config import KNOWLEDGE_BASE_DIR
from src.rag.retriever import retriever
from src.llm.ollama_client import ollama_client
from src.nlp.preprocessing import normalize_text

RAG_QA_SYSTEM = """You are GrievanceGPT's official Civic Advisory Assistant.
Answer the citizen's question based strictly and accurately on the provided context excerpts.
Cite the source file names for the information provided.
Do not fabricate government policies. If the information is not in the context, state that clearly and suggest consulting municipal officers.
"""

DOMAIN_GUIDES = {
    "water_supply.txt": "water",
    "roads.txt": "roads",
    "electricity.txt": "electricity",
    "sanitation.txt": "sanitation",
    "public_health.txt": "public health"
}

def check_substantive_overlap(query: str, chunks: List[Dict[str, Any]]) -> bool:
    """Verify that retrieved chunks contain substantive query keywords beyond generic civic stopwords."""
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

def get_department_escalation(dept_keyword: str) -> Optional[str]:
    """Retrieve escalation contact for department from departments.txt."""
    dept_file = KNOWLEDGE_BASE_DIR / "departments.txt"
    if not dept_file.exists():
        return None
    content = dept_file.read_text(encoding="utf-8")
    for block in content.split("\n\n"):
        if dept_keyword.lower() in block.lower():
            for line in block.splitlines():
                if "Escalation Contact:" in line:
                    return line.split("Escalation Contact:")[-1].strip()
    return None

def parse_guide_sections(doc_name: str) -> Dict[str, List[str]]:
    """Parse structured sections of a citizen filing guide."""
    file_path = KNOWLEDGE_BASE_DIR / doc_name
    if not file_path.exists():
        return {}
    lines = [l.strip() for l in file_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    sections: Dict[str, List[str]] = {"title": []}
    current_key = "title"
    for l in lines:
        if l.endswith(":") and not l.startswith("-") and not (l[0].isdigit() and len(l) > 1 and l[1] in ". "):
            current_key = l[:-1].strip()
            sections[current_key] = []
        elif l.startswith("# "):
            sections["title"] = [l[2:].strip()]
        else:
            sections.setdefault(current_key, []).append(l)
    return sections

def synthesize_domain_guide_answer(doc_name: str, query: str, sources: List[str]) -> str:
    """Synthesize rich, structured, authoritative answer for a departmental guide."""
    sections = parse_guide_sections(doc_name)
    dept_kw = DOMAIN_GUIDES.get(doc_name, "")
    escalation = get_department_escalation(dept_kw)
    
    parts = []
    src_str = ", ".join(sources)
    parts.append(f"Based on verified municipal guidance (**{src_str}**):")
    
    # Department & Authority
    dept_lines = sections.get("Department In-Charge", [])
    dept_name = " ".join(dept_lines) if dept_lines else "Municipal Administration"
    dept_block = f"### Department In-Charge\n**{dept_name}**"
    if escalation:
        dept_block += f"\n• **Escalation Contact**: {escalation}"
    parts.append(dept_block)
    
    # Common Grievances Covered
    grievances = sections.get("Common Grievance Types", [])
    if grievances:
        clean_g = [f"• {g.lstrip('0123456789. -')}" for g in grievances]
        parts.append("### Common Grievances Covered\n" + "\n".join(clean_g))
        
    # Essential Information Required
    essential = sections.get("Information Essential for Quick Resolution", [])
    if essential:
        e_lines = []
        for e in essential:
            clean_e = e.lstrip("- ")
            if ":" in clean_e:
                k, v = clean_e.split(":", 1)
                e_lines.append(f"• **{k.strip()}**: {v.strip()}")
            else:
                e_lines.append(f"• {clean_e}")
        parts.append("### Information Required to File Complaint\n" + "\n".join(e_lines))
        
    # Expected Turnaround Time (TAT)
    tat = sections.get("Expected Turnaround Time (TAT)", [])
    if tat:
        t_lines = []
        for t in tat:
            clean_t = t.lstrip("- ")
            if ":" in clean_t:
                k, v = clean_t.split(":", 1)
                t_lines.append(f"• **{k.strip()}**: {v.strip()}")
            else:
                t_lines.append(f"• {clean_t}")
        parts.append("### Expected Turnaround Time (TAT)\n" + "\n".join(t_lines))

    return "\n\n".join(parts)

def synthesize_procedure_answer(query: str, sources: List[str]) -> str:
    """Synthesize guidance on complaint filing procedures and citizen rights."""
    file_path = KNOWLEDGE_BASE_DIR / "grievance_procedure.txt"
    if not file_path.exists():
        return ""
    lines = [l.strip() for l in file_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    
    parts = []
    src_str = ", ".join(sources)
    parts.append(f"Based on official municipal procedures (**{src_str}**):")
    
    steps = []
    timelines = []
    mode = None
    for line in lines:
        if "Key Steps for Filing" in line:
            mode = "steps"
        elif "Citizen Rights & Timelines" in line:
            mode = "timelines"
        elif "Prototype Disclaimer" in line:
            mode = None
        elif mode == "steps" and (line[0].isdigit() or line.startswith("-")):
            steps.append(f"• {line.lstrip('0123456789. -')}")
        elif mode == "timelines" and line.startswith("-"):
            clean_l = line.lstrip("- ")
            if ":" in clean_l:
                k, v = clean_l.split(":", 1)
                timelines.append(f"• **{k.strip()}**: {v.strip()}")
            else:
                timelines.append(f"• {clean_l}")

    if steps:
        parts.append("### Key Steps for Filing an Actionable Complaint\n" + "\n".join(steps))
    if timelines:
        parts.append("### Citizen Rights & Official Resolution Timelines\n" + "\n".join(timelines))
        
    return "\n\n".join(parts)

def synthesize_faq_answer(query: str, sources: List[str]) -> str:
    """Synthesize matching FAQ response from faq.txt."""
    faq_file = KNOWLEDGE_BASE_DIR / "faq.txt"
    if not faq_file.exists():
        return ""
    content = faq_file.read_text(encoding="utf-8")
    
    qas = []
    current_q = None
    current_a = []
    for line in content.splitlines():
        line = line.strip()
        if line.startswith("Q") and ":" in line:
            if current_q and current_a:
                qas.append((current_q, "\n".join(current_a).strip()))
            current_q = line
            current_a = []
        elif current_q:
            current_a.append(line)
    if current_q and current_a:
        qas.append((current_q, "\n".join(current_a).strip()))

    q_lower = query.lower()
    # Find best matching FAQ
    best_match = None
    for q_title, q_body in qas:
        q_words = [w for w in re.findall(r'\w+', q_lower) if len(w) > 3]
        if any(w in q_title.lower() or w in q_body.lower() for w in q_words):
            best_match = (q_title, q_body)
            break
            
    if not best_match and qas:
        best_match = qas[0]
        
    parts = [f"Based on official municipal FAQ guidelines (**{', '.join(sources)}**):"]
    if best_match:
        parts.append(f"### Frequently Asked Question\n**{best_match[0]}**\n\n{best_match[1]}")
    return "\n\n".join(parts)

def synthesize_departments_directory(sources: List[str]) -> str:
    """Synthesize directory of municipal departments."""
    dept_file = KNOWLEDGE_BASE_DIR / "departments.txt"
    if not dept_file.exists():
        return ""
    content = dept_file.read_text(encoding="utf-8")
    
    parts = [f"Based on municipal civic directory (**{', '.join(sources)}**):"]
    dept_blocks = []
    for block in content.split("\n\n"):
        block = block.strip()
        if not block or block.startswith("#"):
            continue
        lines = [l.strip() for l in block.splitlines() if l.strip()]
        if lines:
            dept_title = lines[0].lstrip("0123456789. -")
            details = []
            for dl in lines[1:]:
                clean_dl = dl.lstrip("- ")
                details.append(f"  • {clean_dl}")
            dept_blocks.append(f"**{dept_title}**\n" + "\n".join(details))
            
    parts.append("### Municipal & Civic Departments Directory\n" + "\n\n".join(dept_blocks[:6]))
    return "\n\n".join(parts)

def synthesize_grounded_answer(query: str, chunks: List[Dict[str, Any]], sources: List[str]) -> str:
    """Comprehensive grounded synthesis from authoritative knowledge base files."""
    # 1. Check if a specific domain guide matches
    for doc_name in sources:
        if doc_name in DOMAIN_GUIDES:
            return synthesize_domain_guide_answer(doc_name, query, sources)
            
    # 2. Check if procedure document matches
    if "grievance_procedure.txt" in sources:
        return synthesize_procedure_answer(query, sources)
        
    # 3. Check if FAQ document matches
    if "faq.txt" in sources:
        return synthesize_faq_answer(query, sources)
        
    # 4. Check if departments directory matches
    if "departments.txt" in sources:
        return synthesize_departments_directory(sources)
        
    # 5. Generic grounded extraction across all retrieved chunks
    content_blocks = []
    for c in chunks:
        c_lines = [l.strip() for l in c["text"].splitlines() if l.strip() and not l.strip().startswith("#")]
        if c_lines:
            content_blocks.append("\n".join(c_lines))
            
    joined = "\n\n".join(content_blocks)
    return (
        f"Based on civic guidance from **{', '.join(sources)}**:\n\n"
        f"{joined}\n\n"
        f"For complete details or escalation, please refer to the corresponding department."
    )

def answer_civic_question(query: str) -> Dict[str, Any]:
    """
    Execute grounded RAG workflow:
    1. Retrieve relevant knowledge chunks.
    2. Validate substantive relevance of retrieved documents.
    3. If no relevant info exists, return honest limitation acknowledgment.
    4. Otherwise, generate grounded answer with source citations.
    """
    chunks = retriever.retrieve(query, top_k=5, min_score=0.02)
    
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
        answer = synthesize_grounded_answer(query, chunks, sources)

    return {
        "query": query,
        "answer": answer,
        "sources": sources,
        "chunks": chunks
    }
