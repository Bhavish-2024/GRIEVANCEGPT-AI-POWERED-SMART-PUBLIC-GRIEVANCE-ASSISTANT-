"""
Intent Detection and Validation Engine for GrievanceGPT.
Distinguishes between Greetings, Ambiguous Inputs, Civic Questions (RAG),
Genuine Civic Complaints, and Irrelevant Messages across English, Tamil, and Tanglish.
Ensures intent validation acts as the gateway to both RAG retrieval and ML classification.
"""
import re
from typing import Dict, Any, List, Optional
from src.nlp.preprocessing import detect_language, normalize_text

# Common Greeting Tokens (English, Tamil, Tanglish)
GREETING_TOKENS = {
    "hi", "hello", "hey", "hai", "helo", "vanakkam", "வணக்கம்", "namaste", "namaskar",
    "namaskaram", "good morning", "good evening", "good afternoon", "howdy", "sup",
    "kaalai vanakkam", "காலை வணக்கம்", "maalai vanakkam", "மாலை வணக்கம்", "nalama",
    "நலமா", "epdi irukinga", "eppadi irukinga", "eppadi irukkeenga", "nalam",
    "who are you", "how are you", "what is your name", "greetings", "hi there", "hello there"
}

# Domain Specific Civic Problem Indicators
CIVIC_PROBLEM_KEYWORDS = [
    # Water Supply
    "water", "தண்ணீர்", "குடிநீர்", "thanni", "thanneer", "tap", "pipeline", "pipe",
    "borewell", "motor", "tanker", "drinking water", "water supply", "water cut",
    # Roads & Traffic
    "road", "pothole", "சாலை", "குழி", "kuzhi", "potholes", "tar road", "asphalt",
    "pavement", "footpath", "speed breaker", "broken road",
    # Electricity & Street Lighting
    "electricity", "power", "current", "voltage", "transformer", "electric", "pole",
    "wire", "power cut", "load shedding", "low voltage", "live wire",
    "light", "streetlight", "street light", "விளக்கு", "bulb", "lamp",
    # Sanitation & Solid Waste
    "garbage", "kuppai", "குப்பை", "waste", "trash", "rubbish", "dustbin", "bin",
    "dump", "dumping", "cleaning", "sweep", "litter",
    # Drainage & Sewerage
    "drain", "sewage", "sewer", "saakadai", "சாக்கடை", "drainage", "manhole",
    "gutter", "overflow", "stagnant water", "sewerage",
    # Public Health & Animal Control
    "mosquito", "fogging", "dengue", "malaria", "stray dog", "dog bite", "dead animal",
    # Revenue & Citizen Services
    "ration", "patta", "chitta", "certificate", "birth certificate", "death certificate", "property tax"
]

# Ambiguous / Generic Problem Indicators (where domain is not specified)
AMBIGUOUS_PROBLEM_KEYWORDS = [
    "problem", "issue", "complaint", "trouble", "broken", "damaged", "fault", "defect",
    "help", "assist", "assistance", "solve", "fix", "repair", "not working", "bad",
    "something wrong", "something is wrong", "பிரச்சனை", "சிக்கல்", "உதவி", "பிரச்சினை",
    "need help", "have an issue", "facing an issue", "near my house", "in my area", "near me",
    "urgent help", "please help", "problem here", "issue here"
]

# Question Starters / Interrogatives
QUESTION_STARTERS = [
    "what", "which", "how", "where", "who", "when", "why", "can i", "could you", "is there",
    "tell me", "explain", "procedure", "process", "guidelines", "rules", "faq", "information",
    "timings", "hours", "contact", "helpline", "officer", "steps to", "how to",
    "எப்படி", "என்ன", "எங்கு", "யார்", "விவரம்"
]

# General Civic Administration Terms
CIVIC_TERMS = [
    "municipality", "municipal", "corporation", "civic", "department", "services", "office",
    "collector", "ward", "panchayat", "government", "govt", "complaint", "grievance", "portal",
    "guideline", "guidelines", "policy", "cpgrams", "helpline", "officer", "scheme"
]

# Specific indicators that a message describes an ongoing local incident/complaint
INCIDENT_MARKERS = [
    "in my street", "near my house", "in our area", "enga area", "days ah", "for 4 days",
    "weeks ah", "not working since", "broken for", "நாட்களாக", "ஆயிற்று", "எடுக்கவில்லை",
    "varala", "வரல", "வரவில்லை", "4 days", "2 weeks", "since yesterday", "from last week"
]

def validate_intent(
    text: str,
    accumulated_data: Optional[Dict[str, Any]] = None,
    session_history: Optional[List[Dict[str, str]]] = None
) -> Dict[str, Any]:
    """
    Validate user message intent before RAG retrieval and ML classification.
    Returns:
        dict containing 'intent', 'confidence', 'language', 'normalized_text', and 'reason'.
    Possible intent values:
        - 'greeting': Salutations and pleasantries.
        - 'ambiguous': Citizen mentions having an issue/problem but omits the civic domain.
        - 'civic_question': Informational/procedural inquiry suitable for RAG.
        - 'complaint': Genuine civic grievance reporting active failure/fault.
        - 'irrelevant': Out-of-domain input unrelated to municipal civic functions.
    """
    norm_text = normalize_text(text)
    cleaned = norm_text.lower().strip()
    language = detect_language(text)
    words = re.findall(r'\b[\w\u0B80-\u0BFF]+\b', cleaned)

    if not cleaned:
        return {
            "intent": "greeting",
            "confidence": 1.0,
            "language": language,
            "normalized_text": norm_text,
            "reason": "empty_input"
        }

    # If this is a follow-up turn in an active complaint session, preserve complaint continuity
    if accumulated_data and accumulated_data.get("problem") and accumulated_data.get("problem") not in ("Public grievance issue", "None"):
        return {
            "intent": "complaint",
            "confidence": 0.95,
            "language": language,
            "normalized_text": norm_text,
            "reason": "active_complaint_continuation"
        }

    has_civic_prob = any(k in cleaned for k in CIVIC_PROBLEM_KEYWORDS)
    has_civic_terms = any(k in cleaned for k in CIVIC_TERMS) or has_civic_prob
    has_ambig = any(k in cleaned for k in AMBIGUOUS_PROBLEM_KEYWORDS)
    is_incident = any(k in cleaned for k in INCIDENT_MARKERS)

    # 1. GREETING CHECK
    # Pure greeting check: either matches token list or consists only of short greeting words
    is_greeting_match = cleaned in GREETING_TOKENS or (
        len(words) <= 3 and any(w in GREETING_TOKENS for w in words)
    )
    if is_greeting_match and not has_civic_prob and not is_incident:
        return {
            "intent": "greeting",
            "confidence": 0.99,
            "language": language,
            "normalized_text": norm_text,
            "reason": "salutation_detected"
        }

    # 2. CIVIC QUESTION CHECK (RAG Candidate)
    is_question_syntax = text.strip().endswith('?') or any(
        cleaned.startswith(q) or f" {q} " in f" {cleaned} " for q in QUESTION_STARTERS
    )
    if is_question_syntax and has_civic_terms and not is_incident:
        return {
            "intent": "civic_question",
            "confidence": 0.92,
            "language": language,
            "normalized_text": norm_text,
            "reason": "civic_inquiry_for_rag"
        }

    # 3. GENUINE CIVIC COMPLAINT CHECK
    if has_civic_prob:
        return {
            "intent": "complaint",
            "confidence": 0.95,
            "language": language,
            "normalized_text": norm_text,
            "reason": "civic_fault_detected"
        }

    # 4. AMBIGUOUS INPUT CHECK
    if has_ambig:
        return {
            "intent": "ambiguous",
            "confidence": 0.90,
            "language": language,
            "normalized_text": norm_text,
            "reason": "vague_problem_statement"
        }

    # 5. GENERAL QUESTION (Check if civic question without explicit keywords)
    if is_question_syntax and has_civic_terms:
        return {
            "intent": "civic_question",
            "confidence": 0.85,
            "language": language,
            "normalized_text": norm_text,
            "reason": "general_civic_question"
        }

    # 6. IRRELEVANT / OUT-OF-DOMAIN
    return {
        "intent": "irrelevant",
        "confidence": 0.85,
        "language": language,
        "normalized_text": norm_text,
        "reason": "out_of_domain_input"
    }
