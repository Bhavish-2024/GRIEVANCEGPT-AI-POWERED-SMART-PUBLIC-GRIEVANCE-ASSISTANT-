"""
Preprocessing and normalization utilities for English, Tamil, Tanglish, and Code-Mixed text.
"""
import re
import unicodedata
from typing import Tuple, Dict

# Tamil Unicode range: U+0B80 to U+0BFF
TAMIL_RANGE_REGEX = re.compile(r'[\u0B80-\u0BFF]')
LATIN_RANGE_REGEX = re.compile(r'[a-zA-Z]')

# Common Tanglish civic colloquial mappings to help standardize intent and problem terms
TANGLISH_CIVIC_MAP: Dict[str, str] = {
    "varala": "வரவில்லை not coming",
    "vara": "வர",
    "varale": "வரவில்லை",
    "ila": "இல்லை not available",
    "illa": "இல்லை not available",
    "illai": "இல்லை",
    "kadikala": "கிடைக்கவில்லை not getting",
    "romba": "ரொம்ப very much",
    "roam": "ரொம்ப",
    "inga": "இங்கே here",
    "enga": "எங்கள் our",
    "ungal": "உங்கள் your",
    "kuzhi": "குழி pothole",
    "kooli": "குழி",
    "thanni": "தண்ணீர் water",
    "thanneer": "தண்ணீர் water",
    "kuppai": "குப்பை garbage",
    "velai": "வேலை work",
    "poluthu": "பொழுது",
    "neram": "நேரம்",
    "seekiram": "சீக்கிரம் quickly",
    "udane": "உடனே immediately",
    "naal": "நாள் day",
    "naala": "நாட்களாக for days",
    "days ah": "நாட்களாக for days",
    "days aachu": "நாட்களாக ஆயிற்று",
}

def detect_language(text: str) -> str:
    """
    Detect the predominant language script and style of the text.
    Returns: 'tamil', 'tanglish', 'english', or 'code_mixed'
    """
    if not text or not text.strip():
        return "english"
        
    has_tamil = bool(TAMIL_RANGE_REGEX.search(text))
    has_latin = bool(LATIN_RANGE_REGEX.search(text))
    
    if has_tamil and has_latin:
        return "code_mixed"
    elif has_tamil:
        return "tamil"
    else:
        # Check if latin text contains distinct Tanglish phonetics
        lowered = text.lower()
        tanglish_words = ["enga", "varala", "illa", "thanni", "kuppai", "romba", "kuzhi", "naala", "mudila", "pannunga", "aachu", "ennoda"]
        if any(w in lowered for w in tanglish_words):
            return "tanglish"
        return "english"

def normalize_text(text: str) -> str:
    """
    Normalize text:
    - Unicode NFC normalization
    - Trim excessive whitespace
    - Lowercase latin text
    - Retain Tamil characters and punctuation marks
    """
    if not text:
        return ""
        
    # Unicode normalize
    text = unicodedata.normalize("NFC", text)
    
    # Standardize whitespace and newlines
    text = re.sub(r'[\r\n\t]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def preprocess_for_classification(text: str) -> str:
    """
    Enhanced normalization for machine learning classification.
    Expands common colloquial Tanglish contractions to improve lexical overlap.
    """
    cleaned = normalize_text(text).lower()
    
    # Light expansion of known Tanglish civic markers
    words = cleaned.split()
    expanded_words = []
    for w in words:
        # strip punctuation for lookup
        stripped = re.sub(r'[^\w\s]', '', w)
        if stripped in TANGLISH_CIVIC_MAP:
            expanded_words.append(TANGLISH_CIVIC_MAP[stripped])
        else:
            expanded_words.append(w)
            
    return " ".join(expanded_words)
