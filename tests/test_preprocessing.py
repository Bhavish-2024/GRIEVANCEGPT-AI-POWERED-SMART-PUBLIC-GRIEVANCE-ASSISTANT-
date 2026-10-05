"""
Unit tests for text preprocessing and language detection.
"""
import pytest
from src.nlp.preprocessing import detect_language, normalize_text, preprocess_for_classification

def test_language_detection_tamil():
    text = "குப்பை 3 நாட்களாக எடுக்கவில்லை"
    assert detect_language(text) == "tamil"

def test_language_detection_tanglish():
    text = "enga area la water varala romba kastam"
    assert detect_language(text) == "tanglish"

def test_language_detection_english():
    text = "There is a deep pothole near the bus stand."
    assert detect_language(text) == "english"

def test_language_detection_code_mixed():
    text = "எங்கள் area ல 4 days ah water வரல"
    assert detect_language(text) == "code_mixed"

def test_normalize_text():
    raw = "  Hello   world \n\n from   Chennai  "
    norm = normalize_text(raw)
    assert norm == "Hello world from Chennai"

def test_preprocess_for_classification():
    text = "enga area la water varala"
    processed = preprocess_for_classification(text)
    assert "வரவில்லை" in processed or "water" in processed
