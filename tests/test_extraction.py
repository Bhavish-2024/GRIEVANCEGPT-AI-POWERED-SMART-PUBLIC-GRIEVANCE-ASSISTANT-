"""
Unit tests for information extraction and entity parsing.
"""
from src.nlp.extraction import extract_grievance_entities, heuristic_extract

def test_extraction_duration_and_problem():
    text = "எங்கள் area ல 4 days ah water வரல."
    data = extract_grievance_entities(text)
    
    assert "water" in data["problem"].lower() or "குடிநீர்" in data["problem"]
    assert "4 days" in data["duration"]
    assert data["category"] == "water_supply"
    assert data["department"] == "Water Supply Department"

def test_extraction_location():
    text = "Big pothole near Katpadi railway station for 2 weeks"
    data = extract_grievance_entities(text)
    
    assert "Katpadi" in data["location"]
    assert "2 weeks" in data["duration"]
    assert data["category"] in ["road_maintenance", "roads"]
