"""
Unit tests for CivicDex ML classifiers.
"""
from src.nlp.classifier import classifier

def test_classifier_models_loaded():
    assert classifier.is_ready is True, "All 4 ML models should be loaded from disk"

def test_tamil_water_grievance_classification():
    # Final demonstration scenario test case
    text = "எங்கள் area ல 4 days ah water வரல."
    res = classifier.predict(text)
    
    assert res["intent"] in ["complaint", "escalation", "follow_up"]
    assert res["category"] == "water_supply"
    assert res["department"] == "Water Supply Department"
    assert res["urgency"] in ["high", "medium"]
    assert res["language"] in ["code_mixed", "tamil", "tanglish"]

def test_roads_grievance_classification():
    text = "Large potholes on Katpadi main road causing accidents"
    res = classifier.predict(text)
    
    assert res["category"] in ["road_maintenance", "roads"]
    assert "Roads" in res["department"]
    assert res["urgency"] in ["high", "medium"]

def test_sanitation_grievance_classification():
    text = "குப்பை 3 நாட்களாக எடுக்கவில்லை"
    res = classifier.predict(text)
    
    assert res["category"] in ["garbage_collection", "sanitation"]
    assert "Sanitation" in res["department"]
