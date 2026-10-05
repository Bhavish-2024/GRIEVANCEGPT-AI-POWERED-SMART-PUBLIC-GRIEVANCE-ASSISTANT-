"""
Unit tests for structured grievance generator.
"""
from src.grievance.generator import generate_structured_grievance

def test_generate_structured_grievance():
    extracted = {
        "problem": "Severe water disruption",
        "location": "Gandhi Nagar, Katpadi",
        "district": "Vellore",
        "duration": "4 days",
        "additional_details": "No tanker supplied"
    }
    classification = {
        "category": "water_supply",
        "department": "Water Supply Department",
        "urgency": "high",
        "severity": "service_failure"
    }
    gid = "GRV-2026-000001"
    
    doc = generate_structured_grievance(extracted, classification, gid)
    
    assert doc["grievance_id"] == "GRV-2026-000001"
    assert "Water Supply" in doc["subject"] or "water" in doc["subject"].lower()
    assert doc["department"] == "Water Supply Department"
    assert doc["urgency"] == "High"
    assert "Gandhi Nagar" in doc["location"]
    assert "4 days" in doc["duration"]
    assert "Respected Sir/Madam" in doc["description"]
