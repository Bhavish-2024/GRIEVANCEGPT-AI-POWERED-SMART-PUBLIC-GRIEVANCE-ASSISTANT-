"""
Unit tests for Grievance Completeness Checker.
"""
from src.grievance.completeness import evaluate_completeness

def test_incomplete_grievance():
    # Only problem mentioned, no location
    extracted = {
        "problem": "No water supply",
        "location": "pending confirmation",
        "category": "water_supply",
        "duration": "Unspecified"
    }
    result = evaluate_completeness(extracted)
    assert result["is_complete"] is False
    assert result["score"] < 70
    assert "Exact location / Landmark" in result["missing_fields"]
    assert len(result["suggestions"]) > 0

def test_complete_grievance():
    # Problem, location, category, duration provided
    extracted = {
        "problem": "Severe drinking water outage",
        "location": "5th Cross Road, Gandhi Nagar, Katpadi",
        "category": "water_supply",
        "affected_service": "Drinking Water Supply",
        "duration": "4 days",
        "district": "Vellore",
        "description": "Entire street has had zero water pressure and taps are dry for 4 days."
    }
    result = evaluate_completeness(extracted)
    assert result["is_complete"] is True
    assert result["score"] >= 70
    assert len(result["missing_fields"]) == 0
