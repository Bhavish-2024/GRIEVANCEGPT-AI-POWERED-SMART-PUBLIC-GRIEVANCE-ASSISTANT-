"""
Comprehensive Unit & Integration Tests for Intent Validation + RAG Gateway in GrievanceGPT.
Verifies the 5 acceptance criteria:
1. "Hi" -> Friendly greeting; no department classification, no RAG retrieval, no complaint draft.
2. "What services does the municipality provide?" -> Civic information retrieved through RAG with citations.
3. "There has been no water supply for four days in my street." -> Validated complaint, accurate classification, complaint intake.
4. "There is a problem near my house." -> Clarification requested before complaint-specific retrieval.
5. Civic question with no relevant knowledge-base result -> Honest fallback acknowledgment.
"""
import pytest
from fastapi.testclient import TestClient
from app import app
from src.nlp.intent import validate_intent
from src.nlp.classifier import classifier
from src.rag.qa import answer_civic_question
from src.rag.retriever import retriever
from src.grievance.analyzer import analyze_grievance_text

client = TestClient(app)

def test_intent_greeting_english_and_tamil():
    """Verify greeting recognition in English, Tamil, and Tanglish."""
    for phrase in ["hi", "Hi", "Hello", "vanakkam", "வணக்கம்", "namaste", "good morning"]:
        res = validate_intent(phrase)
        assert res["intent"] == "greeting", f"Failed for '{phrase}': got {res['intent']}"
        assert res["confidence"] >= 0.90

def test_greeting_not_classified_as_electricity():
    """
    CRITICAL FIX TEST:
    Verify 'hi' is NOT classified as Electricity Department or any complaint department.
    """
    # 1. Test direct classifier interface
    pred = classifier.predict("hi")
    assert pred["intent"] == "greeting"
    assert pred["department"] is None, f"Expected None department, got {pred['department']}"
    assert pred["category"] is None
    assert pred["urgency"] is None

    # 2. Test full analyzer interface
    analyzed = analyze_grievance_text("hi")
    assert analyzed["workflow_stage"] == "GREETING"
    assert analyzed["classification"]["department"] is None
    assert analyzed["structured_grievance"] is None
    assert "rag" not in analyzed or analyzed["rag"] is None
    assert "vanakkam" in analyzed["assistant_reply"].lower() or "welcome" in analyzed["assistant_reply"].lower()

    # 3. Test API endpoint /api/chat
    response = client.post("/api/chat", json={"message": "hi", "session_id": "test_greeting_session"})
    assert response.status_code == 200
    data = response.json()
    assert data["workflow_stage"] == "GREETING"
    assert data["classification"]["department"] is None
    assert data["structured_grievance"] is None

def test_civic_information_question_rag():
    """
    TEST CASE 2:
    'What services does the municipality provide?' -> Civic information retrieved through RAG.
    """
    query = "What services does the municipality provide?"
    
    # 1. Test RAG QA directly
    rag_res = answer_civic_question(query)
    assert len(rag_res["sources"]) > 0, "Expected at least one civic source document citation"
    assert len(rag_res["chunks"]) > 0
    assert len(rag_res["answer"]) > 20

    # 2. Test chat integration
    analyzed = analyze_grievance_text(query)
    assert analyzed["workflow_stage"] == "CIVIC_INFO"
    assert analyzed["structured_grievance"] is None
    assert analyzed["rag"] is not None
    assert len(analyzed["rag"]["sources"]) > 0
    assert "grievance_procedure.txt" in analyzed["rag"]["sources"] or "faq.txt" in analyzed["rag"]["sources"] or "departments.txt" in analyzed["rag"]["sources"]

    # 3. Test API endpoint
    response = client.post("/api/chat", json={"message": query, "session_id": "test_rag_session"})
    assert response.status_code == 200
    data = response.json()
    assert data["workflow_stage"] == "CIVIC_INFO"
    assert "guidance" in data["assistant_reply"].lower() or "steps" in data["assistant_reply"].lower() or "department" in data["assistant_reply"].lower()

def test_genuine_water_supply_complaint():
    """
    TEST CASE 3:
    'There has been no water supply for four days in my street.' -> Validated complaint, accurate classification, complaint intake.
    """
    text = "There has been no water supply for four days in my street."
    
    # 1. Intent validation
    intent_res = validate_intent(text)
    assert intent_res["intent"] == "complaint"

    # 2. Classifier validation
    pred = classifier.predict(text)
    assert pred["category"] == "water_supply"
    assert pred["department"] == "Water Supply Department"
    assert pred["urgency"] == "high"

    # 3. Full analyzer validation
    analyzed = analyze_grievance_text(text)
    assert analyzed["workflow_stage"] == "INFORMATION_COLLECTION"
    assert analyzed["classification"]["department"] == "Water Supply Department"
    assert analyzed["classification"]["category"] == "water_supply"
    assert analyzed["extracted_entities"]["duration"] == "4 days"
    # Location was only 'my street' so it correctly identifies missing specific location
    assert analyzed["completeness"]["is_complete"] is False

def test_ambiguous_input_clarification():
    """
    TEST CASE 4:
    'There is a problem near my house.' -> Ask for clarification before complaint-specific retrieval.
    """
    text = "There is a problem near my house."
    
    # 1. Intent validation
    intent_res = validate_intent(text)
    assert intent_res["intent"] == "ambiguous"

    # 2. Full analyzer validation
    analyzed = analyze_grievance_text(text)
    assert analyzed["workflow_stage"] == "CLARIFICATION"
    assert analyzed["classification"]["department"] is None
    assert analyzed["structured_grievance"] is None
    # Clarification asked politely with examples
    reply_lower = analyzed["assistant_reply"].lower()
    assert "specify" in reply_lower or "what kind" in reply_lower or "for example" in reply_lower
    assert "water" in reply_lower and "potholes" in reply_lower

    # 3. API endpoint
    response = client.post("/api/chat", json={"message": text, "session_id": "test_ambig_session"})
    assert response.status_code == 200
    data = response.json()
    assert data["workflow_stage"] == "CLARIFICATION"
    assert data["classification"]["department"] is None

def test_unanswerable_civic_question_honest_fallback():
    """
    TEST CASE 5:
    A civic question with no relevant knowledge-base result -> Honest fallback response.
    """
    unrelated_query = "What is the municipal policy on pet dinosaurs?"
    
    # 1. Test RAG QA module
    rag_res = answer_civic_question(unrelated_query)
    assert rag_res["sources"] == [], "Should have no sources for dinosaur query"
    assert rag_res["chunks"] == []
    assert "could not find relevant information" in rag_res["answer"].lower()

    # 2. Test chat integration
    analyzed = analyze_grievance_text(unrelated_query)
    assert analyzed["workflow_stage"] == "CIVIC_INFO"
    assert analyzed["rag"]["sources"] == []
    assert "could not find relevant information" in analyzed["assistant_reply"].lower()

def test_streetlight_on_road_not_misclassified_as_road_damage():
    """
    CRITICAL USER PROMPT TEST:
    'Street light is not working on 5th Cross Road, Gandhi Nagar since last Friday.'
    1. Must NOT classify as 'Road damage/potholes' or 'Roads & Street Lighting Department'
    2. Must classify as street_lighting -> 'Civic Maintenance & Street Lighting'
    3. Must extract duration as 'since last Friday'
    4. Must NOT ask for duration again, only District.
    """
    text = "Street light is not working on 5th Cross Road, Gandhi Nagar since last Friday."
    
    # Classification check
    clf = classifier.predict(text)
    assert clf["category"] == "street_lighting", f"Expected street_lighting, got {clf['category']}"
    assert clf["department"] == "Civic Maintenance & Street Lighting", f"Expected Civic Maintenance & Street Lighting, got {clf['department']}"

    # Analyzer check
    analyzed = analyze_grievance_text(text)
    assert analyzed["workflow_stage"] == "INFORMATION_COLLECTION"
    assert analyzed["classification"]["department"] == "Civic Maintenance & Street Lighting"
    assert analyzed["classification"]["category"] == "street_lighting"
    assert "road damage" not in analyzed["extracted_entities"]["problem"].lower()
    assert analyzed["extracted_entities"]["problem"] == "Defective non-functional street lights"
    assert analyzed["extracted_entities"]["duration"] == "since last Friday"
    assert "5th Cross Road" in analyzed["extracted_entities"]["location"]
    assert analyzed["extracted_entities"]["location"] == "5th Cross Road, Gandhi Nagar"
    
    # Missing fields should ONLY be District
    assert "District (e.g. Chennai, Vellore, Coimbatore)" in analyzed["completeness"]["missing_fields"]
    assert "Duration (e.g. 4 days, 2 weeks)" not in analyzed["completeness"]["missing_fields"]
    
    # Verify assistant reply does not mention road damage and does not ask for duration
    reply = analyzed["assistant_reply"]
    assert "road damage" not in reply.lower()
    assert "pothole" not in reply.lower()
    assert "duration" not in reply.lower() or "how long" not in reply.lower()
    assert "District" in reply

