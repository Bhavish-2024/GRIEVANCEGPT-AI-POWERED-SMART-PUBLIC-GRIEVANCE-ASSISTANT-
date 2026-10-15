"""
Comprehensive End-to-End System Testing for GrievanceGPT Complaint Intake,
Validation, Intent Detection, Follow-up Questions, Submission, and Tracking.

Guarantees:
- Tests real backend endpoints and ML pipelines (Zero Mock Data).
- Uses isolated temporary SQLite database per test run (Zero Production DB Contamination).
- Validates acceptable and unacceptable inputs across English, Tamil, and Tanglish.
- Verifies input validation, greetings, gibberish, ambiguous inputs, missing fields,
  multi-turn recovery, draft generation, confirmation, tracking, and access control.
"""
import os
import sqlite3
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

import src.config as config_module
from src.config import ADMIN_PIN
import src.database as db_module
from src.database import (
    init_db,
    save_grievance,
    get_grievance,
    get_grievance_for_tracker
)
from app import app
from src.nlp.intent import validate_intent
from src.nlp.classifier import classifier
from src.grievance.analyzer import analyze_grievance_text

client = TestClient(app)

PROD_DB_PATH = Path("data/grievances.db").resolve()

@pytest.fixture(autouse=True)
def isolated_test_db(monkeypatch, tmp_path):
    """
    Provides an isolated SQLite database for each test.
    Ensures zero false or test complaints ever touch data/grievances.db.
    """
    test_db_path = tmp_path / "test_isolated_intake.db"
    monkeypatch.setattr(db_module, "DATABASE_PATH", test_db_path)
    monkeypatch.setattr(config_module, "DATABASE_PATH", test_db_path)
    
    # Snapshot initial state of production database if it exists
    prod_mtime_before = PROD_DB_PATH.stat().st_mtime if PROD_DB_PATH.exists() else None
    prod_size_before = PROD_DB_PATH.stat().st_size if PROD_DB_PATH.exists() else None

    # Initialize fresh empty isolated schema
    init_db(seed=False)
    
    yield test_db_path
    
    # Teardown: Verify production database was never touched
    if PROD_DB_PATH.exists() and prod_mtime_before is not None:
        prod_mtime_after = PROD_DB_PATH.stat().st_mtime
        prod_size_after = PROD_DB_PATH.stat().st_size
        assert prod_mtime_after == prod_mtime_before, "CRITICAL: Production DB modified during test!"
        assert prod_size_after == prod_size_before, "CRITICAL: Production DB size changed during test!"


# ─── 1. Empty Text and Whitespace Inputs ───────────────────────────────────────
def test_e2e_empty_and_whitespace_inputs():
    """
    Submitting empty text, whitespace, or tabs should not crash,
    must not classify into any department, and must provide guiding prompt.
    """
    for bad_input in ["", "   ", "\t  \n"]:
        # Direct intent check
        intent_info = validate_intent(bad_input)
        assert intent_info["intent"] == "greeting"
        assert intent_info["reason"] == "empty_input"
        
        # Analyzer check
        result = analyze_grievance_text(bad_input)
        assert result["workflow_stage"] == "GREETING"
        assert result["classification"]["department"] is None
        assert result["classification"]["category"] is None
        assert result["structured_grievance"] is None
        assert "describe" in result["assistant_reply"].lower() or "enter" in result["assistant_reply"].lower()

        # API chat check
        resp = client.post("/api/chat", json={"message": bad_input, "session_id": "test_empty_session"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["workflow_stage"] == "GREETING"
        assert data["classification"]["department"] is None
        assert data["structured_grievance"] is None


# ─── 2. Random Characters, Symbols, and Gibberish ──────────────────────────────
def test_e2e_random_characters_and_symbols():
    """
    Submitting gibberish, keyboard mashing, numbers, or symbols must be detected
    as out-of-domain/irrelevant and not trigger complaint intake or department assignment.
    """
    unacceptable_inputs = [
        "asdfghjkl",
        "!@#$%^&*()_+",
        "1234567890",
        "zzzzzzqqqqqq",
        "qwertyuiop asdfghjkl zxcvbnm"
    ]
    for gibberish in unacceptable_inputs:
        intent_info = validate_intent(gibberish)
        assert intent_info["intent"] == "irrelevant", f"Expected irrelevant for '{gibberish}', got {intent_info['intent']}"
        
        result = analyze_grievance_text(gibberish)
        assert result["workflow_stage"] == "IRRELEVANT"
        assert result["classification"]["department"] is None
        assert result["structured_grievance"] is None
        assert "dedicated to public municipal" in result["assistant_reply"]

        resp = client.post("/api/chat", json={"message": gibberish, "session_id": "test_gibberish_session"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["workflow_stage"] == "IRRELEVANT"
        assert data["classification"]["department"] is None


# ─── 3. Multilingual Greetings (English, Tamil, Tanglish) ──────────────────────
def test_e2e_multilingual_greetings():
    """
    Greetings in English, Tamil, and Tanglish must receive friendly responses
    without department classification, RAG lookup, or complaint drafts.
    """
    greetings = [
        ("hello", "english"),
        ("hi there", "english"),
        ("good morning", "english"),
        ("வணக்கம்", "tamil"),
        ("காலை வணக்கம்", "tamil"),
        ("vanakkam", "tanglish"),
        ("epdi irukinga", "tanglish")
    ]
    for text, lang_type in greetings:
        intent_info = validate_intent(text)
        assert intent_info["intent"] == "greeting", f"Greeting failed for '{text}'"
        
        result = analyze_grievance_text(text)
        assert result["workflow_stage"] == "GREETING"
        assert result["classification"]["department"] is None
        assert result["classification"]["category"] is None
        assert result["structured_grievance"] is None
        
        reply = result["assistant_reply"]
        if lang_type in ("tamil", "tanglish"):
            assert "வணக்கம்" in reply or "Vanakkam" in reply
        else:
            assert "Hello" in reply or "Welcome" in reply

        resp = client.post("/api/chat", json={"message": text, "session_id": f"test_greet_{lang_type}"})
        assert resp.status_code == 200
        assert resp.json()["workflow_stage"] == "GREETING"


# ─── 4. Ambiguous Complaints Prompting Clarification ───────────────────────────
def test_e2e_ambiguous_complaints_prompt_clarification():
    """
    Generic problem statements lacking a specific civic domain (water, roads, etc.)
    must trigger CLARIFICATION stage with domain examples before any classification.
    """
    ambiguous_inputs = [
        "There is a problem near my house.",
        "Something is broken in our area, please fix it.",
        "En area la oru problem irukku help pannunga",
        "We are facing a major issue here."
    ]
    for ambig in ambiguous_inputs:
        intent_info = validate_intent(ambig)
        assert intent_info["intent"] == "ambiguous", f"Expected ambiguous for '{ambig}', got {intent_info['intent']}"
        
        result = analyze_grievance_text(ambig)
        assert result["workflow_stage"] == "CLARIFICATION"
        assert result["classification"]["department"] is None
        assert result["structured_grievance"] is None
        assert "specify what kind of civic problem" in result["assistant_reply"]
        assert "Drinking Water" in result["assistant_reply"]
        assert "Roads & Pavements" in result["assistant_reply"]

        resp = client.post("/api/chat", json={"message": ambig, "session_id": "test_ambig_sess"})
        assert resp.status_code == 200
        assert resp.json()["workflow_stage"] == "CLARIFICATION"


# ─── 5. Incomplete Complaint & Multi-Turn State Merging ───────────────────────
def test_e2e_incomplete_complaint_and_multiturn_recovery():
    """
    When essential details are missing (e.g. location/district), GrievanceGPT
    asks targeted follow-up questions. When the user provides them in turn 2,
    state merges cleanly and transitions to REVIEW stage with generated draft.
    """
    # Turn 1: Incomplete complaint missing location and district
    msg_turn1 = "There has been no water supply for 4 days."
    resp1 = client.post("/api/chat", json={"message": msg_turn1, "session_id": "turn_session_water"})
    assert resp1.status_code == 200
    data1 = resp1.json()
    
    assert data1["workflow_stage"] == "INFORMATION_COLLECTION"
    assert data1["classification"]["department"] == "Water Supply Department"
    assert data1["classification"]["category"] == "water_supply"
    assert data1["extracted_entities"]["duration"] == "4 days"
    assert data1["completeness"]["is_complete"] is False
    assert any("location" in field.lower() for field in data1["completeness"]["missing_fields"])
    assert any("district" in field.lower() for field in data1["completeness"]["missing_fields"])
    assert data1["structured_grievance"] is None
    assert "District" in data1["assistant_reply"]

    # Turn 2: Citizen provides missing Location and District
    msg_turn2 = "5th Cross Street, Gandhi Nagar, Katpadi"
    accumulated_from_turn1 = {
        **data1["extracted_entities"],
        "category": data1["classification"]["category"],
        "department": data1["classification"]["department"],
        "urgency": data1["classification"]["urgency"],
        "reference_id": data1["reference_id"]
    }
    
    resp2 = client.post("/api/chat", json={
        "message": msg_turn2,
        "session_id": "turn_session_water",
        "accumulated_data": accumulated_from_turn1
    })
    assert resp2.status_code == 200
    data2 = resp2.json()

    # Verify all details were merged and grievance is now COMPLETE in REVIEW stage
    assert data2["workflow_stage"] == "REVIEW"
    assert data2["classification"]["department"] == "Water Supply Department"
    assert data2["classification"]["category"] == "water_supply"
    assert data2["extracted_entities"]["duration"] == "4 days"
    assert "Gandhi Nagar" in data2["extracted_entities"]["location"]
    assert data2["extracted_entities"]["district"] == "Katpadi"
    assert data2["completeness"]["is_complete"] is True
    assert data2["structured_grievance"] is not None
    assert data2["reference_id"].startswith("GRV-")


# ─── 6. Streetlight vs Road Defect Classification & Since Friday ──────────────
def test_e2e_streetlight_address_vs_road_defect():
    """
    Test distinction between street addresses containing 'Road' and actual road defects.
    'Street light is not working on 5th Cross Road, Gandhi Nagar since last Friday.'
    - Must classify under Civic Maintenance & Street Lighting
    - Must NOT classify as Road damage/potholes
    - Must extract duration 'since last Friday'
    - Must ask ONLY for District in turn 1
    """
    text1 = "Street light is not working on 5th Cross Road, Gandhi Nagar since last Friday."
    resp1 = client.post("/api/chat", json={"message": text1, "session_id": "streetlight_session"})
    assert resp1.status_code == 200
    data1 = resp1.json()

    assert data1["workflow_stage"] == "INFORMATION_COLLECTION"
    assert data1["classification"]["department"] == "Civic Maintenance & Street Lighting"
    assert data1["classification"]["category"] == "street_lighting"
    assert "road damage" not in data1["extracted_entities"]["problem"].lower()
    assert data1["extracted_entities"]["duration"] == "since last Friday"
    assert "5th Cross Road" in data1["extracted_entities"]["location"]
    
    # Missing fields must contain District but NOT Duration
    missing = data1["completeness"]["missing_fields"]
    assert any("District" in f for f in missing)
    assert not any("Duration" in f for f in missing)

    # Turn 2: Citizen supplies District
    accumulated = {
        **data1["extracted_entities"],
        "category": data1["classification"]["category"],
        "department": data1["classification"]["department"],
        "urgency": data1["classification"]["urgency"],
        "reference_id": data1["reference_id"]
    }
    resp2 = client.post("/api/chat", json={
        "message": "Vellore",
        "session_id": "streetlight_session",
        "accumulated_data": accumulated
    })
    assert resp2.status_code == 200
    data2 = resp2.json()
    assert data2["workflow_stage"] == "REVIEW"
    assert data2["extracted_entities"]["district"] == "Vellore"
    assert data2["extracted_entities"]["duration"] == "since last Friday"
    assert data2["completeness"]["is_complete"] is True


# ─── 7. Full English Complaint Intake to Confirmation and Tracking ─────────────
def test_e2e_english_complaint_full_cycle():
    """
    End-to-end cycle for a complete English complaint:
    1. Single-turn intake reaches REVIEW stage
    2. Confirmation persists to isolated SQLite DB
    3. Citizen tracks grievance using reference ID
    4. Unauthorized public access to staff registry is blocked (401)
    """
    text = "Drinking water supply disrupted on 10th Cross Street, Anna Nagar, Chennai for past 5 days."
    resp_chat = client.post("/api/chat", json={"message": text, "session_id": "full_en_session"})
    assert resp_chat.status_code == 200
    chat_data = resp_chat.json()

    assert chat_data["workflow_stage"] == "REVIEW"
    assert chat_data["classification"]["department"] == "Water Supply Department"
    assert chat_data["classification"]["category"] == "water_supply"
    assert "10th Cross Street" in chat_data["extracted_entities"]["location"]
    assert chat_data["extracted_entities"]["district"] == "Chennai"
    assert chat_data["extracted_entities"]["duration"] == "5 days"
    assert chat_data["completeness"]["is_complete"] is True

    draft = chat_data["structured_grievance"]
    ref_id = chat_data["reference_id"]

    # Confirm and Submit
    confirm_payload = {
        "grievance_id": ref_id,
        "subject": draft["subject"],
        "description": draft["description"],
        "location": draft["location"],
        "district": chat_data["extracted_entities"]["district"],
        "duration": chat_data["extracted_entities"]["duration"],
        "category": chat_data["classification"]["category"],
        "department": chat_data["classification"]["department"],
        "urgency": chat_data["classification"]["urgency"],
        "severity": chat_data["extracted_entities"].get("severity", "service_issue")
    }
    resp_confirm = client.post("/api/grievance/confirm", json=confirm_payload)
    assert resp_confirm.status_code == 200
    confirm_data = resp_confirm.json()
    assert confirm_data["status"] == "CONFIRMED"
    assert confirm_data["grievance_id"] == ref_id

    # Verify persisted record in isolated DB
    rec = get_grievance(ref_id)
    assert rec is not None
    assert rec["status"] == "CONFIRMED"
    assert rec["district"] == "Chennai"
    assert rec["department"] == "Water Supply Department"

    # Public Citizen Tracker verification
    resp_track = client.get(f"/api/track/{ref_id}")
    assert resp_track.status_code == 200
    track_data = resp_track.json()
    assert track_data["grievance_id"] == ref_id
    assert track_data["status"] == "CONFIRMED"
    assert track_data["department"] == "Water Supply Department"
    # Ensure safe public fields only
    assert "original_text" not in track_data
    assert "normalized_text" not in track_data

    # Invalid tracking ID returns 404
    resp_bad_track = client.get("/api/track/GRV-9999-999999")
    assert resp_bad_track.status_code == 404

    # Security check: public cannot list all grievances without admin PIN
    resp_unauth = client.get("/api/grievances")
    assert resp_unauth.status_code == 401

    # Authorized admin can list
    resp_auth = client.get("/api/grievances", headers={"X-Admin-PIN": ADMIN_PIN})
    assert resp_auth.status_code == 200
    glist = resp_auth.json()
    assert any(g["grievance_id"] == ref_id for g in glist)


# ─── 8. Full Tamil Complaint Intake and Submission ─────────────────────────────
def test_e2e_tamil_complaint_full_cycle():
    """
    End-to-end cycle for a complete Tamil complaint:
    'காட்பாடி பேருந்து நிலையம் அருகில் 2 வாரங்களாக சாலையில் பெரிய பள்ளங்கள் உள்ளன.'
    - Classifies under Roads & Infrastructure Department
    - Extracts Katpadi district, Tamil landmark location, and '2 weeks' duration
    - Reaches REVIEW stage, confirms, and stores in isolated DB
    """
    tamil_text = "காட்பாடி பேருந்து நிலையம் அருகில் 2 வாரங்களாக சாலையில் பெரிய பள்ளங்கள் உள்ளன."
    resp_chat = client.post("/api/chat", json={"message": tamil_text, "session_id": "full_tamil_session"})
    assert resp_chat.status_code == 200
    data = resp_chat.json()

    assert data["workflow_stage"] == "REVIEW"
    assert data["classification"]["department"] == "Roads & Infrastructure Department"
    assert data["classification"]["category"] == "road_maintenance"
    assert "காட்பாடி" in data["extracted_entities"]["location"]
    assert data["extracted_entities"]["district"] == "Katpadi"
    assert data["extracted_entities"]["duration"] == "2 weeks"
    assert data["completeness"]["is_complete"] is True

    draft = data["structured_grievance"]
    ref_id = data["reference_id"]

    # Confirm
    confirm_payload = {
        "grievance_id": ref_id,
        "subject": draft["subject"],
        "description": draft["description"],
        "location": draft["location"],
        "district": data["extracted_entities"]["district"],
        "duration": data["extracted_entities"]["duration"],
        "category": data["classification"]["category"],
        "department": data["classification"]["department"],
        "urgency": data["classification"]["urgency"],
        "severity": data["extracted_entities"].get("severity", "safety_hazard")
    }
    resp_confirm = client.post("/api/grievance/confirm", json=confirm_payload)
    assert resp_confirm.status_code == 200
    assert resp_confirm.json()["status"] == "CONFIRMED"

    # Verify saved correctly
    saved = get_grievance(ref_id)
    assert saved is not None
    assert saved["status"] == "CONFIRMED"
    assert saved["district"] == "Katpadi"
    assert saved["duration"] == "2 weeks"


# ─── 9. Full Tanglish Complaint Intake and Submission ──────────────────────────
def test_e2e_tanglish_complaint_full_cycle():
    """
    End-to-end cycle for a complete Tanglish complaint:
    'Enga area 5th Cross Street, Gandhi Nagar, Katpadi la 4 days ah water supply varala.'
    - Classifies under Water Supply Department
    - Extracts Katpadi, 4 days, and street location
    - Reaches REVIEW stage, confirms, and verifies tracking
    """
    tanglish_text = "Enga area 5th Cross Street, Gandhi Nagar, Katpadi la 4 days ah water supply varala."
    resp_chat = client.post("/api/chat", json={"message": tanglish_text, "session_id": "full_tanglish_session"})
    assert resp_chat.status_code == 200
    data = resp_chat.json()

    assert data["workflow_stage"] == "REVIEW"
    assert data["classification"]["department"] == "Water Supply Department"
    assert data["classification"]["category"] == "water_supply"
    assert "Gandhi Nagar" in data["extracted_entities"]["location"]
    assert data["extracted_entities"]["district"] == "Katpadi"
    assert data["extracted_entities"]["duration"] == "4 days"
    assert data["completeness"]["is_complete"] is True

    draft = data["structured_grievance"]
    ref_id = data["reference_id"]

    confirm_payload = {
        "grievance_id": ref_id,
        "subject": draft["subject"],
        "description": draft["description"],
        "location": draft["location"],
        "district": data["extracted_entities"]["district"],
        "duration": data["extracted_entities"]["duration"],
        "category": data["classification"]["category"],
        "department": data["classification"]["department"],
        "urgency": data["classification"]["urgency"],
        "severity": data["extracted_entities"].get("severity", "service_issue")
    }
    resp_confirm = client.post("/api/grievance/confirm", json=confirm_payload)
    assert resp_confirm.status_code == 200
    assert resp_confirm.json()["status"] == "CONFIRMED"

    # Tracker check
    resp_track = client.get(f"/api/track/{ref_id}")
    assert resp_track.status_code == 200
    assert resp_track.json()["status"] == "CONFIRMED"
