"""
Integration tests for GrievanceGPT FastAPI application and endpoints.
"""
from fastapi.testclient import TestClient
from src.database import init_db
from app import app

init_db()
client = TestClient(app)

def test_home_page():
    response = client.get("/")
    assert response.status_code == 200
    assert "GrievanceGPT" in response.text

def test_health_check():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["models_ready"] is True

def test_chat_dialogue_missing_info():
    # Test case from prompt: "எங்கள் area ல 4 days ah water வரல."
    payload = {
        "message": "எங்கள் area ல 4 days ah water வரல.",
        "session_id": "test_sess_001"
    }
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    # Verifications per user spec:
    # 1. understands the complaint as a water-supply grievance
    # 2. recognizes Tamil/Tanglish/codemixed input
    # 3. identifies missing location information
    # 4. asks for the location
    assert data["classification"]["category"] == "water_supply"
    assert data["classification"]["department"] == "Water Supply Department"
    assert data["completeness"]["is_complete"] is False
    assert "Exact location / Landmark" in data["completeness"]["missing_fields"]
    assert "location" in data["assistant_reply"].lower() or "locality" in data["assistant_reply"].lower()

def test_chat_dialogue_complete_grievance():
    # User provides location in second turn
    payload = {
        "message": "The water disruption is on 5th Cross Road, Gandhi Nagar, Katpadi.",
        "session_id": "test_sess_001",
        "accumulated_data": {
            "problem": "Drinking water supply disruption or shortage",
            "category": "water_supply",
            "department": "Water Supply Department",
            "duration": "4 days",
            "urgency": "high"
        }
    }
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    assert data["completeness"]["is_complete"] is True
    assert data["structured_grievance"] is not None
    assert "Gandhi Nagar" in data["structured_grievance"]["location"]

def test_completeness_audit_endpoint():
    response = client.post("/api/completeness/audit", json={"text": "Road broken near Katpadi for 2 weeks."})
    assert response.status_code == 200
    data = response.json()
    assert "score" in data
    assert "missing_fields" in data

def test_rag_query_endpoint():
    response = client.post("/api/rag/query", json={"query": "Which department handles water supply complaints?"})
    assert response.status_code == 200
    data = response.json()
    assert len(data["answer"]) > 10
    assert "water_supply.txt" in data["sources"] or "departments.txt" in data["sources"]

def test_analytics_endpoint():
    response = client.get("/api/analytics")
    assert response.status_code == 200
    data = response.json()
    assert "total_grievances" in data
    assert "departments" in data
