"""
GrievanceGPT - AI-Powered Smart Public Grievance Assistant.
FastAPI Application serving modern web dashboard and full civic intelligence REST API.
Engineered for zero-overhead deployment on Render and local environments.
"""
import os
import uvicorn
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, Request, HTTPException, Header, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from src.config import (
    HOST,
    PORT,
    DEBUG,
    STATIC_DIR,
    TEMPLATES_DIR,
    DATABASE_PATH,
    CIVICDEX_CSV,
    RAG_INDEX_PATH,
    ADMIN_PIN
)
from src.database import (
    init_db,
    save_grievance,
    get_grievance,
    list_grievances,
    get_analytics_summary,
    save_conversation_turn,
    get_session_history,
    generate_grievance_id,
    update_grievance_by_admin,
    get_admin_dashboard_stats,
    get_grievance_for_tracker,
    VALID_STATUSES
)
from src.nlp.classifier import classifier
from src.nlp.extraction import extract_grievance_entities
from src.grievance.completeness import evaluate_completeness
from src.grievance.generator import generate_structured_grievance
from src.grievance.analyzer import analyze_grievance_text
from src.rag.qa import answer_civic_question
from src.rag.ingest import build_knowledge_index
from src.llm.ollama_client import ollama_client

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Ensure database tables, train models if missing, build RAG index if missing
    init_db()
    
    # Auto-train models if not yet generated
    if not classifier.is_ready:
        print("[Startup] Classifier models not found. Triggering automated CivicDex training...")
        try:
            from src.nlp.train_models import main as train_models_main
            train_models_main()
            classifier._load_models()
        except Exception as e:
            print(f"[Startup] Warning: Automated training error: {e}")

    # Auto-build RAG index if missing
    if not RAG_INDEX_PATH.exists():
        print("[Startup] Knowledge base index not found. Building RAG vector store...")
        try:
            build_knowledge_index()
        except Exception as e:
            print(f"[Startup] Warning: RAG index error: {e}")
            
    yield
    # Shutdown logic (if any)

app = FastAPI(
    title="GrievanceGPT",
    description="AI-Powered Smart Public Grievance Assistant API & Dashboard",
    version="1.0.0",
    lifespan=lifespan
)

# Static and Template mounts
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Request / Response Schemas
class ChatRequest(BaseModel):
    message: str
    session_id: str
    accumulated_data: Optional[Dict[str, Any]] = None

class CompletenessRequest(BaseModel):
    text: str

class RagQueryRequest(BaseModel):
    query: str

class GenerateRequest(BaseModel):
    extracted_data: Dict[str, Any]
    classification: Dict[str, Any]
    grievance_id: Optional[str] = None

class ConfirmRequest(BaseModel):
    grievance_id: str
    subject: str
    description: str
    location: str
    duration: Optional[str] = ""
    category: str
    department: str
    urgency: str
    severity: Optional[str] = "service_issue"

class AdminUpdateRequest(BaseModel):
    status: str
    admin_remarks: Optional[str] = ""

# Admin PIN Dependency
async def verify_admin(x_admin_pin: Optional[str] = Header(default=None)):
    """FastAPI dependency: validates X-Admin-PIN header for admin-protected routes."""
    if x_admin_pin != ADMIN_PIN:
        raise HTTPException(status_code=401, detail="Unauthorized: invalid admin PIN")

# Initialize SQLite database schema
init_db()

# Routes

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard(request: Request):
    """Render the Government-grade interactive Web Dashboard."""
    return templates.TemplateResponse(request=request, name="index.html")

@app.post("/api/chat")
async def api_chat(req: ChatRequest):
    """
    Core conversational intake endpoint:
    Processes user grievance message, updates multi-turn state, extracts entities,
    predicts classification, evaluates completeness, and returns conversational response.
    """
    history = get_session_history(req.session_id)
    save_conversation_turn(req.session_id, "user", req.message)
    
    result = analyze_grievance_text(
        text=req.message,
        accumulated_data=req.accumulated_data,
        session_history=history
    )
    
    save_conversation_turn(
        req.session_id, 
        "assistant", 
        result["assistant_reply"],
        metadata=f"Stage:{result['workflow_stage']}"
    )
    
    return JSONResponse(result)

@app.post("/api/classify")
async def api_classify(req: CompletenessRequest):
    """Direct ML classification using CivicDex models."""
    preds = classifier.predict(req.text)
    return JSONResponse(preds)

@app.post("/api/completeness/audit")
async def api_completeness_audit(req: CompletenessRequest):
    """Standalone completeness audit for a grievance text."""
    extracted = extract_grievance_entities(req.text)
    comp = evaluate_completeness(extracted)
    return JSONResponse(comp)

@app.post("/api/rag/query")
async def api_rag_query(req: RagQueryRequest):
    """RAG-backed FAQ search over civic procedures and guidelines."""
    answer_data = answer_civic_question(req.query)
    return JSONResponse(answer_data)

@app.post("/api/grievance/generate")
async def api_generate_draft(req: GenerateRequest):
    """Generate or regenerate structured grievance document."""
    gid = req.grievance_id or generate_grievance_id()
    draft = generate_structured_grievance(req.extracted_data, req.classification, gid)
    return JSONResponse(draft)

@app.post("/api/grievance/confirm")
async def api_confirm_grievance(req: ConfirmRequest):
    """Persist confirmed prototype grievance to SQLite."""
    gid = save_grievance(
        original_text=req.description,
        normalized_text=req.description,
        category=req.category,
        department=req.department,
        urgency=req.urgency,
        severity=req.severity or "service_issue",
        location=req.location,
        district="",
        duration=req.duration or "",
        summary=req.subject,
        generated_grievance=req.description,
        status="CONFIRMED",
        grievance_id=req.grievance_id
    )
    return JSONResponse({"status": "CONFIRMED", "grievance_id": gid})

@app.get("/api/grievances")
async def api_list_grievances(
    status: Optional[str] = None,
    department: Optional[str] = None,
    search: Optional[str] = None,
    limit: int = 50,
    offset: int = 0
):
    """Fetch stored grievance records with filtering."""
    records = list_grievances(status=status, department=department, search=search, limit=limit, offset=offset)
    return JSONResponse(records)

@app.get("/api/grievances/{grievance_id}")
async def api_get_grievance(grievance_id: str):
    """Fetch single grievance record by ID."""
    rec = get_grievance(grievance_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Grievance not found")
    return JSONResponse(rec)

@app.get("/api/analytics")
async def api_analytics():
    """Retrieve aggregated stats for Chart.js dashboard."""
    stats = get_analytics_summary()
    return JSONResponse(stats)

@app.get("/api/health")
async def api_health():
    """System health check for Ollama, ML models, RAG index, and SQLite database."""
    ollama_info = ollama_client.check_connection()
    return JSONResponse({
        "status": "healthy",
        "models_ready": classifier.is_ready,
        "rag_ready": RAG_INDEX_PATH.exists(),
        "database_connected": DATABASE_PATH.exists(),
        "ollama": ollama_info
    })

# ─── Admin Portal Endpoints (PIN-protected) ────────────────────────────────────

@app.get("/api/admin/stats", dependencies=[Depends(verify_admin)])
async def api_admin_stats():
    """Admin dashboard KPI stats — requires X-Admin-PIN header."""
    stats = get_admin_dashboard_stats()
    return JSONResponse(stats)

@app.get("/api/admin/grievances", dependencies=[Depends(verify_admin)])
async def api_admin_list_grievances(
    status: Optional[str] = None,
    department: Optional[str] = None,
    search: Optional[str] = None,
    limit: int = 100,
    offset: int = 0
):
    """Admin: fetch all grievances with full details — requires X-Admin-PIN header."""
    records = list_grievances(status=status, department=department, search=search, limit=limit, offset=offset)
    return JSONResponse(records)

@app.patch("/api/admin/grievances/{grievance_id}", dependencies=[Depends(verify_admin)])
async def api_admin_update_grievance(grievance_id: str, req: AdminUpdateRequest):
    """Admin: update grievance status and/or remarks — requires X-Admin-PIN header."""
    if req.status not in VALID_STATUSES:
        raise HTTPException(status_code=400, detail=f"Invalid status. Must be one of: {', '.join(VALID_STATUSES)}")
    success = update_grievance_by_admin(grievance_id, req.status, req.admin_remarks or "")
    if not success:
        raise HTTPException(status_code=404, detail="Grievance not found")
    return JSONResponse({"updated": True, "grievance_id": grievance_id, "new_status": req.status})

# ─── Citizen Complaint Tracker (Public) ────────────────────────────────────────

@app.get("/api/track/{grievance_id}")
async def api_track_grievance(grievance_id: str):
    """
    Public endpoint for citizens to track their complaint by GRV ID.
    Returns safe public fields only — no internal admin data beyond official remarks.
    """
    rec = get_grievance_for_tracker(grievance_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Grievance ID not found. Please check and try again.")
    return JSONResponse(rec)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", PORT))
    uvicorn.run("app:app", host=HOST, port=port, reload=DEBUG)

