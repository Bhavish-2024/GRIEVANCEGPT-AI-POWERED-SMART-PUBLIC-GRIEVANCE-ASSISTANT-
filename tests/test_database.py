"""
Unit tests for SQLite database operations and grievance ID generation.
"""
from src.database import (
    init_db,
    save_grievance,
    get_grievance,
    update_grievance_status,
    list_grievances,
    get_analytics_summary,
    generate_grievance_id
)

def test_database_lifecycle():
    init_db()
    
    gid = generate_grievance_id()
    assert gid.startswith("GRV-")
    
    saved_id = save_grievance(
        original_text="No water supply for 4 days",
        normalized_text="No water supply for 4 days",
        category="water_supply",
        department="Water Supply Department",
        urgency="high",
        severity="service_failure",
        location="Gandhi Nagar, Katpadi",
        district="Vellore",
        duration="4 days",
        summary="Urgent drinking water shortage",
        generated_grievance="Formal complaint body...",
        status="DRAFT",
        grievance_id=gid
    )
    assert saved_id == gid
    
    # Retrieve
    rec = get_grievance(gid)
    assert rec is not None
    assert rec["grievance_id"] == gid
    assert rec["category"] == "water_supply"
    assert rec["status"] == "DRAFT"
    
    # Update status to CONFIRMED
    updated = update_grievance_status(gid, "CONFIRMED")
    assert updated is True
    
    rec_updated = get_grievance(gid)
    assert rec_updated["status"] == "CONFIRMED"
    
    # List
    records = list_grievances(status="CONFIRMED")
    assert any(r["grievance_id"] == gid for r in records)
    
    # Analytics
    stats = get_analytics_summary()
    assert stats["total_grievances"] >= 1
    assert stats["confirmed_grievances"] >= 1
