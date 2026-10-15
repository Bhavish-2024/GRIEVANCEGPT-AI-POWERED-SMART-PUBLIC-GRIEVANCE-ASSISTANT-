"""
Comprehensive Audit & Remediation Test Suite (TEST 1 through TEST 15).
Verifies database integrity, KPI calculations, draft isolation, resolution metrics,
idempotency, transaction safety, API schemas, and verified ML evaluation.
"""
import os
import sqlite3
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

import src.config as config_module
from src.config import ADMIN_PIN
import src.database as db_module
from src.database import (
    db_session,
    init_db,
    save_grievance,
    get_grievance,
    update_grievance_status,
    update_grievance_by_admin,
    list_grievances,
    get_analytics_summary,
    seed_initial_grievances,
    generate_grievance_id,
    VALID_STATUSES
)
from app import app
from src.nlp.evaluation import (
    run_full_evaluation,
    get_verified_ml_metrics,
    check_train_test_leakage
)

@pytest.fixture
def temp_db(monkeypatch, tmp_path):
    """Provides an isolated SQLite database for each test case."""
    test_db_path = tmp_path / "test_isolated.db"
    monkeypatch.setattr(db_module, "DATABASE_PATH", test_db_path)
    monkeypatch.setattr(config_module, "DATABASE_PATH", test_db_path)
    init_db(seed=False)
    yield test_db_path
    if test_db_path.exists():
        try:
            test_db_path.unlink()
        except Exception:
            pass

# ─── TEST 1: Empty Database Metrics ───
def test_1_empty_database_metrics(temp_db):
    """Empty database must return zero counts, empty lists, and N/A for averages without division by zero."""
    summary = get_analytics_summary()
    assert summary["total_registered"] == 0
    assert summary["total_submitted"] == 0
    assert summary["confirmed_grievances"] == 0
    assert summary["draft_grievances"] == 0
    assert summary["pending_action"] == 0
    assert summary["in_progress"] == 0
    assert summary["resolved_grievances"] == 0
    assert summary["high_priority_count"] == 0
    assert summary["critical_severity_count"] == 0
    assert summary["departments"] == []
    assert summary["categories"] == []
    assert summary["urgencies"] == []
    assert summary["daily_submissions"] == []
    
    res = summary["resolution_metrics"]
    assert res["avg_hours"] is None
    assert res["avg_formatted"] == "N/A"
    assert res["resolved_count_with_duration"] == 0
    assert res["total_resolved_count"] == 0

# ─── TEST 2: Three Confirmed Grievances ───
def test_2_three_confirmed_grievances(temp_db):
    """Three confirmed grievances should produce total=3, confirmed=3, draft=0, pending=3, resolved=0."""
    depts = ["Water Supply Department", "Electricity Board (TANGEDCO)", "Water Supply Department"]
    for i, dept in enumerate(depts, 1):
        save_grievance(
            original_text=f"Complaint number {i}",
            normalized_text=f"Complaint number {i}",
            category="water_supply" if "Water" in dept else "power_cut",
            department=dept,
            urgency="high" if i == 1 else "medium",
            severity="service_issue",
            location="Zone 1",
            district="Chennai",
            duration="2 days",
            summary=f"Summary {i}",
            generated_grievance=f"Body {i}",
            status="CONFIRMED"
        )

    summary = get_analytics_summary()
    assert summary["total_registered"] == 3
    assert summary["total_submitted"] == 3
    assert summary["confirmed_grievances"] == 3
    assert summary["draft_grievances"] == 0
    assert summary["pending_action"] == 3
    assert summary["resolved_grievances"] == 0
    assert summary["high_priority_count"] == 1

    # Check department aggregation
    dept_counts = {d["label"]: d["count"] for d in summary["departments"]}
    assert dept_counts["Water Supply Department"] == 2
    assert dept_counts["Electricity Board (TANGEDCO)"] == 1
    assert sum(dept_counts.values()) == 3

# ─── TEST 3: Incomplete Draft Isolation ───
def test_3_draft_isolation(temp_db):
    """Draft grievances must never be counted as confirmed, submitted, pending, or resolved."""
    # Insert 2 confirmed grievances
    for i in range(2):
        save_grievance(
            original_text=f"Confirmed complaint {i}",
            normalized_text=f"Confirmed complaint {i}",
            category="water_supply",
            department="Water Supply Department",
            urgency="medium",
            severity="service_issue",
            location="Anna Nagar",
            district="Chennai",
            duration="1 day",
            summary=f"Summary {i}",
            generated_grievance=f"Grievance {i}",
            status="CONFIRMED"
        )
    # Insert 3 incomplete draft grievances
    for i in range(3):
        save_grievance(
            original_text=f"Incomplete draft {i}",
            normalized_text=f"Incomplete draft {i}",
            category="civic_maintenance",
            department="Municipal Administration",
            urgency="low",
            severity="general_query",
            location="",
            district="",
            duration="",
            summary=f"Draft summary {i}",
            generated_grievance="",
            status="DRAFT"
        )

    summary = get_analytics_summary()
    assert summary["total_registered"] == 5
    assert summary["draft_grievances"] == 3
    assert summary["total_submitted"] == 2
    assert summary["confirmed_grievances"] == 2
    assert summary["pending_action"] == 2
    assert summary["resolved_grievances"] == 0

# ─── TEST 4: Status Distribution Reconciliation ───
def test_4_status_distribution_reconciliation(temp_db):
    """Sum of all mutually exclusive status categories must equal total registered grievances."""
    statuses = ["CONFIRMED", "UNDER_REVIEW", "IN_PROGRESS", "FORWARDED", "RESOLVED", "DRAFT"]
    for idx, st in enumerate(statuses, 1):
        gid = f"GRV-2026-TESTST0{idx}"
        save_grievance(
            original_text=f"Complaint with status {st}",
            normalized_text=f"Complaint with status {st}",
            category="road_repair",
            department="Roads & Infrastructure Department",
            urgency="medium",
            severity="service_issue",
            location="Main Road",
            district="Madurai",
            duration="3 days",
            summary=f"Summary {st}",
            generated_grievance="",
            status=st,
            grievance_id=gid
        )

    summary = get_analytics_summary()
    assert summary["total_registered"] == len(statuses)
    by_st = summary["by_status"]
    assert sum(by_st.values()) == summary["total_registered"]
    assert summary["pending_action"] == 2  # CONFIRMED (1) + UNDER_REVIEW (1)
    assert summary["in_progress"] == 2     # IN_PROGRESS (1) + FORWARDED (1)
    assert summary["resolved_grievances"] == 1
    assert summary["draft_grievances"] == 1

# ─── TEST 5: Average Resolution Time Calculation ───
def test_5_resolution_time_calculation(temp_db):
    """Resolution time calculated strictly from valid records where resolved_at > created_at."""
    now = datetime(2026, 3, 10, 10, 0, 0)

    # Valid Case 1: 4 hours
    c1 = (now - timedelta(hours=4)).isoformat()
    r1 = now.isoformat()

    # Valid Case 2: 24 hours (1 day)
    c2 = (now - timedelta(hours=24)).isoformat()
    r2 = now.isoformat()

    with db_session() as conn:
        cursor = conn.cursor()
        # Insert Valid Case 1
        cursor.execute("""
            INSERT INTO grievances (
                grievance_id, created_at, updated_at, resolved_at, original_text, normalized_text,
                category, department, urgency, severity, location, district, duration,
                summary, generated_grievance, status
            ) VALUES ('GRV-RES-01', ?, ?, ?, 'text', 'norm', 'water_supply', 'Water Supply Department', 'medium', 'service_issue', 'loc', 'dist', 'dur', 'sum', 'gen', 'RESOLVED')
        """, (c1, r1, r1))

        # Insert Valid Case 2
        cursor.execute("""
            INSERT INTO grievances (
                grievance_id, created_at, updated_at, resolved_at, original_text, normalized_text,
                category, department, urgency, severity, location, district, duration,
                summary, generated_grievance, status
            ) VALUES ('GRV-RES-02', ?, ?, ?, 'text', 'norm', 'power_cut', 'Electricity Board (TANGEDCO)', 'medium', 'service_issue', 'loc', 'dist', 'dur', 'sum', 'gen', 'RESOLVED')
        """, (c2, r2, r2))

        # Insert Corrupt Case 3: resolved_at < created_at (must be excluded)
        cursor.execute("""
            INSERT INTO grievances (
                grievance_id, created_at, updated_at, resolved_at, original_text, normalized_text,
                category, department, urgency, severity, location, district, duration,
                summary, generated_grievance, status
            ) VALUES ('GRV-RES-03', ?, ?, ?, 'text', 'norm', 'power_cut', 'Electricity Board (TANGEDCO)', 'medium', 'service_issue', 'loc', 'dist', 'dur', 'sum', 'gen', 'RESOLVED')
        """, (r2, c2, c2))

        # Insert Case 4: Resolved but NULL resolved_at (must be excluded)
        cursor.execute("""
            INSERT INTO grievances (
                grievance_id, created_at, updated_at, resolved_at, original_text, normalized_text,
                category, department, urgency, severity, location, district, duration,
                summary, generated_grievance, status
            ) VALUES ('GRV-RES-04', ?, ?, NULL, 'text', 'norm', 'power_cut', 'Electricity Board (TANGEDCO)', 'medium', 'service_issue', 'loc', 'dist', 'dur', 'sum', 'gen', 'RESOLVED')
        """, (c1, c1))

    summary = get_analytics_summary()
    res = summary["resolution_metrics"]
    assert res["total_resolved_count"] == 4
    assert res["resolved_count_with_duration"] == 2
    # Expected average: (4 + 24) / 2 = 14.0 hours
    assert res["avg_hours"] == 14.0
    assert res["avg_formatted"] == "14.0 hrs"
    assert res["fastest_hours"] == 4.0
    assert res["slowest_hours"] == 24.0

# ─── TEST 6: Date Range Filter ───
def test_6_date_range_filter(temp_db):
    """Date filter must isolate grievances created within the start_date and end_date window."""
    dates = ["2026-01-10T10:00:00", "2026-02-15T12:00:00", "2026-03-20T14:00:00"]
    with db_session() as conn:
        cursor = conn.cursor()
        for idx, dt in enumerate(dates, 1):
            cursor.execute("""
                INSERT INTO grievances (
                    grievance_id, created_at, updated_at, original_text, normalized_text,
                    category, department, urgency, severity, location, district, duration,
                    summary, generated_grievance, status
                ) VALUES (?, ?, ?, 'text', 'norm', 'water_supply', 'Water Supply Department', 'medium', 'service_issue', 'loc', 'dist', 'dur', 'sum', 'gen', 'CONFIRMED')
            """, (f"GRV-DATE-0{idx}", dt, dt))

    # All time
    all_summary = get_analytics_summary()
    assert all_summary["total_registered"] == 3

    # Filter for February only
    feb_summary = get_analytics_summary(start_date="2026-02-01", end_date="2026-02-28")
    assert feb_summary["total_registered"] == 1
    assert feb_summary["confirmed_grievances"] == 1

# ─── TEST 7: Department Filter ───
def test_7_department_filter(temp_db):
    """Department filter must restrict all counts and aggregations to the target department."""
    save_grievance(
        original_text="Water complaint 1", normalized_text="Water 1", category="water_supply",
        department="Water Supply Department", urgency="high", severity="service_issue",
        location="Loc 1", district="Chennai", duration="1d", summary="s1", generated_grievance="", status="CONFIRMED"
    )
    save_grievance(
        original_text="Water complaint 2", normalized_text="Water 2", category="water_supply",
        department="Water Supply Department", urgency="low", severity="service_issue",
        location="Loc 2", district="Chennai", duration="2d", summary="s2", generated_grievance="", status="RESOLVED"
    )
    save_grievance(
        original_text="Electric complaint 1", normalized_text="Electric 1", category="power_cut",
        department="Electricity Board (TANGEDCO)", urgency="high", severity="service_issue",
        location="Loc 3", district="Chennai", duration="3d", summary="s3", generated_grievance="", status="CONFIRMED"
    )

    water_summary = get_analytics_summary(department="Water Supply Department")
    assert water_summary["total_registered"] == 2
    assert len(water_summary["departments"]) == 1
    assert water_summary["departments"][0]["label"] == "Water Supply Department"
    assert water_summary["departments"][0]["count"] == 2

    elec_summary = get_analytics_summary(department="Electricity Board (TANGEDCO)")
    assert elec_summary["total_registered"] == 1
    assert elec_summary["departments"][0]["label"] == "Electricity Board (TANGEDCO)"

# ─── TEST 8: Prevent Duplicate Grievance Records ───
def test_8_duplicate_prevention(temp_db):
    """Re-saving with an existing grievance_id must update in-place without duplicating records."""
    gid = "GRV-2026-UNIQUE-TEST"
    save_grievance(
        original_text="Original text", normalized_text="Original norm", category="water_supply",
        department="Water Supply Department", urgency="medium", severity="service_issue",
        location="Loc", district="Chennai", duration="1d", summary="Initial Summary",
        generated_grievance="Initial body", status="DRAFT", grievance_id=gid
    )
    assert get_analytics_summary()["total_registered"] == 1

    # Overwrite / update grievance with same ID
    save_grievance(
        original_text="Updated text", normalized_text="Updated norm", category="water_supply",
        department="Water Supply Department", urgency="high", severity="service_issue",
        location="Loc Updated", district="Chennai", duration="2d", summary="Updated Summary",
        generated_grievance="Updated body", status="CONFIRMED", grievance_id=gid
    )

    summary = get_analytics_summary()
    assert summary["total_registered"] == 1
    record = get_grievance(gid)
    assert record["status"] == "CONFIRMED"
    assert record["urgency"] == "high"
    assert record["summary"] == "Updated Summary"

# ─── TEST 9: Persistent Transaction Safety & Rollback ───
def test_9_transaction_safety_and_rollback(temp_db):
    """Operations raising exceptions inside db_session must roll back without corrupting database."""
    # Ensure clean starting state
    assert get_analytics_summary()["total_registered"] == 0

    # Attempt faulty transaction
    with pytest.raises(RuntimeError):
        with db_session() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO grievances (
                    grievance_id, created_at, updated_at, original_text, normalized_text,
                    category, department, urgency, severity, location, district, duration,
                    summary, generated_grievance, status
                ) VALUES ('GRV-FAIL-01', '2026-03-01', '2026-03-01', 'text', 'norm', 'cat', 'dept', 'med', 'sev', 'loc', 'dist', 'dur', 'sum', 'gen', 'CONFIRMED')
            """)
            raise RuntimeError("Simulated transaction failure")

    # Confirm rollback
    assert get_analytics_summary()["total_registered"] == 0
    assert get_grievance("GRV-FAIL-01") is None

# ─── TEST 10: Status Lifecycle & Resolved Timestamp Handling ───
def test_10_status_lifecycle_and_resolved_timestamp(temp_db):
    """Transitioning to RESOLVED sets resolved_at; moving away from RESOLVED clears resolved_at."""
    gid = "GRV-2026-LIFECYCLE"
    save_grievance(
        original_text="Streetlight broken", normalized_text="Streetlight broken", category="street_lighting",
        department="Civic Maintenance & Street Lighting", urgency="medium", severity="service_issue",
        location="Cross St", district="Salem", duration="5d", summary="Broken light",
        generated_grievance="", status="CONFIRMED", grievance_id=gid
    )

    rec = get_grievance(gid)
    assert rec["status"] == "CONFIRMED"
    assert rec["resolved_at"] is None

    # Transition to RESOLVED
    assert update_grievance_status(gid, "RESOLVED") is True
    rec_resolved = get_grievance(gid)
    assert rec_resolved["status"] == "RESOLVED"
    assert rec_resolved["resolved_at"] is not None

    # Admin reopens case to IN_PROGRESS -> resolved_at must be cleared
    assert update_grievance_status(gid, "IN_PROGRESS") is True
    rec_reopened = get_grievance(gid)
    assert rec_reopened["status"] == "IN_PROGRESS"
    assert rec_reopened["resolved_at"] is None

# ─── TEST 11: Seed Idempotency Without Data Loss ───
def test_11_seed_idempotency_no_deletion(temp_db):
    """seed_initial_grievances must not delete user complaints or re-seed when records exist."""
    user_gid = "GRV-2026-000999"
    save_grievance(
        original_text="Citizen filed complaint", normalized_text="Citizen filed complaint", category="water_supply",
        department="Water Supply Department", urgency="high", severity="service_issue",
        location="Zone 9", district="Coimbatore", duration="3d", summary="User complaint",
        generated_grievance="", status="CONFIRMED", grievance_id=user_gid
    )

    # Calling seed without force must do nothing
    seeded_count = seed_initial_grievances(force=False)
    assert seeded_count == 0

    # User complaint must still exist
    user_rec = get_grievance(user_gid)
    assert user_rec is not None
    assert user_rec["grievance_id"] == user_gid

# ─── TEST 12: API Endpoints Schema & Authentication ───
def test_12_api_endpoints_schema_and_auth(temp_db):
    """Verify HTTP status codes, schema validation, and admin authentication."""
    client = TestClient(app)

    # 1. /api/analytics
    res_analytics = client.get("/api/analytics")
    assert res_analytics.status_code == 200
    adata = res_analytics.json()
    assert "total_registered" in adata
    assert "total_submitted" in adata
    assert "pending_action" in adata
    assert "in_progress" in adata
    assert "resolved_grievances" in adata
    assert "resolution_metrics" in adata
    assert "ml_metrics" in adata

    # 2. /api/ml/metrics
    res_ml = client.get("/api/ml/metrics")
    assert res_ml.status_code == 200
    mdata = res_ml.json()
    assert mdata["is_verified"] is True
    assert "tasks" in mdata
    for task_name in ["intent", "department", "category", "urgency"]:
        assert task_name in mdata["tasks"]
        t = mdata["tasks"][task_name]
        assert "accuracy" in t
        assert "weighted_f1" in t

    # 3. Admin Authentication check
    res_unauth = client.get("/api/admin/stats")
    assert res_unauth.status_code == 401

    res_auth = client.get("/api/admin/stats", headers={"X-Admin-PIN": ADMIN_PIN})
    assert res_auth.status_code == 200
    sdata = res_auth.json()
    assert "total" in sdata
    assert "confirmed" in sdata
    assert "resolved" in sdata

# ─── TEST 13: Admin Stats Reconciliation ───
def test_13_admin_stats_reconciliation(temp_db):
    """Admin dashboard stats must reconcile exactly with public analytics calculations."""
    save_grievance(
        original_text="C1", normalized_text="C1", category="water_supply",
        department="Water Supply Department", urgency="high", severity="service_issue",
        location="L1", district="D1", duration="1d", summary="S1", generated_grievance="", status="CONFIRMED"
    )
    save_grievance(
        original_text="C2", normalized_text="C2", category="power_cut",
        department="Electricity Board (TANGEDCO)", urgency="medium", severity="service_issue",
        location="L2", district="D2", duration="2d", summary="S2", generated_grievance="", status="UNDER_REVIEW"
    )
    save_grievance(
        original_text="C3", normalized_text="C3", category="road_repair",
        department="Roads & Infrastructure Department", urgency="low", severity="service_issue",
        location="L3", district="D3", duration="3d", summary="S3", generated_grievance="", status="RESOLVED"
    )

    client = TestClient(app)
    admin_res = client.get("/api/admin/stats", headers={"X-Admin-PIN": ADMIN_PIN}).json()
    analytics_res = client.get("/api/analytics").json()

    assert admin_res["total"] == analytics_res["total_registered"]
    assert admin_res["resolved"] == analytics_res["resolved_grievances"]
    assert admin_res["under_review"] == analytics_res["by_status"]["UNDER_REVIEW"]
    assert admin_res["confirmed"] == analytics_res["by_status"]["CONFIRMED"]

# ─── TEST 14: ML Evaluation Verification & Data Leakage ───
def test_14_ml_evaluation_metrics_and_leakage():
    """Verify reproducible evaluation on held-out test.csv with no train-test data leakage."""
    leakage = check_train_test_leakage()
    assert leakage["duplicate_text_count"] == 0, f"Found {leakage['duplicate_text_count']} leaked texts between train and test sets!"
    assert leakage["overlap_percentage"] == 0.0

    metrics = get_verified_ml_metrics()
    assert metrics["is_verified"] is True
    assert metrics.get("test_sample_count") == 60 or metrics.get("dataset_size") == 60

    tasks = metrics["tasks"]
    # Check Intent: Accuracy ~96.67%, Weighted F1 ~97.75%
    assert abs(tasks["intent"]["accuracy"] - 96.67) < 0.5
    assert abs(tasks["intent"]["weighted_f1"] - 97.75) < 0.5

    # Check Department: Accuracy ~83.33%, Weighted F1 ~81.01%
    assert abs(tasks["department"]["accuracy"] - 83.33) < 0.5
    assert abs(tasks["department"]["weighted_f1"] - 81.01) < 0.5

    # Check Category: Accuracy ~76.67%, Weighted F1 ~74.59%
    assert abs(tasks["category"]["accuracy"] - 76.67) < 0.5
    assert abs(tasks["category"]["weighted_f1"] - 74.59) < 0.5

    # Check Urgency: Accuracy ~58.33%, Weighted F1 ~59.22%
    assert abs(tasks["urgency"]["accuracy"] - 58.33) < 0.5
    assert abs(tasks["urgency"]["weighted_f1"] - 59.22) < 0.5

    # Ensure accuracy and weighted F1 are strictly distinguished
    for t_name, t_data in tasks.items():
        assert "accuracy" in t_data
        assert "weighted_f1" in t_data

# ─── TEST 15: Full End-to-End Complaint Lifecycle ───
def test_15_full_e2e_complaint_lifecycle(temp_db):
    """Complete lifecycle: citizen intake draft -> confirm -> admin review -> admin resolve -> tracker check."""
    client = TestClient(app)

    # 1. Draft generated
    draft_id = "GRV-2026-E2E-TEST"
    save_grievance(
        original_text="Heavy water leak on Anna Salai near bus stop",
        normalized_text="Heavy water leak on Anna Salai near bus stop",
        category="water_supply",
        department="Water Supply Department",
        urgency="high",
        severity="safety_hazard",
        location="Anna Salai, Chennai",
        district="Chennai",
        duration="3 days",
        summary="Water main burst causing road flooding",
        generated_grievance="Formal complaint for urgent repair",
        status="DRAFT",
        grievance_id=draft_id
    )

    s1 = client.get("/api/analytics").json()
    assert s1["draft_grievances"] == 1
    assert s1["total_submitted"] == 0

    # 2. Citizen confirms grievance
    res_confirm = client.post("/api/grievance/confirm", json={
        "grievance_id": draft_id,
        "subject": "Urgent Water Main Burst",
        "description": "Flooding road on Anna Salai",
        "location": "Anna Salai, Chennai",
        "duration": "3 days",
        "category": "water_supply",
        "department": "Water Supply Department",
        "urgency": "high",
        "severity": "safety_hazard"
    })
    assert res_confirm.status_code == 200

    s2 = client.get("/api/analytics").json()
    assert s2["draft_grievances"] == 0
    assert s2["total_submitted"] == 1
    assert s2["confirmed_grievances"] == 1
    assert s2["pending_action"] == 1

    # 3. Citizen tracks grievance
    res_track1 = client.get(f"/api/track/{draft_id}")
    assert res_track1.status_code == 200
    assert res_track1.json()["status"] == "CONFIRMED"

    # 4. Admin updates to IN_PROGRESS
    res_admin1 = client.patch(
        f"/api/admin/grievances/{draft_id}",
        headers={"X-Admin-PIN": ADMIN_PIN},
        json={"status": "IN_PROGRESS", "admin_remarks": "Maintenance team dispatched to site."}
    )
    assert res_admin1.status_code == 200

    s3 = client.get("/api/analytics").json()
    assert s3["pending_action"] == 0
    assert s3["in_progress"] == 1

    # 5. Admin marks as RESOLVED
    res_admin2 = client.patch(
        f"/api/admin/grievances/{draft_id}",
        headers={"X-Admin-PIN": ADMIN_PIN},
        json={"status": "RESOLVED", "admin_remarks": "Pipeline patched and water flow normalized."}
    )
    assert res_admin2.status_code == 200

    s4 = client.get("/api/analytics").json()
    assert s4["in_progress"] == 0
    assert s4["resolved_grievances"] == 1

    # 6. Citizen tracker verifies resolution and remarks
    res_track2 = client.get(f"/api/track/{draft_id}")
    assert res_track2.status_code == 200
    tdata = res_track2.json()
    assert tdata["status"] == "RESOLVED"
    assert "Pipeline patched" in tdata["admin_remarks"]


# ─── TEST 16: Public Navigation & Registry Access Restriction ───
def test_16_public_navigation_and_registry_access_restriction(temp_db):
    """
    Verifies that public callers cannot access sensitive complaint records via /api/grievances,
    while authorized staff with X-Admin-PIN can access them, and public citizens can still
    track individual complaints via /api/track/{id}.
    """
    client = TestClient(app)
    
    # Create a confirmed complaint
    gid = save_grievance(
        original_text="Broken transformer emitting sparks near market",
        normalized_text="Broken transformer emitting sparks near market",
        category="power_cut",
        department="Electricity Board (TANGEDCO)",
        urgency="high",
        severity="safety_hazard",
        location="Gandhi Market, Trichy",
        district="Tiruchirappalli",
        duration="1 day",
        summary="Sparking transformer",
        generated_grievance="Urgent fix required",
        status="CONFIRMED"
    )

    # 1. Unauthenticated request to /api/grievances MUST return 401 Unauthorized
    res_public_list = client.get("/api/grievances")
    assert res_public_list.status_code == 401
    assert "Unauthorized" in res_public_list.json()["detail"]

    # 2. Unauthenticated request to /api/grievances/{id} MUST return 401 Unauthorized
    res_public_get = client.get(f"/api/grievances/{gid}")
    assert res_public_get.status_code == 401
    assert "Unauthorized" in res_public_get.json()["detail"]

    # 3. Request with wrong PIN MUST return 401 Unauthorized
    res_wrong_pin = client.get("/api/grievances", headers={"X-Admin-PIN": "WRONG_PIN"})
    assert res_wrong_pin.status_code == 401

    # 4. Request with valid Admin PIN MUST return 200 OK
    res_admin_list = client.get("/api/grievances", headers={"X-Admin-PIN": ADMIN_PIN})
    assert res_admin_list.status_code == 200
    records = res_admin_list.json()
    assert len(records) >= 1
    assert records[0]["grievance_id"] == gid

    # 5. Single item with valid Admin PIN MUST return 200 OK
    res_admin_get = client.get(f"/api/grievances/{gid}", headers={"X-Admin-PIN": ADMIN_PIN})
    assert res_admin_get.status_code == 200
    assert res_admin_get.json()["grievance_id"] == gid

    # 6. Public tracking via /api/track/{id} MUST remain open without PIN
    res_track = client.get(f"/api/track/{gid}")
    assert res_track.status_code == 200
    track_data = res_track.json()
    assert track_data["grievance_id"] == gid
    assert track_data["status"] == "CONFIRMED"
    assert track_data["department"] == "Electricity Board (TANGEDCO)"
    assert track_data["location"] == "Gandhi Market, Trichy"

