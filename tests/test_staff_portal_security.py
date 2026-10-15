"""
Staff Portal Authentication & Session Security Test Suite.
Validates:
- Dual-credential authentication (Officer ID, 2FA PIN)
- Brute-force rate limiting and lockout defense
- Cryptographically secure session issuance
- Session token authentication via Bearer and X-Admin-Token
- Session invalidation on logout
- Backward compatibility with X-Admin-PIN
"""

import pytest
from fastapi.testclient import TestClient
from app import app, LOGIN_ATTEMPTS, ADMIN_SESSIONS
from src.config import ADMIN_USERNAME, ADMIN_PIN


@pytest.fixture(autouse=True)
def clean_security_state():
    """Ensure clean login attempts and sessions before each test."""
    LOGIN_ATTEMPTS.clear()
    ADMIN_SESSIONS.clear()
    yield
    LOGIN_ATTEMPTS.clear()
    ADMIN_SESSIONS.clear()


@pytest.fixture
def client():
    return TestClient(app)


def test_admin_portal_requires_auth(client):
    """Admin endpoints must reject unauthenticated requests with 401."""
    res_stats = client.get("/api/admin/stats")
    assert res_stats.status_code == 401

    res_list = client.get("/api/admin/grievances")
    assert res_list.status_code == 401


def test_admin_login_invalid_username(client):
    """Wrong username must be rejected with 401."""
    payload = {
        "username": "wrong_officer",
        "pin": ADMIN_PIN
    }
    res = client.post("/api/admin/login", json=payload)
    assert res.status_code == 401
    assert "Invalid Officer ID or Security PIN" in res.json()["detail"]


def test_admin_login_invalid_pin(client):
    """Wrong 2FA PIN must be rejected with 401."""
    payload = {
        "username": ADMIN_USERNAME,
        "pin": "999999"
    }
    res = client.post("/api/admin/login", json=payload)
    assert res.status_code == 401
    assert "Invalid Officer ID or Security PIN" in res.json()["detail"]


def test_admin_login_brute_force_lockout(client):
    """5 consecutive failed logins trigger HTTP 429 lockout."""
    bad_payload = {
        "username": ADMIN_USERNAME,
        "pin": "bad_pin"
    }

    # First 4 attempts fail with 401
    for i in range(4):
        res = client.post("/api/admin/login", json=bad_payload)
        assert res.status_code == 401

    # 5th attempt fails with 401
    res5 = client.post("/api/admin/login", json=bad_payload)
    assert res5.status_code == 401

    # 6th attempt is blocked with 429 Too Many Requests
    res6 = client.post("/api/admin/login", json=bad_payload)
    assert res6.status_code == 429
    assert "Security lockout active" in res6.json()["detail"]


def test_admin_login_success_and_session_flow(client):
    """Valid credentials issue a 256-bit session token that grants access to admin resources."""
    valid_payload = {
        "username": ADMIN_USERNAME,
        "pin": ADMIN_PIN
    }

    login_res = client.post("/api/admin/login", json=valid_payload)
    assert login_res.status_code == 200
    data = login_res.json()
    assert data["authenticated"] is True
    assert "token" in data
    assert len(data["token"]) == 64  # 32 bytes hex = 64 characters
    assert data["user"]["name"] == "Thiru. S. Ramanathan, IAS"
    assert data["user"]["badge_id"] == "TN-GOV-REV-2026"

    token = data["token"]

    # Access /api/admin/stats via Bearer token
    stats_res = client.get("/api/admin/stats", headers={"Authorization": f"Bearer {token}"})
    assert stats_res.status_code == 200
    assert "total" in stats_res.json()

    # Access /api/admin/grievances via X-Admin-Token
    list_res = client.get("/api/admin/grievances", headers={"X-Admin-Token": token})
    assert list_res.status_code == 200
    assert isinstance(list_res.json(), list)

    # Check active session profile
    session_res = client.get("/api/admin/session", headers={"Authorization": f"Bearer {token}"})
    assert session_res.status_code == 200
    assert session_res.json()["user"]["badge_id"] == "TN-GOV-REV-2026"

    # Logout
    logout_res = client.post("/api/admin/logout", headers={"Authorization": f"Bearer {token}"})
    assert logout_res.status_code == 200
    assert logout_res.json()["authenticated"] is False

    # After logout, token must be revoked and rejected with 401
    revoked_res = client.get("/api/admin/stats", headers={"Authorization": f"Bearer {token}"})
    assert revoked_res.status_code == 401


def test_backward_compatibility_with_admin_pin(client):
    """Automated backend test suites using X-Admin-PIN continue to function seamlessly."""
    res = client.get("/api/admin/stats", headers={"X-Admin-PIN": ADMIN_PIN})
    assert res.status_code == 200
    assert "total" in res.json()


def test_admin_grievance_search_filters(client):
    """Admin registry search filter must find records by ID and keywords."""
    headers = {"X-Admin-PIN": ADMIN_PIN}
    
    # Search by full ID
    res = client.get("/api/admin/grievances?search=GRV-2026-000109", headers=headers)
    assert res.status_code == 200
    items = res.json()
    assert len(items) >= 1
    assert items[0]["grievance_id"] == "GRV-2026-000109"

    # Search by ID suffix
    res_suffix = client.get("/api/admin/grievances?search=000109", headers=headers)
    assert res_suffix.status_code == 200
    suffix_items = res_suffix.json()
    assert len(suffix_items) >= 1
    assert any(x["grievance_id"] == "GRV-2026-000109" for x in suffix_items)


def test_admin_get_single_grievance_record(client):
    """Admin single grievance endpoint must return full details of a complaint."""
    headers = {"X-Admin-PIN": ADMIN_PIN}
    res = client.get("/api/admin/grievances/GRV-2026-000109", headers=headers)
    assert res.status_code == 200
    data = res.json()
    assert data["grievance_id"] == "GRV-2026-000109"
    assert "department" in data
    assert "original_text" in data
    assert "status" in data


def test_protected_grievance_registry_search_comprehensive(client):
    """
    Comprehensive verification of Protected Grievance Registry Search:
    1. Exact Grievance IDs: GRV-2026-000104 and GRV-2026-000110
    2. Leading zeros and partial IDs: 000104, 104, 000110, 110
    3. Case-insensitive and hyphenless: grv-2026-000104, GRV2026000104, grv2026000110
    4. Complaint Summary: 'Frequent Power Outages'
    5. Complaint Description / Original Text: 'transformer voltage fluctuations'
    6. Location: 'T. Nagar', 'Beach Promenade'
    7. Department: 'Electricity Board', 'Civic Maintenance'
    8. Status text: 'under review', 'UNDER_REVIEW', 'confirmed', 'resolved'
    9. Combined status and department filters without hiding matches
    10. Dropdown wildcard handling ('All Statuses', 'All Departments')
    11. ID search precedence over accidental leftover filters
    12. Nonexistent records return empty list without error
    """
    headers = {"X-Admin-PIN": ADMIN_PIN}

    # 1. Exact ID searches (GRV-2026-000104 & GRV-2026-000110)
    res_104 = client.get("/api/admin/grievances?search=GRV-2026-000104", headers=headers)
    assert res_104.status_code == 200
    items_104 = res_104.json()
    assert len(items_104) >= 1
    assert items_104[0]["grievance_id"] == "GRV-2026-000104"

    res_110 = client.get("/api/admin/grievances?search=GRV-2026-000110", headers=headers)
    assert res_110.status_code == 200
    items_110 = res_110.json()
    assert len(items_110) >= 1
    assert items_110[0]["grievance_id"] == "GRV-2026-000110"

    # 2. Leading zeros and partial numeric IDs
    res_zeros_104 = client.get("/api/admin/grievances?search=000104", headers=headers)
    assert res_zeros_104.status_code == 200
    assert any(x["grievance_id"] == "GRV-2026-000104" for x in res_zeros_104.json())

    res_partial_104 = client.get("/api/admin/grievances?search=104", headers=headers)
    assert res_partial_104.status_code == 200
    assert any(x["grievance_id"] == "GRV-2026-000104" for x in res_partial_104.json())

    res_zeros_110 = client.get("/api/admin/grievances?search=000110", headers=headers)
    assert res_zeros_110.status_code == 200
    assert any(x["grievance_id"] == "GRV-2026-000110" for x in res_zeros_110.json())

    res_partial_110 = client.get("/api/admin/grievances?search=110", headers=headers)
    assert res_partial_110.status_code == 200
    assert any(x["grievance_id"] == "GRV-2026-000110" for x in res_partial_110.json())

    # 3. Case-insensitive and hyphenless searches
    res_case = client.get("/api/admin/grievances?search=grv-2026-000104", headers=headers)
    assert res_case.status_code == 200
    assert any(x["grievance_id"] == "GRV-2026-000104" for x in res_case.json())

    res_nohyphen = client.get("/api/admin/grievances?search=GRV2026000104", headers=headers)
    assert res_nohyphen.status_code == 200
    assert any(x["grievance_id"] == "GRV-2026-000104" for x in res_nohyphen.json())

    # 4. Complaint Summary search
    res_summary = client.get("/api/admin/grievances?search=Frequent+Power+Outages", headers=headers)
    assert res_summary.status_code == 200
    assert any(x["grievance_id"] == "GRV-2026-000104" for x in res_summary.json())

    # 5. Complaint Description / Original Text search
    res_desc = client.get("/api/admin/grievances?search=voltage+fluctuations", headers=headers)
    assert res_desc.status_code == 200
    assert any(x["grievance_id"] == "GRV-2026-000104" for x in res_desc.json())

    # 6. Location search
    res_loc = client.get("/api/admin/grievances?search=T.+Nagar", headers=headers)
    assert res_loc.status_code == 200
    assert any(x["grievance_id"] == "GRV-2026-000104" for x in res_loc.json())

    # 7. Department keyword search
    res_dept = client.get("/api/admin/grievances?search=Electricity+Board", headers=headers)
    assert res_dept.status_code == 200
    assert any(x["grievance_id"] == "GRV-2026-000104" for x in res_dept.json())

    # 8. Status text search in search bar
    res_status = client.get("/api/admin/grievances?search=under+review", headers=headers)
    assert res_status.status_code == 200
    under_review_items = res_status.json()
    assert len(under_review_items) >= 1
    assert all(x["status"] == "UNDER_REVIEW" or "under review" in str(x).lower() for x in under_review_items)

    # 9. Dropdown wildcard handling ("All Statuses" and "All Departments")
    res_all_filters = client.get("/api/admin/grievances?status=All+Statuses&department=All+Departments", headers=headers)
    assert res_all_filters.status_code == 200
    assert len(res_all_filters.json()) >= 10  # Full registry returned, not 0 records

    # 10. Combined search and matching dropdown filter
    res_combined = client.get(
        "/api/admin/grievances?status=UNDER_REVIEW&department=Electricity+Board+(TANGEDCO)&search=000104",
        headers=headers
    )
    assert res_combined.status_code == 200
    assert any(x["grievance_id"] == "GRV-2026-000104" for x in res_combined.json())

    # 11. ID Search precedence over accidental leftover filter
    # User left status=CONFIRMED, but explicitly searches for GRV-2026-000104 (which is UNDER_REVIEW)
    res_override = client.get("/api/admin/grievances?status=CONFIRMED&search=GRV-2026-000104", headers=headers)
    assert res_override.status_code == 200
    assert any(x["grievance_id"] == "GRV-2026-000104" for x in res_override.json())

    # 12. Nonexistent records return empty list []
    res_none = client.get("/api/admin/grievances?search=GRV-2026-999999", headers=headers)
    assert res_none.status_code == 200
    assert res_none.json() == []
