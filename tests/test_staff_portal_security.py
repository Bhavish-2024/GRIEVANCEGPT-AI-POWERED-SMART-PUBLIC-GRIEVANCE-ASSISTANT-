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
