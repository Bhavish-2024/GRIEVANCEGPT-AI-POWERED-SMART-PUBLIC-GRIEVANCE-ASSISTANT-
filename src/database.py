"""
SQLite Database interface for GrievanceGPT.
Manages persistent storage for grievances, multi-turn conversations, and user feedback.
"""
import sqlite3
from datetime import datetime
from typing import List, Dict, Any, Optional
from src.config import DATABASE_PATH

def get_db_connection() -> sqlite3.Connection:
    """Establish and return a SQLite database connection with row factory."""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db() -> None:
    """Initialize SQLite database tables if they do not exist."""
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Grievances table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS grievances (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                grievance_id TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                original_text TEXT NOT NULL,
                normalized_text TEXT,
                category TEXT NOT NULL,
                department TEXT NOT NULL,
                urgency TEXT NOT NULL,
                severity TEXT DEFAULT 'service_issue',
                location TEXT DEFAULT '',
                district TEXT DEFAULT '',
                duration TEXT DEFAULT '',
                summary TEXT,
                generated_grievance TEXT NOT NULL,
                status TEXT DEFAULT 'DRAFT',
                admin_remarks TEXT DEFAULT '',
                resolved_at TIMESTAMP
            )
        """)
        
        # Safe ALTER TABLE: add new columns if the table already existed without them
        for col, col_def in [
            ("admin_remarks", "TEXT DEFAULT ''"),
            ("resolved_at", "TIMESTAMP"),
        ]:
            try:
                cursor.execute(f"ALTER TABLE grievances ADD COLUMN {col} {col_def}")
            except Exception:
                pass  # Column already exists — safe to ignore
        
        # Conversations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
                message TEXT NOT NULL,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Feedback table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                grievance_id TEXT,
                rating INTEGER CHECK(rating BETWEEN 1 AND 5),
                comments TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (grievance_id) REFERENCES grievances (grievance_id)
            )
        """)
        
        conn.commit()

def generate_grievance_id() -> str:
    """Generate a unique sequential grievance identifier e.g., GRV-2026-000001."""
    current_year = datetime.now().year
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM grievances")
        count = cursor.fetchone()[0] + 1
        return f"GRV-{current_year}-{count:06d}"

def save_grievance(
    original_text: str,
    normalized_text: str,
    category: str,
    department: str,
    urgency: str,
    severity: str,
    location: str,
    district: str,
    duration: str,
    summary: str,
    generated_grievance: str,
    status: str = "DRAFT",
    grievance_id: Optional[str] = None
) -> str:
    """Insert or update a structured grievance record."""
    gid = grievance_id or generate_grievance_id()
    now = datetime.now().isoformat()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO grievances (
                grievance_id, created_at, updated_at, original_text, normalized_text,
                category, department, urgency, severity, location, district, duration,
                summary, generated_grievance, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(grievance_id) DO UPDATE SET
                updated_at = excluded.updated_at,
                category = excluded.category,
                department = excluded.department,
                urgency = excluded.urgency,
                severity = excluded.severity,
                location = excluded.location,
                district = excluded.district,
                duration = excluded.duration,
                summary = excluded.summary,
                generated_grievance = excluded.generated_grievance,
                status = excluded.status
        """, (
            gid, now, now, original_text, normalized_text,
            category, department, urgency, severity, location, district, duration,
            summary, generated_grievance, status
        ))
        conn.commit()
    return gid

def update_grievance_status(grievance_id: str, new_status: str) -> bool:
    """Update grievance status (e.g., CONFIRMED, RESOLVED)."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE grievances SET status = ?, updated_at = CURRENT_TIMESTAMP
            WHERE grievance_id = ?
        """, (new_status, grievance_id))
        conn.commit()
        return cursor.rowcount > 0

def get_grievance(grievance_id: str) -> Optional[Dict[str, Any]]:
    """Fetch single grievance by ID."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM grievances WHERE grievance_id = ?", (grievance_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

def list_grievances(
    status: Optional[str] = None,
    department: Optional[str] = None,
    search: Optional[str] = None,
    limit: int = 50,
    offset: int = 0
) -> List[Dict[str, Any]]:
    """Query list of grievances with optional filtering."""
    query = "SELECT * FROM grievances WHERE 1=1"
    params = []
    
    if status and status.upper() != "ALL":
        query += " AND status = ?"
        params.append(status.upper())
    if department and department.lower() != "all":
        query += " AND department = ?"
        params.append(department)
    if search:
        query += " AND (original_text LIKE ? OR location LIKE ? OR grievance_id LIKE ?)"
        term = f"%{search}%"
        params.extend([term, term, term])
        
    query += " ORDER BY id DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]

def get_analytics_summary() -> Dict[str, Any]:
    """Calculate aggregated metrics for dashboard charts."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Total counts
        cursor.execute("SELECT COUNT(*) FROM grievances")
        total = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM grievances WHERE status = 'CONFIRMED'")
        confirmed = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM grievances WHERE status = 'DRAFT'")
        drafts = cursor.fetchone()[0]
        
        # Category distribution
        cursor.execute("""
            SELECT category, COUNT(*) as count FROM grievances
            GROUP BY category ORDER BY count DESC
        """)
        categories = [{"label": row["category"], "count": row["count"]} for row in cursor.fetchall()]
        
        # Department distribution
        cursor.execute("""
            SELECT department, COUNT(*) as count FROM grievances
            GROUP BY department ORDER BY count DESC
        """)
        departments = [{"label": row["department"], "count": row["count"]} for row in cursor.fetchall()]
        
        # Urgency distribution
        cursor.execute("""
            SELECT urgency, COUNT(*) as count FROM grievances
            GROUP BY urgency ORDER BY count DESC
        """)
        urgencies = [{"label": row["urgency"], "count": row["count"]} for row in cursor.fetchall()]
        
        # Status distribution
        cursor.execute("""
            SELECT status, COUNT(*) as count FROM grievances
            GROUP BY status ORDER BY count DESC
        """)
        statuses = [{"label": row["status"], "count": row["count"]} for row in cursor.fetchall()]
        
        return {
            "total_grievances": total,
            "confirmed_grievances": confirmed,
            "draft_grievances": drafts,
            "categories": categories,
            "departments": departments,
            "urgencies": urgencies,
            "statuses": statuses
        }

def save_conversation_turn(session_id: str, role: str, message: str, metadata: str = "") -> None:
    """Record a chat turn in session history."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO conversations (session_id, role, message, metadata)
            VALUES (?, ?, ?, ?)
        """, (session_id, role, message, metadata))
        conn.commit()

def get_session_history(session_id: str, limit: int = 20) -> List[Dict[str, Any]]:
    """Retrieve chat history for a session."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT role, message, created_at FROM conversations
            WHERE session_id = ? ORDER BY id ASC LIMIT ?
        """, (session_id, limit))
        return [dict(row) for row in cursor.fetchall()]

def save_feedback(grievance_id: Optional[str], rating: int, comments: str) -> None:
    """Record citizen feedback."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO feedback (grievance_id, rating, comments)
            VALUES (?, ?, ?)
        """, (grievance_id, rating, comments))
        conn.commit()


# ─── Admin Portal Functions ────────────────────────────────────────────────────

VALID_STATUSES = ("DRAFT", "CONFIRMED", "UNDER_REVIEW", "IN_PROGRESS", "FORWARDED", "RESOLVED")

def update_grievance_by_admin(
    grievance_id: str,
    new_status: str,
    admin_remarks: str = ""
) -> bool:
    """Admin updates grievance status and internal remarks. Sets resolved_at if RESOLVED."""
    if new_status not in VALID_STATUSES:
        return False
    now = datetime.now().isoformat()
    resolved_at = now if new_status == "RESOLVED" else None
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE grievances
            SET status = ?,
                admin_remarks = ?,
                updated_at = ?,
                resolved_at = CASE WHEN ? = 'RESOLVED' THEN ? ELSE resolved_at END
            WHERE grievance_id = ?
        """, (new_status, admin_remarks, now, new_status, resolved_at, grievance_id))
        conn.commit()
        return cursor.rowcount > 0


def get_admin_dashboard_stats() -> Dict[str, Any]:
    """Return status-level counts for the admin portal KPI cards."""
    with get_db_connection() as conn:
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM grievances")
        total = cursor.fetchone()[0]

        cursor.execute("SELECT status, COUNT(*) as cnt FROM grievances GROUP BY status")
        by_status: Dict[str, int] = {row["status"]: row["cnt"] for row in cursor.fetchall()}

        cursor.execute("""
            SELECT department, COUNT(*) as cnt
            FROM grievances GROUP BY department ORDER BY cnt DESC LIMIT 5
        """)
        top_depts = [{"department": r["department"], "count": r["cnt"]} for r in cursor.fetchall()]

        return {
            "total": total,
            "confirmed": by_status.get("CONFIRMED", 0),
            "under_review": by_status.get("UNDER_REVIEW", 0),
            "in_progress": by_status.get("IN_PROGRESS", 0),
            "forwarded": by_status.get("FORWARDED", 0),
            "resolved": by_status.get("RESOLVED", 0),
            "draft": by_status.get("DRAFT", 0),
            "top_departments": top_depts,
        }


def get_grievance_for_tracker(grievance_id: str) -> Optional[Dict[str, Any]]:
    """
    Citizen-safe tracker lookup: returns public fields only.
    Excludes internal fields like original_text / normalized_text.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT grievance_id, created_at, updated_at, category, department,
                   urgency, location, duration, summary, status, admin_remarks, resolved_at
            FROM grievances
            WHERE grievance_id = ?
        """, (grievance_id.strip().upper(),))
        row = cursor.fetchone()
        return dict(row) if row else None

