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
    seed_initial_grievances()


SEED_GRIEVANCES = [
    {
        "grievance_id": "GRV-2026-000101",
        "original_text": "Severe drinking water pipeline breakage leading to street flooding and water supply shortage in Anna Nagar 4th Main Road.",
        "normalized_text": "Severe drinking water pipeline breakage leading to street flooding and water supply shortage in Anna Nagar 4th Main Road.",
        "category": "water_supply",
        "department": "Water Supply Department",
        "urgency": "high",
        "severity": "critical",
        "location": "Anna Nagar 4th Main Road, Zone 8",
        "district": "Chennai Metro",
        "duration": "5 days",
        "summary": "Severe Pipeline Leakage & Drinking Water Shortage",
        "generated_grievance": "To: The Executive Engineer, Water Supply & Sewerage Board\nSubject: Urgent Request to Repair Main Drinking Water Pipeline Leakage at Anna Nagar 4th Main Road\n\nRespected Sir/Madam,\n\nI am writing to report a major main pipeline breach at Anna Nagar 4th Main Road. Drinking water has been severely disrupted for over 5 days, causing severe inconvenience to residents and contamination risks.\n\nWe request immediate deployment of repair teams to restore regular water supply.",
        "status": "CONFIRMED",
        "admin_remarks": "Inspection team dispatched. Repair crew assigned for pipeline welding."
    },
    {
        "grievance_id": "GRV-2026-000102",
        "original_text": "Hazardous deep potholes and broken asphalt on Katpadi Main Road causing vehicle damage and traffic bottlenecks near bus stand.",
        "normalized_text": "Hazardous deep potholes and broken asphalt on Katpadi Main Road causing vehicle damage and traffic bottlenecks near bus stand.",
        "category": "road_maintenance",
        "department": "Roads & Infrastructure Department",
        "urgency": "high",
        "severity": "safety_hazard",
        "location": "Katpadi Main Road, Near Bus Stand",
        "district": "Vellore Zone",
        "duration": "2 weeks",
        "summary": "Hazardous Potholes and Road Damage on Katpadi Main Road",
        "generated_grievance": "To: Divisional Engineer, Highways & Urban Roads Division\nSubject: Complaint Regarding Hazardous Potholes on Katpadi Main Road\n\nRespected Sir/Madam,\n\nThis is to bring to your urgent attention the dangerous condition of Katpadi Main Road near the Bus Stand. Multiple deep potholes have opened up over the past 2 weeks, leading to frequent minor accidents and major traffic congestion.\n\nKindly initiate urgent resurfacing and pothole patching work.",
        "status": "UNDER_REVIEW",
        "admin_remarks": "Work order issued to contractor for asphalt filling."
    },
    {
        "grievance_id": "GRV-2026-000103",
        "original_text": "Uncollected municipal waste and overflowing garbage bins at Gandhi Market causing foul smell and health hazards.",
        "normalized_text": "Uncollected municipal waste and overflowing garbage bins at Gandhi Market causing foul smell and health hazards.",
        "category": "garbage_collection",
        "department": "Sanitation & Waste Management",
        "urgency": "medium",
        "severity": "health_hazard",
        "location": "Gandhi Street Market, Ward 12",
        "district": "Central Division",
        "duration": "3 days",
        "summary": "Uncollected Solid Waste & Overflowing Garbage Bins",
        "generated_grievance": "To: Health Officer & Sanitation Inspector, City Corporation\nSubject: Urgent Removal of Accumulated Waste at Gandhi Market\n\nRespected Sir/Madam,\n\nGarbage compactors have not visited the Gandhi Street Market area for 3 consecutive days. Waste is spilling onto the road, creating unhygienic conditions for shopkeepers and shoppers alike.\n\nWe request immediate clearance of the dumpsters.",
        "status": "UNDER_REVIEW",
        "admin_remarks": "Sanitation supervisor notified for immediate clearance."
    },
    {
        "grievance_id": "GRV-2026-000104",
        "original_text": "Frequent transformer voltage fluctuations and unannounced power outages in T. Nagar 7th Cross Street.",
        "normalized_text": "Frequent transformer voltage fluctuations and unannounced power outages in T. Nagar 7th Cross Street.",
        "category": "electricity",
        "department": "Electricity Board (TANGEDCO)",
        "urgency": "high",
        "severity": "service_issue",
        "location": "T. Nagar 7th Cross Street",
        "district": "South Zone",
        "duration": "4 days",
        "summary": "Frequent Power Outages & High Voltage Fluctuations",
        "generated_grievance": "To: Superintending Engineer, Electricity Distribution Board\nSubject: Complaint Regarding Unstable Voltage and Daily Power Outages\n\nRespected Sir/Madam,\n\nResidents of T. Nagar 7th Cross Street are facing continuous voltage spikes and power trips due to an overloaded local transformer. Household appliances are at risk of damage.\n\nKindly inspect the transformer unit and step up line stability.",
        "status": "UNDER_REVIEW",
        "admin_remarks": "Forwarded to Electrical Sub-Station Superintending Engineer."
    },
    {
        "grievance_id": "GRV-2026-000105",
        "original_text": "Non-functional street lights along National Highway Bypass service road creating dark spots and safety concerns for commuters at night.",
        "normalized_text": "Non-functional street lights along National Highway Bypass service road creating dark spots and safety concerns for commuters at night.",
        "category": "street_lighting",
        "department": "Civic Maintenance & Street Lighting",
        "urgency": "medium",
        "severity": "safety_hazard",
        "location": "National Highway Bypass Service Road",
        "district": "North Division",
        "duration": "10 days",
        "summary": "Non-Functional Street Lights along Bypass Road",
        "generated_grievance": "To: Assistant Executive Engineer, Electrical & Public Lighting Wing\nSubject: Restoration of Street Lighting along NH Bypass Service Road\n\nRespected Sir/Madam,\n\nOver 15 LED street lights along the NH Bypass Service Road have been non-functional for 10 days, posing severe safety risks to pedestrian and vehicular traffic during nighttime.\n\nKindly replace defective bulbs and check line fuses promptly.",
        "status": "RESOLVED",
        "admin_remarks": "LED fixtures replaced and main control panel switch rectified. Resolved on 2026-10-06."
    },
    {
        "grievance_id": "GRV-2026-000106",
        "original_text": "Overflowing underground drainage chamber near Primary Health Centre spilling sewage water onto public road.",
        "normalized_text": "Overflowing underground drainage chamber near Primary Health Centre spilling sewage water onto public road.",
        "category": "drainage_sewage",
        "department": "Sanitation & Waste Management",
        "urgency": "high",
        "severity": "health_hazard",
        "location": "Adyar 2nd Avenue, Near PHC Clinic",
        "district": "South Chennai",
        "duration": "2 days",
        "summary": "Overflowing Underground Drainage Chamber",
        "generated_grievance": "To: Superintending Engineer, Metropolitan Drainage Board\nSubject: Emergency Clearance of Overflowing Drainage near PHC Clinic\n\nRespected Sir/Madam,\n\nAn underground sewage chamber has clogged and overflowed in front of the Primary Health Centre, presenting a direct health risk to patients and visitors.\n\nWe request super-sucker jetting machines to clear the blockage immediately.",
        "status": "CONFIRMED",
        "admin_remarks": "Jetting machine truck routed to location."
    },
    {
        "grievance_id": "GRV-2026-000107",
        "original_text": "Brownish muddy water arriving through municipal taps in Velachery Extension 3rd Block.",
        "normalized_text": "Brownish muddy water arriving through municipal taps in Velachery Extension 3rd Block.",
        "category": "water_quality",
        "department": "Water Supply Department",
        "urgency": "high",
        "severity": "health_hazard",
        "location": "Velachery Extension 3rd Block",
        "district": "South East Zone",
        "duration": "1 week",
        "summary": "Contaminated Tap Water Supply with High Turbidity",
        "generated_grievance": "To: Chief Quality Analyst, Public Health & Water Testing Division\nSubject: Complaint Regarding Contaminated Municipal Drinking Water\n\nRespected Sir/Madam,\n\nResidents of Velachery Extension 3rd Block are receiving muddy, discolored tap water for the past 7 days. Soil infiltration into distribution lines is suspected.\n\nPlease sample water quality and flush affected supply lines.",
        "status": "UNDER_REVIEW",
        "admin_remarks": "Water samples collected for lab analysis; line flushing underway."
    },
    {
        "grievance_id": "GRV-2026-000108",
        "original_text": "Stagnant rainwater pool in vacant plot near Mylapore 5th Layout causing heavy mosquito breeding and fever outbreak threat.",
        "normalized_text": "Stagnant rainwater pool in vacant plot near Mylapore 5th Layout causing heavy mosquito breeding and fever outbreak threat.",
        "category": "public_health",
        "department": "Public Health Department",
        "urgency": "medium",
        "severity": "health_hazard",
        "location": "Mylapore 5th Layout",
        "district": "Central Chennai",
        "duration": "10 days",
        "summary": "Stagnant Water Accumulation & Mosquito Vector Control",
        "generated_grievance": "To: City Vector Control Officer, Department of Public Health\nSubject: Request for Anti-Mosquito Fogging and De-watering in Vacant Plot\n\nRespected Sir/Madam,\n\nRainwater has stagnated in an abandoned open plot in Mylapore 5th Layout for over 10 days, becoming a breeding ground for mosquitoes.\n\nWe request immediate de-watering and anti-larval chemical spraying.",
        "status": "CONFIRMED",
        "admin_remarks": "Vector control team scheduled for fogging drive."
    },
    {
        "grievance_id": "GRV-2026-000109",
        "original_text": "Unmarked broken speed breakers on Tambaram Main Road causing severe jolts to elderly passengers and vehicle underscribe damage.",
        "normalized_text": "Unmarked broken speed breakers on Tambaram Main Road causing severe jolts to elderly passengers and vehicle underscribe damage.",
        "category": "road_maintenance",
        "department": "Roads & Infrastructure Department",
        "urgency": "medium",
        "severity": "safety_hazard",
        "location": "Tambaram Main Road Crossing",
        "district": "South Outer Zone",
        "duration": "3 weeks",
        "summary": "Unmarked Damaged Speed Breaker requiring Reflective Painting",
        "generated_grievance": "To: Traffic Engineering Cell, Highways Division\nSubject: Request to Repair and Paint Speed Breakers on Tambaram Main Road\n\nRespected Sir/Madam,\n\nUnpainted and damaged speed breakers on Tambaram Main Road pose severe hazards at night due to poor visibility.\n\nWe request repainting with luminous yellow paint and installation of warning cat-eyes.",
        "status": "RESOLVED",
        "admin_remarks": "Speed humps repainted with reflective yellow strips on 2026-10-05."
    },
    {
        "grievance_id": "GRV-2026-000110",
        "original_text": "Hanging high-tension electrical wire near Chromepet Children's Park posing extreme electrocution hazard.",
        "normalized_text": "Hanging high-tension electrical wire near Chromepet Children's Park posing extreme electrocution hazard.",
        "category": "electricity",
        "department": "Electricity Board (TANGEDCO)",
        "urgency": "high",
        "severity": "life_safety",
        "location": "Chromepet Park Avenue",
        "district": "South Zone",
        "duration": "1 day",
        "summary": "Dangling Overhead High Voltage Line near Children's Park",
        "generated_grievance": "To: Assistant Engineer (O&M), Electricity Distribution Circle\nSubject: CRITICAL: Dangling Overhead Power Cable near Chromepet Children Park\n\nRespected Sir/Madam,\n\nA snapping branch has damaged a utility pole cross-arm, causing a high-tension cable to hang low near the park entrance.\n\nURGENT action required to isolate power and re-tension the cable line.",
        "status": "CONFIRMED",
        "admin_remarks": "Emergency lineman squad dispatched to isolate grid segment."
    }
]

def seed_initial_grievances(force: bool = False) -> None:
    """Seed sample grievances if the table is empty or needs seeding."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM grievances")
        count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT department) FROM grievances")
        dept_count = cursor.fetchone()[0]
        
        if count == 0 or dept_count < 3 or force:
            if force or count < 10:
                cursor.execute("DELETE FROM grievances WHERE grievance_id LIKE 'GRV-2026-00000%' OR grievance_id LIKE 'GRV-2026-0001%'")
            for g in SEED_GRIEVANCES:
                cursor.execute("""
                    INSERT OR REPLACE INTO grievances (
                        grievance_id, original_text, normalized_text, category, department,
                        urgency, severity, location, district, duration, summary,
                        generated_grievance, status, admin_remarks
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    g["grievance_id"], g["original_text"], g["normalized_text"], g["category"],
                    g["department"], g["urgency"], g["severity"], g["location"], g["district"],
                    g["duration"], g["summary"], g["generated_grievance"], g["status"], g["admin_remarks"]
                ))
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

