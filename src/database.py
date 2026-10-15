"""
SQLite Database interface for GrievanceGPT.
Manages persistent storage for grievances, multi-turn conversations, and user feedback.
Engineered with ACID transaction safety, verified metrics aggregation, and audit resilience.
"""
import sqlite3
import logging
from contextlib import contextmanager
from datetime import datetime
from typing import List, Dict, Any, Optional
from src.config import DATABASE_PATH

logger = logging.getLogger("GrievanceGPT.Database")

VALID_STATUSES = ("DRAFT", "CONFIRMED", "UNDER_REVIEW", "IN_PROGRESS", "FORWARDED", "RESOLVED")

SUBMITTED_STATUSES = ("CONFIRMED", "UNDER_REVIEW", "IN_PROGRESS", "FORWARDED", "RESOLVED")
PENDING_STATUSES = ("CONFIRMED", "UNDER_REVIEW")
ACTIVE_PROGRESS_STATUSES = ("IN_PROGRESS", "FORWARDED")

def get_db_connection() -> sqlite3.Connection:
    """Establish and return a SQLite database connection with row factory."""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn

@contextmanager
def db_session():
    """
    Context manager for SQLite connections.
    Guarantees automatic commit on success, rollback on error, and reliable connection closure.
    """
    conn = get_db_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

def init_db(seed: bool = True) -> None:
    """Initialize SQLite database tables if they do not exist."""
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with db_session() as conn:
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
        
        # Safe ALTER TABLE: ensure all expected columns exist
        for col, col_def in [
            ("admin_remarks", "TEXT DEFAULT ''"),
            ("resolved_at", "TIMESTAMP"),
        ]:
            try:
                cursor.execute(f"ALTER TABLE grievances ADD COLUMN {col} {col_def}")
            except Exception:
                pass  # Column already exists
        
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
        
    if seed:
        seed_initial_grievances()


SEED_GRIEVANCES = [
    {
        "grievance_id": "GRV-2026-000101",
        "created_at": "2026-10-01T09:15:00",
        "updated_at": "2026-10-01T09:15:00",
        "original_text": "Severe drinking water pipeline breakage leading to street flooding and water supply shortage in Anna Nagar 4th Main Road.",
        "normalized_text": "Severe drinking water pipeline breakage leading to street flooding and water supply shortage in Anna Nagar 4th Main Road.",
        "category": "water_supply",
        "department": "Water Supply Department",
        "urgency": "high",
        "severity": "critical",
        "location": "Anna Nagar 4th Main Road, Zone 8",
        "district": "Chennai",
        "duration": "5 days",
        "summary": "Severe Pipeline Leakage & Drinking Water Shortage",
        "generated_grievance": "To: The Executive Engineer, Water Supply & Sewerage Board\nSubject: Urgent Request to Repair Main Drinking Water Pipeline Leakage at Anna Nagar 4th Main Road\n\nRespected Sir/Madam,\n\nI am writing to report a major main pipeline breach at Anna Nagar 4th Main Road. Drinking water has been severely disrupted for over 5 days, causing severe inconvenience to residents and contamination risks.\n\nWe request immediate deployment of repair teams to restore regular water supply.",
        "status": "CONFIRMED",
        "admin_remarks": "Inspection team dispatched. Repair crew assigned for pipeline welding."
    },
    {
        "grievance_id": "GRV-2026-000102",
        "created_at": "2026-10-01T14:30:00",
        "updated_at": "2026-10-02T10:00:00",
        "original_text": "Hazardous deep potholes and broken asphalt on Katpadi Main Road causing vehicle damage and traffic bottlenecks near bus stand.",
        "normalized_text": "Hazardous deep potholes and broken asphalt on Katpadi Main Road causing vehicle damage and traffic bottlenecks near bus stand.",
        "category": "road_maintenance",
        "department": "Roads & Infrastructure Department",
        "urgency": "high",
        "severity": "safety_hazard",
        "location": "Katpadi Main Road, Near Bus Stand",
        "district": "Vellore",
        "duration": "2 weeks",
        "summary": "Hazardous Potholes and Road Damage on Katpadi Main Road",
        "generated_grievance": "To: Divisional Engineer, Highways & Urban Roads Division\nSubject: Complaint Regarding Hazardous Potholes on Katpadi Main Road\n\nRespected Sir/Madam,\n\nThis is to bring to your urgent attention the dangerous condition of Katpadi Main Road near the Bus Stand. Multiple deep potholes have opened up over the past 2 weeks, leading to frequent minor accidents and major traffic congestion.\n\nKindly initiate urgent resurfacing and pothole patching work.",
        "status": "UNDER_REVIEW",
        "admin_remarks": "Work order issued to contractor for asphalt filling."
    },
    {
        "grievance_id": "GRV-2026-000103",
        "created_at": "2026-10-02T08:45:00",
        "updated_at": "2026-10-02T16:20:00",
        "original_text": "Uncollected municipal waste and overflowing garbage bins at Gandhi Market causing foul smell and health hazards.",
        "normalized_text": "Uncollected municipal waste and overflowing garbage bins at Gandhi Market causing foul smell and health hazards.",
        "category": "garbage_collection",
        "department": "Sanitation & Waste Management",
        "urgency": "medium",
        "severity": "health_hazard",
        "location": "Gandhi Street Market, Ward 12",
        "district": "Chennai",
        "duration": "3 days",
        "summary": "Uncollected Solid Waste & Overflowing Garbage Bins",
        "generated_grievance": "To: Health Officer & Sanitation Inspector, City Corporation\nSubject: Urgent Removal of Accumulated Waste at Gandhi Market\n\nRespected Sir/Madam,\n\nGarbage compactors have not visited the Gandhi Street Market area for 3 consecutive days. Waste is spilling onto the road, creating unhygienic conditions for shopkeepers and shoppers alike.\n\nWe request immediate clearance of the dumpsters.",
        "status": "UNDER_REVIEW",
        "admin_remarks": "Sanitation supervisor notified for immediate clearance."
    },
    {
        "grievance_id": "GRV-2026-000104",
        "created_at": "2026-10-02T11:10:00",
        "updated_at": "2026-10-03T09:00:00",
        "original_text": "Frequent transformer voltage fluctuations and unannounced power outages in T. Nagar 7th Cross Street.",
        "normalized_text": "Frequent transformer voltage fluctuations and unannounced power outages in T. Nagar 7th Cross Street.",
        "category": "electricity",
        "department": "Electricity Board (TANGEDCO)",
        "urgency": "high",
        "severity": "service_issue",
        "location": "T. Nagar 7th Cross Street",
        "district": "Chennai",
        "duration": "4 days",
        "summary": "Frequent Power Outages & High Voltage Fluctuations",
        "generated_grievance": "To: Superintending Engineer, Electricity Distribution Board\nSubject: Complaint Regarding Unstable Voltage and Daily Power Outages\n\nRespected Sir/Madam,\n\nResidents of T. Nagar 7th Cross Street are facing continuous voltage spikes and power trips due to an overloaded local transformer. Household appliances are at risk of damage.\n\nKindly inspect the transformer unit and step up line stability.",
        "status": "UNDER_REVIEW",
        "admin_remarks": "Forwarded to Electrical Sub-Station Superintending Engineer."
    },
    {
        "grievance_id": "GRV-2026-000105",
        "created_at": "2026-10-02T16:00:00",
        "updated_at": "2026-10-06T15:30:00",
        "resolved_at": "2026-10-06T15:30:00",
        "original_text": "Non-functional street lights along National Highway Bypass service road creating dark spots and safety concerns for commuters at night.",
        "normalized_text": "Non-functional street lights along National Highway Bypass service road creating dark spots and safety concerns for commuters at night.",
        "category": "street_lighting",
        "department": "Civic Maintenance & Street Lighting",
        "urgency": "medium",
        "severity": "safety_hazard",
        "location": "National Highway Bypass Service Road",
        "district": "Chennai",
        "duration": "10 days",
        "summary": "Non-Functional Street Lights along Bypass Road",
        "generated_grievance": "To: Assistant Executive Engineer, Electrical & Public Lighting Wing\nSubject: Restoration of Street Lighting along NH Bypass Service Road\n\nRespected Sir/Madam,\n\nOver 15 LED street lights along the NH Bypass Service Road have been non-functional for 10 days, posing severe safety risks to pedestrian and vehicular traffic during nighttime.\n\nKindly replace defective bulbs and check line fuses promptly.",
        "status": "RESOLVED",
        "admin_remarks": "LED fixtures replaced and main control panel switch rectified. Resolved on 2026-10-06."
    },
    {
        "grievance_id": "GRV-2026-000106",
        "created_at": "2026-10-03T10:20:00",
        "updated_at": "2026-10-03T10:20:00",
        "original_text": "Overflowing underground drainage chamber near Primary Health Centre spilling sewage water onto public road.",
        "normalized_text": "Overflowing underground drainage chamber near Primary Health Centre spilling sewage water onto public road.",
        "category": "drainage_sewage",
        "department": "Sanitation & Waste Management",
        "urgency": "high",
        "severity": "health_hazard",
        "location": "Adyar 2nd Avenue, Near PHC Clinic",
        "district": "Chennai",
        "duration": "2 days",
        "summary": "Overflowing Underground Drainage Chamber",
        "generated_grievance": "To: Superintending Engineer, Metropolitan Drainage Board\nSubject: Emergency Clearance of Overflowing Drainage near PHC Clinic\n\nRespected Sir/Madam,\n\nAn underground sewage chamber has clogged and overflowed in front of the Primary Health Centre, presenting a direct health risk to patients and visitors.\n\nWe request super-sucker jetting machines to clear the blockage immediately.",
        "status": "CONFIRMED",
        "admin_remarks": "Jetting machine truck routed to location."
    },
    {
        "grievance_id": "GRV-2026-000107",
        "created_at": "2026-10-03T13:40:00",
        "updated_at": "2026-10-04T11:15:00",
        "original_text": "Brownish muddy water arriving through municipal taps in Velachery Extension 3rd Block.",
        "normalized_text": "Brownish muddy water arriving through municipal taps in Velachery Extension 3rd Block.",
        "category": "water_quality",
        "department": "Water Supply Department",
        "urgency": "high",
        "severity": "health_hazard",
        "location": "Velachery Extension 3rd Block",
        "district": "Chennai",
        "duration": "1 week",
        "summary": "Contaminated Tap Water Supply with High Turbidity",
        "generated_grievance": "To: Chief Quality Analyst, Public Health & Water Testing Division\nSubject: Complaint Regarding Contaminated Municipal Drinking Water\n\nRespected Sir/Madam,\n\nResidents of Velachery Extension 3rd Block are receiving muddy, discolored tap water for the past 7 days. Soil infiltration into distribution lines is suspected.\n\nPlease sample water quality and flush affected supply lines.",
        "status": "UNDER_REVIEW",
        "admin_remarks": "Water samples collected for lab analysis; line flushing underway."
    },
    {
        "grievance_id": "GRV-2026-000108",
        "created_at": "2026-10-04T09:05:00",
        "updated_at": "2026-10-04T09:05:00",
        "original_text": "Stagnant rainwater pool in vacant plot near Mylapore 5th Layout causing heavy mosquito breeding and fever outbreak threat.",
        "normalized_text": "Stagnant rainwater pool in vacant plot near Mylapore 5th Layout causing heavy mosquito breeding and fever outbreak threat.",
        "category": "public_health",
        "department": "Public Health Department",
        "urgency": "medium",
        "severity": "health_hazard",
        "location": "Mylapore 5th Layout",
        "district": "Chennai",
        "duration": "10 days",
        "summary": "Stagnant Water Accumulation & Mosquito Vector Control",
        "generated_grievance": "To: City Vector Control Officer, Department of Public Health\nSubject: Request for Anti-Mosquito Fogging and De-watering in Vacant Plot\n\nRespected Sir/Madam,\n\nRainwater has stagnated in an abandoned open plot in Mylapore 5th Layout for over 10 days, becoming a breeding ground for mosquitoes.\n\nWe request immediate de-watering and anti-larval chemical spraying.",
        "status": "CONFIRMED",
        "admin_remarks": "Vector control team scheduled for fogging drive."
    },
    {
        "grievance_id": "GRV-2026-000109",
        "created_at": "2026-10-01T10:00:00",
        "updated_at": "2026-10-05T14:00:00",
        "resolved_at": "2026-10-05T14:00:00",
        "original_text": "Unmarked broken speed breakers on Tambaram Main Road causing severe jolts to elderly passengers and vehicle underscribe damage.",
        "normalized_text": "Unmarked broken speed breakers on Tambaram Main Road causing severe jolts to elderly passengers and vehicle underscribe damage.",
        "category": "road_maintenance",
        "department": "Roads & Infrastructure Department",
        "urgency": "medium",
        "severity": "safety_hazard",
        "location": "Tambaram Main Road Crossing",
        "district": "Chennai",
        "duration": "3 weeks",
        "summary": "Unmarked Damaged Speed Breaker requiring Reflective Painting",
        "generated_grievance": "To: Traffic Engineering Cell, Highways Division\nSubject: Request to Repair and Paint Speed Breakers on Tambaram Main Road\n\nRespected Sir/Madam,\n\nUnpainted and damaged speed breakers on Tambaram Main Road pose severe hazards at night due to poor visibility.\n\nWe request repainting with luminous yellow paint and installation of warning cat-eyes.",
        "status": "RESOLVED",
        "admin_remarks": "Speed humps repainted with reflective yellow strips on 2026-10-05."
    },
    {
        "grievance_id": "GRV-2026-000110",
        "created_at": "2026-10-05T11:30:00",
        "updated_at": "2026-10-05T11:30:00",
        "original_text": "Hanging high-tension electrical wire near Chromepet Children's Park posing extreme electrocution hazard.",
        "normalized_text": "Hanging high-tension electrical wire near Chromepet Children's Park posing extreme electrocution hazard.",
        "category": "electricity",
        "department": "Electricity Board (TANGEDCO)",
        "urgency": "high",
        "severity": "life_safety",
        "location": "Chromepet Park Avenue",
        "district": "Chennai",
        "duration": "1 day",
        "summary": "Dangling Overhead High Voltage Line near Children's Park",
        "generated_grievance": "To: Assistant Engineer (O&M), Electricity Distribution Circle\nSubject: CRITICAL: Dangling Overhead Power Cable near Chromepet Children Park\n\nRespected Sir/Madam,\n\nA snapping branch has damaged a utility pole cross-arm, causing a high-tension cable to hang low near the park entrance.\n\nURGENT action required to isolate power and re-tension the cable line.",
        "status": "CONFIRMED",
        "admin_remarks": "Emergency lineman squad dispatched to isolate grid segment."
    }
]

def seed_initial_grievances(force: bool = False) -> None:
    """
    Seed initial baseline grievances only if table is empty or explicitly forced.
    NEVER deletes citizen complaints or runs wildcard deletions on user tokens.
    """
    with db_session() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM grievances")
        count = cursor.fetchone()[0]
        
        if count == 0 or force:
            for g in SEED_GRIEVANCES:
                cursor.execute("""
                    INSERT OR REPLACE INTO grievances (
                        grievance_id, created_at, updated_at, original_text, normalized_text,
                        category, department, urgency, severity, location, district, duration,
                        summary, generated_grievance, status, admin_remarks, resolved_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    g["grievance_id"], g.get("created_at", datetime.now().isoformat()),
                    g.get("updated_at", datetime.now().isoformat()),
                    g["original_text"], g["normalized_text"], g["category"],
                    g["department"], g["urgency"], g["severity"], g["location"], g["district"],
                    g["duration"], g["summary"], g["generated_grievance"], g["status"],
                    g.get("admin_remarks", ""), g.get("resolved_at")
                ))
            return len(SEED_GRIEVANCES)
        return 0


def generate_grievance_id() -> str:
    """Generate a unique sequential grievance identifier e.g., GRV-2026-000111."""
    current_year = datetime.now().year
    with db_session() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT grievance_id FROM grievances")
        rows = cursor.fetchall()
        max_seq = 0
        for r in rows:
            gid = r[0]
            if gid and '-' in gid:
                parts = gid.split('-')
                if parts[-1].isdigit():
                    max_seq = max(max_seq, int(parts[-1]))
        seq = max(max_seq, len(rows)) + 1
        return f"GRV-{current_year}-{seq:06d}"

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
    """Insert or update a structured grievance record with transaction safety."""
    gid = grievance_id or generate_grievance_id()
    now = datetime.now().isoformat()
    with db_session() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO grievances (
                grievance_id, created_at, updated_at, original_text, normalized_text,
                category, department, urgency, severity, location, district, duration,
                summary, generated_grievance, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(grievance_id) DO UPDATE SET
                updated_at = excluded.updated_at,
                original_text = excluded.original_text,
                normalized_text = excluded.normalized_text,
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
    return gid

def update_grievance_status(grievance_id: str, new_status: str) -> bool:
    """Update grievance status. Sets or clears resolved_at appropriately."""
    now = datetime.now().isoformat()
    with db_session() as conn:
        cursor = conn.cursor()
        if new_status == "RESOLVED":
            cursor.execute("""
                UPDATE grievances
                SET status = ?, updated_at = ?, resolved_at = COALESCE(resolved_at, ?)
                WHERE grievance_id = ?
            """, (new_status, now, now, grievance_id))
        else:
            cursor.execute("""
                UPDATE grievances
                SET status = ?, updated_at = ?, resolved_at = CASE WHEN status = 'RESOLVED' THEN NULL ELSE resolved_at END
                WHERE grievance_id = ?
            """, (new_status, now, grievance_id))
        return cursor.rowcount > 0

def get_grievance(grievance_id: str) -> Optional[Dict[str, Any]]:
    """Fetch single grievance by ID."""
    with db_session() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM grievances WHERE grievance_id = ?", (grievance_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

def list_grievances(
    status: Optional[str] = None,
    department: Optional[str] = None,
    search: Optional[str] = None,
    limit: int = 200,
    offset: int = 0
) -> List[Dict[str, Any]]:
    """Query list of grievances with parameterized multi-field filtering and flexible search."""
    query = "SELECT * FROM grievances WHERE 1=1"
    params: List[Any] = []

    # Clean status filter
    status_filter = None
    if status:
        s_clean = status.strip().upper()
        if s_clean not in ("", "ALL", "ALL STATUSES", "ALL_STATUSES", "ANY", "NONE"):
            status_filter = s_clean

    # Clean department filter
    dept_filter = None
    if department:
        d_clean = department.strip()
        if d_clean.lower() not in ("", "all", "all departments", "all_departments", "any", "none"):
            dept_filter = d_clean

    if search and search.strip():
        s = search.strip()
        s_hl = s.replace("-", "").replace(" ", "")
        s_sp = s.replace("_", " ")
        s_un = s.replace(" ", "_")
        term = f"%{s}%"
        term_hl = f"%{s_hl}%"
        term_sp = f"%{s_sp}%"
        term_un = f"%{s_un}%"

        is_id_search = bool(
            ("GRV" in s.upper()) or
            (s_hl.isdigit() and len(s_hl) >= 3)
        )

        search_clause = """(
            grievance_id LIKE ?
            OR REPLACE(REPLACE(grievance_id, '-', ''), ' ', '') LIKE ?
            OR summary LIKE ?
            OR original_text LIKE ?
            OR generated_grievance LIKE ?
            OR department LIKE ?
            OR location LIKE ?
            OR district LIKE ?
            OR status LIKE ?
            OR status LIKE ?
            OR REPLACE(status, '_', ' ') LIKE ?
            OR category LIKE ?
            OR REPLACE(category, '_', ' ') LIKE ?
            OR urgency LIKE ?
            OR admin_remarks LIKE ?
        )"""
        search_params = [
            term, term_hl,
            term, term, term,
            term, term, term,
            term, term_un, term_sp,
            term, term_sp,
            term, term
        ]

        if is_id_search:
            # When searching by Grievance ID, allow matching the ID directly even if a status or dept filter was active
            id_clause = "(grievance_id LIKE ? OR REPLACE(REPLACE(grievance_id, '-', ''), ' ', '') LIKE ?)"
            id_params = [term, term_hl]

            combined_filters = []
            combined_params = []
            if status_filter:
                combined_filters.append("UPPER(status) = ?")
                combined_params.append(status_filter)
            if dept_filter:
                combined_filters.append("(department = ? OR department LIKE ?)")
                combined_params.extend([dept_filter, f"%{dept_filter}%"])

            if combined_filters:
                filter_sql = " AND ".join(combined_filters)
                query += f" AND (({filter_sql} AND {search_clause}) OR {id_clause})"
                params.extend(combined_params + search_params + id_params)
            else:
                query += f" AND {search_clause}"
                params.extend(search_params)
        else:
            if status_filter:
                query += " AND UPPER(status) = ?"
                params.append(status_filter)
            if dept_filter:
                query += " AND (department = ? OR department LIKE ?)"
                params.extend([dept_filter, f"%{dept_filter}%"])
            query += f" AND {search_clause}"
            params.extend(search_params)
    else:
        if status_filter:
            query += " AND UPPER(status) = ?"
            params.append(status_filter)
        if dept_filter:
            query += " AND (department = ? OR department LIKE ?)"
            params.extend([dept_filter, f"%{dept_filter}%"])

    query += " ORDER BY id DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    with db_session() as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]

def _parse_timestamp(ts: Any) -> Optional[datetime]:
    """Helper to parse timestamps in either ISO 8601 or SQLite SQL format."""
    if not ts:
        return None
    s = str(ts).strip()
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None

def get_analytics_summary(
    department: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> Dict[str, Any]:
    """
    Calculate verifiable aggregated metrics for dashboard charts.
    Adheres strictly to documented business rules:
    - total_registered: all records matching filter
    - total_submitted: CONFIRMED, UNDER_REVIEW, IN_PROGRESS, FORWARDED, RESOLVED
    - draft_grievances: DRAFT
    - pending_action: CONFIRMED, UNDER_REVIEW
    - in_progress: IN_PROGRESS, FORWARDED
    - resolved_grievances: RESOLVED
    - average_resolution_time: computed ONLY on resolved records with valid resolved_at > created_at
    """
    filter_sql = " WHERE 1=1"
    params: List[Any] = []

    if department and department.lower() not in ("all", ""):
        filter_sql += " AND department = ?"
        params.append(department)

    if start_date:
        filter_sql += " AND created_at >= ?"
        params.append(start_date)

    if end_date:
        filter_sql += " AND created_at <= ?"
        params.append(end_date)

    with db_session() as conn:
        cursor = conn.cursor()

        # 1. Total registered
        cursor.execute(f"SELECT COUNT(*) FROM grievances{filter_sql}", params)
        total_registered = cursor.fetchone()[0]

        # 2. Confirmed & Submitted citizen grievances (non-draft)
        cursor.execute(
            f"SELECT COUNT(*) FROM grievances{filter_sql} AND status IN ('CONFIRMED', 'UNDER_REVIEW', 'IN_PROGRESS', 'FORWARDED', 'RESOLVED')",
            params
        )
        total_submitted = cursor.fetchone()[0]

        # 3. Draft grievances
        cursor.execute(f"SELECT COUNT(*) FROM grievances{filter_sql} AND status = 'DRAFT'", params)
        draft_grievances = cursor.fetchone()[0]

        # 4. Pending administrative action
        cursor.execute(f"SELECT COUNT(*) FROM grievances{filter_sql} AND status IN ('CONFIRMED', 'UNDER_REVIEW')", params)
        pending_action = cursor.fetchone()[0]

        # 5. In-Progress & Forwarded
        cursor.execute(f"SELECT COUNT(*) FROM grievances{filter_sql} AND status IN ('IN_PROGRESS', 'FORWARDED')", params)
        in_progress = cursor.fetchone()[0]

        # 6. Resolved grievances
        cursor.execute(f"SELECT COUNT(*) FROM grievances{filter_sql} AND status = 'RESOLVED'", params)
        resolved_grievances = cursor.fetchone()[0]

        # 7. Status breakdown
        cursor.execute(
            f"SELECT status, COUNT(*) as count FROM grievances{filter_sql} GROUP BY status ORDER BY count DESC",
            params
        )
        status_rows = cursor.fetchall()
        by_status: Dict[str, int] = {st: 0 for st in VALID_STATUSES}
        statuses_list = []
        for r in status_rows:
            st = r["status"]
            cnt = r["count"]
            by_status[st] = cnt
            statuses_list.append({"label": st, "count": cnt})

        # 8. High priority and critical severity
        cursor.execute(f"SELECT COUNT(*) FROM grievances{filter_sql} AND LOWER(urgency) = 'high'", params)
        high_priority_count = cursor.fetchone()[0]

        cursor.execute(
            f"SELECT COUNT(*) FROM grievances{filter_sql} AND LOWER(severity) IN ('critical', 'safety_hazard', 'life_safety', 'health_hazard')",
            params
        )
        critical_severity_count = cursor.fetchone()[0]

        # 9. Resolution metrics calculation
        cursor.execute(
            f"SELECT created_at, resolved_at FROM grievances{filter_sql} AND status = 'RESOLVED' AND resolved_at IS NOT NULL",
            params
        )
        resolved_rows = cursor.fetchall()
        valid_durations_hours: List[float] = []

        for row in resolved_rows:
            c_at = _parse_timestamp(row["created_at"])
            r_at = _parse_timestamp(row["resolved_at"])
            if c_at and r_at and r_at > c_at:
                diff_hours = (r_at - c_at).total_seconds() / 3600.0
                valid_durations_hours.append(diff_hours)

        if valid_durations_hours:
            avg_hours = round(sum(valid_durations_hours) / len(valid_durations_hours), 1)
            fastest_hours = round(min(valid_durations_hours), 1)
            slowest_hours = round(max(valid_durations_hours), 1)
            if avg_hours < 24.0:
                avg_formatted = f"{avg_hours:.1f} hrs"
            else:
                days = avg_hours / 24.0
                avg_formatted = f"{days:.1f} days"
        else:
            avg_hours = None
            avg_formatted = "N/A"
            fastest_hours = None
            slowest_hours = None

        resolution_metrics = {
            "avg_hours": avg_hours,
            "avg_formatted": avg_formatted,
            "fastest_hours": fastest_hours,
            "slowest_hours": slowest_hours,
            "resolved_count_with_duration": len(valid_durations_hours),
            "total_resolved_count": resolved_grievances
        }

        # 10. Department distribution (breakdown of submitted grievances)
        cursor.execute(
            f"SELECT department, COUNT(*) as count FROM grievances{filter_sql} GROUP BY department ORDER BY count DESC",
            params
        )
        departments = [{"label": row["department"], "count": row["count"]} for row in cursor.fetchall()]

        # 11. Category distribution
        cursor.execute(
            f"SELECT category, COUNT(*) as count FROM grievances{filter_sql} GROUP BY category ORDER BY count DESC",
            params
        )
        categories = [{"label": row["category"], "count": row["count"]} for row in cursor.fetchall()]

        # 12. Urgency distribution
        cursor.execute(
            f"SELECT urgency, COUNT(*) as count FROM grievances{filter_sql} GROUP BY urgency ORDER BY count DESC",
            params
        )
        urgencies = [{"label": row["urgency"], "count": row["count"]} for row in cursor.fetchall()]

        # 13. Daily submission trend (last 30 distinct dates)
        cursor.execute(
            f"SELECT substr(created_at, 1, 10) as date, COUNT(*) as count FROM grievances{filter_sql} GROUP BY date ORDER BY date ASC LIMIT 30",
            params
        )
        daily_submissions = [{"date": row["date"], "count": row["count"]} for row in cursor.fetchall()]

    # 14. Real, verified ML classification performance
    try:
        from src.nlp.evaluation import get_verified_ml_metrics
        ml_metrics = get_verified_ml_metrics()
    except Exception as e:
        logger.warning(f"ML metrics fetch error: {e}")
        ml_metrics = {"is_verified": False, "tasks": {}}

    return {
        "total_grievances": total_registered,
        "total_registered": total_registered,
        "total_submitted": total_submitted,
        "confirmed_grievances": by_status.get("CONFIRMED", 0),
        "draft_grievances": draft_grievances,
        "pending_action": pending_action,
        "in_progress": in_progress,
        "resolved_grievances": resolved_grievances,
        "by_status": by_status,
        "high_priority_count": high_priority_count,
        "critical_severity_count": critical_severity_count,
        "resolution_metrics": resolution_metrics,
        "categories": categories,
        "departments": departments,
        "urgencies": urgencies,
        "statuses": statuses_list,
        "daily_submissions": daily_submissions,
        "ml_metrics": ml_metrics,
        "filter_applied": {
            "department": department or "all",
            "start_date": start_date or None,
            "end_date": end_date or None
        },
        "last_updated": datetime.now().isoformat()
    }

def save_conversation_turn(session_id: str, role: str, message: str, metadata: str = "") -> None:
    """Record a chat turn in session history."""
    with db_session() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO conversations (session_id, role, message, metadata)
            VALUES (?, ?, ?, ?)
        """, (session_id, role, message, metadata))

def get_session_history(session_id: str, limit: int = 20) -> List[Dict[str, Any]]:
    """Retrieve chat history for a session."""
    with db_session() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT role, message, created_at FROM conversations
            WHERE session_id = ? ORDER BY id ASC LIMIT ?
        """, (session_id, limit))
        return [dict(row) for row in cursor.fetchall()]

def save_feedback(grievance_id: Optional[str], rating: int, comments: str) -> None:
    """Record citizen feedback."""
    with db_session() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO feedback (grievance_id, rating, comments)
            VALUES (?, ?, ?)
        """, (grievance_id, rating, comments))

def update_grievance_by_admin(
    grievance_id: str,
    new_status: str,
    admin_remarks: str = ""
) -> bool:
    """Admin updates grievance status and internal remarks. Accurately manages resolved_at."""
    if new_status not in VALID_STATUSES:
        return False
    now = datetime.now().isoformat()
    with db_session() as conn:
        cursor = conn.cursor()
        if new_status == "RESOLVED":
            cursor.execute("""
                UPDATE grievances
                SET status = ?,
                    admin_remarks = ?,
                    updated_at = ?,
                    resolved_at = COALESCE(resolved_at, ?)
                WHERE grievance_id = ?
            """, (new_status, admin_remarks, now, now, grievance_id))
        else:
            cursor.execute("""
                UPDATE grievances
                SET status = ?,
                    admin_remarks = ?,
                    updated_at = ?,
                    resolved_at = CASE WHEN status = 'RESOLVED' THEN NULL ELSE resolved_at END
                WHERE grievance_id = ?
            """, (new_status, admin_remarks, now, grievance_id))
        return cursor.rowcount > 0

def get_admin_dashboard_stats() -> Dict[str, Any]:
    """Return status-level counts for the admin portal KPI cards."""
    with db_session() as conn:
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

        cursor.execute("SELECT COUNT(*) FROM grievances WHERE status != 'DRAFT'")
        submitted = cursor.fetchone()[0]

        return {
            "total": total,
            "total_submitted": submitted,
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
    Excludes internal fields like original_text, normalized_text, and generated_grievance.
    Supports flexible matching (exact token, hyphenless, or ID suffix token).
    """
    cleaned = grievance_id.strip().upper()
    hyphenless = cleaned.replace('-', '')
    if not cleaned:
        return None

    with db_session() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT grievance_id, created_at, updated_at, category, department,
                   urgency, location, duration, summary, status, admin_remarks, resolved_at
            FROM grievances
            WHERE UPPER(grievance_id) = ? 
               OR UPPER(REPLACE(grievance_id, '-', '')) = ?
        """, (cleaned, hyphenless))
        row = cursor.fetchone()
        if row:
            return dict(row)

        # Allow partial token matching only if query contains at least 6 characters (e.g. GRV-101 or 000101)
        if len(cleaned) >= 6:
            cursor.execute("""
                SELECT grievance_id, created_at, updated_at, category, department,
                       urgency, location, duration, summary, status, admin_remarks, resolved_at
                FROM grievances
                WHERE grievance_id LIKE ? OR REPLACE(grievance_id, '-', '') LIKE ?
                ORDER BY id DESC LIMIT 1
            """, (f"%{cleaned}%", f"%{hyphenless}%"))
            row = cursor.fetchone()
            return dict(row) if row else None

        return None
