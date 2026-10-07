"""
Completeness Evaluation Engine for Civic Grievances.
Evaluates whether a citizen's complaint provides sufficient actionable details
to be routed, inspected, and acted upon by municipal authorities.
"""
from typing import Dict, Any, List

def evaluate_completeness(extracted_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate grievance completeness based on required fields per user policy:
    - Problem description (25%)
    - Location / Street Name (25%)
    - District / City (25%)
    - Duration of Issue (25%)
    
    All 4 fields MUST be provided before a grievance can be filed.
    """
    problem = str(extracted_data.get("problem", "")).strip()
    location = str(extracted_data.get("location", "")).strip()
    district = str(extracted_data.get("district", "")).strip()
    duration = str(extracted_data.get("duration", "")).strip()

    vague_locations = {
        "", "pending confirmation", "unspecified", "area", "my area", "inga",
        "enga area", "nearby", "not specified", "this place", "this area",
        "here", "same place", "same area", "our area", "place"
    }
    vague_districts = {"", "not specified", "unspecified", "unknown", "none", "pending"}
    vague_durations = {"", "unspecified", "unknown", "none", "pending", "not specified"}

    score = 0
    missing_fields = []
    suggestions = []
    field_status = {}

    # 1. Problem Check (25%)
    if problem and len(problem) > 3 and problem.lower() not in ("issue", "problem", "none"):
        score += 25
        field_status["problem"] = True
    else:
        missing_fields.append("Problem description")
        suggestions.append("Please specify the exact nature of the issue (e.g. no water supply, broken road, power outage).")
        field_status["problem"] = False

    # 2. Location Check (25%)
    if location and location.lower() not in vague_locations and len(location) >= 3:
        score += 25
        field_status["location"] = True
    else:
        missing_fields.append("Exact location / Landmark")
        suggestions.append("Please provide the specific street name, ward number, or a nearby landmark.")
        field_status["location"] = False

    # 3. District Check (25%)
    if district and district.lower() not in vague_districts and len(district) >= 3:
        score += 25
        field_status["district"] = True
    else:
        missing_fields.append("District (e.g. Chennai, Vellore, Coimbatore)")
        suggestions.append("Specify the district or city name (e.g. Chennai, Vellore, Coimbatore, Madurai).")
        field_status["district"] = False

    # 4. Duration Check (25%)
    if duration and duration.lower() not in vague_durations:
        score += 25
        field_status["duration"] = True
    else:
        missing_fields.append("Duration (e.g. 4 days, 2 weeks)")
        suggestions.append("Mention how long the issue has persisted (e.g. 4 days, 2 weeks).")
        field_status["duration"] = False

    # Mandatory rule: Problem, Location, District, AND Duration MUST all be present
    critical_missing = (
        not field_status["problem"] or
        not field_status["location"] or
        not field_status["district"] or
        not field_status["duration"]
    )
    is_complete = not critical_missing and (score >= 100)

    return {
        "score": score,
        "is_complete": is_complete,
        "missing_fields": missing_fields,
        "suggestions": suggestions,
        "field_status": field_status
    }
