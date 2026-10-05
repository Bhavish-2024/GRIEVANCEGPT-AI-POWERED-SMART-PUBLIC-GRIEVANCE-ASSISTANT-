"""
Completeness Evaluation Engine for Civic Grievances.
Evaluates whether a citizen's complaint provides sufficient actionable details
to be routed, inspected, and acted upon by municipal authorities.
"""
from typing import Dict, Any, List

def evaluate_completeness(extracted_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate grievance completeness based on required and optional fields.
    
    Required Core Fields (Weights sum to 80%):
    - Problem description: 25%
    - Location / Landmark: 25%
    - Category / Service: 20%
    - Detailed description: 10%
    
    Optional High-Value Fields (Weights sum to 20%):
    - Duration: 10%
    - District / Locality scope: 10%
    
    Returns:
        dict: {
            "score": int (0-100),
            "is_complete": bool (score >= 70 and not critical_missing),
            "missing_fields": list of str,
            "suggestions": list of str,
            "field_status": dict mapping field to bool
        }
    """
    problem = str(extracted_data.get("problem", "")).strip()
    location = str(extracted_data.get("location", "")).strip()
    category = str(extracted_data.get("category", "")).strip()
    service = str(extracted_data.get("affected_service", "")).strip()
    description = str(extracted_data.get("description", "") or extracted_data.get("additional_details", "")).strip()
    duration = str(extracted_data.get("duration", "")).strip()
    district = str(extracted_data.get("district", "")).strip()

    # Vague location placeholders to reject
    vague_locations = {"", "pending confirmation", "unspecified", "area", "my area", "inga", "enga area", "nearby"}
    
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
        suggestions.append("Please specify the exact nature of the issue (e.g. no water supply, broken road, overflowing garbage).")
        field_status["problem"] = False

    # 2. Location Check (25%)
    if location and location.lower() not in vague_locations and len(location) >= 3:
        score += 25
        field_status["location"] = True
    else:
        missing_fields.append("Exact location / Landmark")
        suggestions.append("Please provide the specific street name, ward number, or a nearby landmark.")
        field_status["location"] = False

    # 3. Category / Service Check (20%)
    if (category and category.lower() not in ("unknown", "other", "")) or (service and len(service) > 2):
        score += 20
        field_status["category"] = True
    else:
        missing_fields.append("Affected service / department")
        suggestions.append("Identify which municipal service is affected (e.g. Water Supply, Roads, Electricity, Sanitation).")
        field_status["category"] = False

    # 4. Description Depth (10%)
    if (description and len(description) > 10) or (problem and len(problem) > 20):
        score += 10
        field_status["description"] = True
    else:
        missing_fields.append("Detailed description")
        suggestions.append("Add a brief sentence describing how the issue affects you or your neighborhood.")
        field_status["description"] = False

    # 5. Duration (10% - Optional but valuable)
    if duration and duration.lower() not in ("unspecified", "unknown", "none", ""):
        score += 10
        field_status["duration"] = True
    else:
        missing_fields.append("Duration")
        suggestions.append("Mention how long the issue has persisted (e.g. 4 days, 2 weeks).")
        field_status["duration"] = False

    # 6. District / Locality Scope (10% - Optional)
    if district and len(district) > 2:
        score += 10
        field_status["district"] = True
    else:
        field_status["district"] = False

    # Critical requirement: Both problem and location must be present for a grievance to be considered ready
    critical_missing = not field_status["problem"] or not field_status["location"]
    is_complete = (score >= 70) and not critical_missing

    return {
        "score": min(score, 100),
        "is_complete": is_complete,
        "missing_fields": missing_fields,
        "suggestions": suggestions,
        "field_status": field_status
    }
