"""
Structured Grievance Draft Generator.
Formats formal, actionable civic grievance documents ready for citizen review, editing, and submission.
"""
from typing import Dict, Any

def generate_structured_grievance(
    extracted_data: Dict[str, Any],
    classification: Dict[str, Any],
    grievance_id: str
) -> Dict[str, Any]:
    """
    Generate a formal structured civic grievance document.
    """
    category = classification.get("category", extracted_data.get("category", "civic_maintenance")).replace("_", " ").title()
    department = classification.get("department", extracted_data.get("department", "Municipal Administration"))
    urgency = classification.get("urgency", extracted_data.get("urgency", "medium")).capitalize()
    severity = classification.get("severity", extracted_data.get("severity", "service_issue")).replace("_", " ").title()
    
    problem = extracted_data.get("problem", "Civic Grievance")
    location = extracted_data.get("location", "Not specified")
    district = extracted_data.get("district", "Not specified")
    duration = extracted_data.get("duration", "Ongoing")
    details = extracted_data.get("additional_details", "")
    
    # Construct professional subject line
    subject = f"Urgent Grievance: {problem} at {location}"
    if len(subject) > 90:
        subject = f"Grievance: {problem[:50]}... at {location}"

    # Construct formal description
    description_lines = [
        f"To: The Executive Officer / Grievance Redressal Cell",
        f"Department: {department}",
        f"Reference Token: {grievance_id}",
        "",
        f"Respected Sir/Madam,",
        "",
        f"I am submitting this formal grievance regarding {problem.lower()} in our locality.",
        f"Location Details: {location}" + (f", District: {district}" if district and district != "Not specified" else ""),
        f"Duration of Issue: Persisting for {duration}." if duration else "",
        f"Public Impact / Severity: {severity}.",
    ]
    
    if details:
        description_lines.append(f"Additional Observations: {details}")
        
    description_lines.extend([
        "",
        f"This disruption requires immediate intervention by the {department} to prevent further inconvenience and potential safety/health hazards to residents.",
        "",
        f"Priority Level: {urgency}",
        f"Status: Prototype Submission (Pending Citizen Verification)",
        f"Date: Generated via GrievanceGPT Citizen Redressal Portal"
    ])
    
    formatted_body = "\n".join(description_lines)

    return {
        "grievance_id": grievance_id,
        "subject": subject,
        "department": department,
        "category": category,
        "urgency": urgency,
        "severity": severity,
        "location": location,
        "district": district,
        "duration": duration,
        "description": formatted_body,
        "raw_problem": problem
    }
