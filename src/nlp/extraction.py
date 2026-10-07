"""
Information Extraction Engine for GrievanceGPT.
Extracts structured grievance entities (problem, location, district, duration, category, department, urgency, severity)
using Ollama Qwen3 JSON-mode with a resilient multilingual heuristic fallback parser.
"""
import json
import re
import logging
from typing import Dict, Any, Optional
from src.llm.ollama_client import ollama_client
from src.nlp.preprocessing import detect_language

logger = logging.getLogger("GrievanceGPT.Extraction")

EXTRACTION_SYSTEM_PROMPT = """You are an expert civic intelligence assistant specializing in public grievance parsing.
Your task is to extract structured entities from citizen complaints (in English, Tamil, Tanglish, or Code-mixed).

Output ONLY valid JSON matching this schema:
{
  "problem": "Clear concise summary of the core issue",
  "category": "One of: water_supply, roads, sanitation, drainage, electricity, street_lighting, certificates, public_health, transport, welfare",
  "subcategory": "Specific aspect (e.g., pipeline_leak, potholes, uncollected_garbage)",
  "department": "Corresponding department (e.g., Water Supply Department, Roads Department, Sanitation Department, Electricity Department, Revenue Department, Drainage Department, Public Health Department)",
  "location": "Exact street, area, landmark or locality. If not specified or only 'my area', set to 'pending confirmation'",
  "district": "District or city name if mentioned (e.g. Chennai, Vellore) else 'Not specified'",
  "duration": "Time duration the issue has existed (e.g. 4 days, 2 weeks) else 'Unspecified'",
  "affected_service": "Name of public service affected",
  "urgency": "One of: high, medium, low",
  "severity": "One of: public_hazard, service_failure, service_delay, information_query",
  "additional_details": "Any supporting notes, health risks, or context"
}
"""

# Regex patterns for heuristic extraction (especially Tamil & Tanglish)
DURATION_PATTERNS = [
    re.compile(r'(\d+)\s*(?:days?|naal|naala|நாட்களாக|நாட்களா|நாள்|days\s*ah|days\s*aachu)', re.IGNORECASE),
    re.compile(r'(\d+)\s*(?:weeks?|vaaram|வாரமாக|வாரங்களாக)', re.IGNORECASE),
    re.compile(r'(\d+)\s*(?:months?|maasam|மாதமாக)', re.IGNORECASE),
    re.compile(r'(\d+)\s*(?:hours?|mani|மணி நேரமாக)', re.IGNORECASE),
]

LOCATION_PATTERNS = [
    re.compile(r'\b(?:[0-9]+(?:st|nd|rd|th)?\s+)?(?:[A-Z][a-zA-Z0-9\.-]+\s+)*(?:Nagar|Street|Road|Salai|Colony|Cross|Ward|Village|Town|Station|Bazaar|Puram|Garden|Layout)\b', re.IGNORECASE),
    re.compile(r'(?:near|opp|opposite|at|in|on|near by|பக்கத்துல|அருகில்)\s+([A-Za-z0-9\s,\.-]+?)(?:\s+(?:for|since|last|from|having|water|road|is|was)|\.|$)', re.IGNORECASE),
]

KNOWN_DISTRICTS = ["chennai", "vellore", "coimbatore", "madurai", "salem", "trichy", "tiruchirappalli", "tirunelveli", "kanchipuram", "tiruvallur", "thanjavur", "katpadi"]

def heuristic_extract(text: str) -> Dict[str, Any]:
    """
    Multilingual heuristic extraction fallback when LLM is unavailable or outputs malformed JSON.
    Accurately captures duration, location, and issue type for Tamil, Tanglish, and English text.
    """
    lowered = text.lower()
    
    # 1. Duration extraction
    duration = "Unspecified"
    for pat in DURATION_PATTERNS:
        match = pat.search(text)
        if match:
            num = match.group(1)
            raw = match.group(0).lower()
            if "week" in raw or "vaaram" in raw or "வாரம்" in raw:
                duration = f"{num} weeks"
            elif "month" in raw or "maasam" in raw or "மாதம்" in raw:
                duration = f"{num} months"
            elif "hour" in raw or "mani" in raw or "மணி" in raw:
                duration = f"{num} hours"
            else:
                duration = f"{num} days"
            break

    # 2. Location extraction
    location = "pending confirmation"
    # Check for known districts / landmarks
    found_locs = []
    # Pattern 1: Find all named streets/areas/roads
    for match in LOCATION_PATTERNS[0].finditer(text):
        cand = match.group(0).strip()
        if len(cand) > 3 and cand.lower() not in ("my area", "enga area", "our area", "area", "this issue"):
            found_locs.append(cand)
            
    # Pattern 2: Preposition-based landmark
    match_prep = LOCATION_PATTERNS[1].search(text)
    if match_prep and not found_locs:
        cand = match_prep.group(1).strip()
        if len(cand) > 3 and cand.lower() not in ("my area", "enga area", "our area", "area", "this issue"):
            found_locs.append(cand)
                
    for d in KNOWN_DISTRICTS:
        if d in lowered:
            found_locs.append(d.title())
            
    if found_locs:
        location = ", ".join(dict.fromkeys(found_locs))

    # 3. District detection
    district = "Not specified"
    for d in KNOWN_DISTRICTS:
        if d in lowered:
            district = d.title()
            break

    # 4. Problem & Category heuristics
    has_water = any(k in text or k in lowered for k in ["water", "தண்ணீர்", "குடிநீர்", "thanni", "pipeline", "tap"])
    has_elec = any(k in text or k in lowered for k in ["electricity", "power", "current", "voltage", "transformer", "electric", "elctric"])
    has_road = any(k in text or k in lowered for k in ["road", "pothole", "சாலை", "குழி", "kuzhi", "tar road"])
    has_light = any(k in text or k in lowered for k in ["light", "விளக்கு", "streetlight", "street light", "bulb"])
    has_garbage = any(k in text or k in lowered for k in ["garbage", "waste", "குப்பை", "kuppai", "dustbin", "bin"])
    has_drainage = any(k in text or k in lowered for k in ["drain", "sewer", "வடிகால்", "drainage", "saakadai", "சாக்கடை"])

    if has_water and has_elec:
        category = "water_supply & electricity"
        department = "Water Supply & Electricity Board"
        problem = "Drinking water supply disruption AND electrical power outage"
        affected_service = "Potable Water Supply & Electrical Power Grid"
        urgency = "high"
        severity = "service_failure"
    elif has_road and has_light:
        category = "roads & street_lighting"
        department = "Roads & Street Lighting Department"
        problem = "Road damage/potholes AND defective street lighting"
        affected_service = "Road Infrastructure & Street Lighting"
        urgency = "high"
        severity = "public_hazard"
    elif has_water and has_drainage:
        category = "water_supply & drainage"
        department = "Water Supply & Sewerage Board"
        problem = "Drinking water supply disruption AND underground drainage blockage"
        affected_service = "Potable Water Supply & Drainage Network"
        urgency = "high"
        severity = "service_failure"
    elif has_water:
        category = "water_supply"
        department = "Water Supply Department"
        problem = "Drinking water supply disruption or shortage"
        affected_service = "Potable Drinking Water Supply"
        urgency = "high"
        severity = "service_failure"
    elif has_elec:
        category = "electricity"
        department = "Electricity Board (TANGEDCO)"
        problem = "Frequent power cuts or voltage fluctuation"
        affected_service = "Electrical Power Supply"
        urgency = "high"
        severity = "service_failure"
    elif has_road:
        category = "road_maintenance"
        department = "Roads & Infrastructure Department"
        problem = "Road damage and hazardous potholes"
        affected_service = "Road Infrastructure & Safety"
        urgency = "high"
        severity = "public_hazard"
    elif has_garbage:
        category = "garbage_collection"
        department = "Sanitation & Waste Management"
        problem = "Uncollected garbage accumulation"
        affected_service = "Solid Waste Conservancy"
        urgency = "medium"
        severity = "service_delay"
    elif has_drainage:
        category = "drainage_sewage"
        department = "Sanitation & Waste Management"
        problem = "Blocked drainage and sewage overflow"
        affected_service = "Underground Drainage System"
        urgency = "high"
        severity = "public_hazard"
    elif has_light:
        category = "street_lighting"
        department = "Civic Maintenance & Street Lighting"
        problem = "Defective non-functional street lights"
        affected_service = "Street Lighting Maintenance"
        urgency = "medium"
        severity = "service_delay"
    elif any(k in text or k in lowered for k in ["ration", "card", "certificate", "சான்றிதழ்", "patta"]):
        category = "certificates"
        department = "Revenue Department"
        problem = "Certificate or citizen document status"
        affected_service = "Revenue & Civil Supplies"
        urgency = "low"
        severity = "information_query"
    else:
        category = "civic_maintenance"
        department = "Municipal Administration"
        problem = text[:80] + ("..." if len(text) > 80 else "")
        affected_service = "Municipal Services"
        urgency = "medium"
        severity = "service_delay"

    return {
        "problem": problem,
        "category": category,
        "subcategory": "general_issue",
        "department": department,
        "location": location,
        "district": district,
        "duration": duration,
        "affected_service": affected_service,
        "urgency": urgency,
        "severity": severity,
        "additional_details": f"Processed via GrievanceGPT multilingual engine. Language: {detect_language(text)}"
    }

def extract_grievance_entities(text: str) -> Dict[str, Any]:
    """
    Extract structured grievance entities using Ollama with JSON parsing,
    falling back to robust heuristic analysis on failure or offline status.
    """
    if not text or not text.strip():
        return heuristic_extract("")

    # If Ollama is connected, attempt LLM extraction
    if ollama_client.check_connection().get("connected"):
        user_prompt = f"Extract structured grievance entities from the following citizen input:\n\"\"\"{text}\"\"\""
        raw_response = ollama_client.generate(user_prompt, system=EXTRACTION_SYSTEM_PROMPT, json_format=True)
        
        # Try parsing JSON from LLM output
        try:
            # Strip markdown code fences if model enclosed them
            cleaned = re.sub(r'^```json\s*', '', raw_response, flags=re.MULTILINE)
            cleaned = re.sub(r'```$', '', cleaned, flags=re.MULTILINE).strip()
            parsed = json.loads(cleaned)
            
            # Validate required fields
            expected_keys = ["problem", "category", "department", "location", "duration", "urgency"]
            if all(k in parsed for k in expected_keys):
                return parsed
        except Exception as e:
            logger.warning(f"Failed to parse LLM JSON extraction: {e}. Falling back to heuristic extractor.")

    # Fallback to intelligent multilingual heuristic extractor
    return heuristic_extract(text)
