"""
Grievance Analysis Orchestrator.
Coordinates language understanding, ML classification, entity extraction,
completeness evaluation, follow-up questioning, and grievance draft generation.
"""
from typing import Dict, Any, List, Optional
from src.nlp.classifier import classifier
from src.nlp.extraction import extract_grievance_entities
from src.grievance.completeness import evaluate_completeness
from src.grievance.generator import generate_structured_grievance
from src.database import generate_grievance_id
from src.llm.ollama_client import ollama_client

CONVERSATIONAL_AGENT_SYSTEM = """You are GrievanceGPT, an AI-powered smart public grievance assistant.
Your goal is to help citizens prepare complete, verified, and correctly categorized public complaints.
Rules:
1. Be empathetic, polite, and professional.
2. If important details (especially location, duration, or exact issue) are missing, ask targeted, helpful questions.
3. If the citizen wrote in Tamil or Tanglish, acknowledge and respond appropriately.
4. Do NOT immediately generate a final grievance if critical location or problem information is missing.
5. When enough details are collected, inform the citizen that the formal grievance is ready for review.
"""

def analyze_grievance_text(
    text: str,
    accumulated_data: Optional[Dict[str, Any]] = None,
    session_history: Optional[List[Dict[str, str]]] = None
) -> Dict[str, Any]:
    """
    Full pipeline analysis of citizen message:
    1. ML Classification (Intent, Category, Department, Urgency)
    2. Information Extraction
    3. Completeness Checking
    4. Follow-up / Assistant Response Generation
    5. Structured Grievance Draft (if ready)
    """
    # ML Classification
    classification = classifier.predict(text)
    
    # Entity Extraction
    extracted = extract_grievance_entities(text)
    
    # Merge with accumulated data from multi-turn conversation if present
    if accumulated_data:
        # If user previously stated a problem and is now providing follow-up info, retain prior core problem & classification
        prior_problem = accumulated_data.get("problem")
        prior_category = accumulated_data.get("category")
        prior_department = accumulated_data.get("department")
        prior_urgency = accumulated_data.get("urgency")

        # Check if the new message is primarily providing location or duration rather than a whole new complaint
        is_followup_info = (
            extracted.get("location") != "pending confirmation" or 
            extracted.get("duration") != "Unspecified" or 
            len(text.split()) <= 10
        )
        
        if prior_problem and prior_problem not in ("Public grievance issue", "None") and is_followup_info:
            extracted["problem"] = prior_problem
            if prior_category:
                extracted["category"] = prior_category
                classification["category"] = prior_category
            if prior_department:
                extracted["department"] = prior_department
                classification["department"] = prior_department
            if prior_urgency:
                extracted["urgency"] = prior_urgency
                classification["urgency"] = prior_urgency

        for k, v in accumulated_data.items():
            curr_val = extracted.get(k)
            if (not curr_val or curr_val in ("pending confirmation", "Unspecified", "Not specified")) and v:
                extracted[k] = v
            elif k == "location" and curr_val and curr_val != "pending confirmation":
                extracted[k] = curr_val
                
    # Evaluate completeness
    completeness = evaluate_completeness(extracted)
    
    # Temporary or pending ID
    temp_id = generate_grievance_id()

    # Determine workflow stage and generate assistant response
    if not completeness["is_complete"]:
        stage = "INFORMATION_COLLECTION"
        missing_items = ", ".join(completeness["missing_fields"])
        
        # Formulate assistant question
        if ollama_client.check_connection().get("connected"):
            history_prompt = ""
            if session_history:
                history_prompt = "\n".join([f"{m['role']}: {m['content']}" for m in session_history[-4:]])
            prompt = (
                f"Conversation so far:\n{history_prompt}\n\n"
                f"Citizen latest input: \"{text}\"\n"
                f"Missing critical fields: {missing_items}\n"
                f"Extracted info: {extracted}\n\n"
                f"Respond conversationally to the citizen acknowledging their issue ({extracted.get('problem')}) "
                f"and politely asking for the missing information: {missing_items}."
            )
            assistant_reply = ollama_client.generate(prompt, system=CONVERSATIONAL_AGENT_SYSTEM)
        else:
            # High-fidelity conversational fallback
            prob = extracted.get('problem', 'your issue')
            loc_missing = not completeness['field_status'].get('location', False)
            dist_missing = not completeness['field_status'].get('district', False)
            dur_missing = not completeness['field_status'].get('duration', False)
            
            questions = []
            if loc_missing:
                questions.append("• **Location / Street**: Which specific street, ward, or locality is facing this issue? (A landmark is very helpful)")
            if dist_missing:
                questions.append("• **District**: Which district or city is this located in (e.g., Chennai, Vellore, Coimbatore, Madurai)?")
            if dur_missing:
                questions.append("• **Duration**: How long has this problem persisted (e.g., 4 days, 2 weeks)?")
            
            q_list = "\n".join(questions)
            assistant_reply = (
                f"I understand your grievance regarding **{prob}**.\n\n"
                f"To file an official, actionable complaint for the **{classification['department']}**, "
                f"please provide the missing required details:\n\n{q_list}"
            )
            
        structured_grievance = None
    else:
        stage = "REVIEW"
        structured_grievance = generate_structured_grievance(extracted, classification, temp_id)
        
        # Auto-save DRAFT grievance to SQLite immediately so reference ID is persistent and trackable
        try:
            from src.database import save_grievance
            gen_body = structured_grievance.get("description", "") if isinstance(structured_grievance, dict) else str(structured_grievance)
            gen_subject = structured_grievance.get("subject", "Public Grievance") if isinstance(structured_grievance, dict) else "Public Grievance"
            save_grievance(
                original_text=text,
                normalized_text=text,
                category=classification.get("category", "general"),
                department=classification.get("department", "General Administration"),
                urgency=classification.get("urgency", "medium"),
                severity=extracted.get("severity", "service_issue"),
                location=extracted.get("location", ""),
                district=extracted.get("district", ""),
                duration=extracted.get("duration", ""),
                summary=gen_subject,
                generated_grievance=gen_body,
                status="DRAFT",
                grievance_id=temp_id
            )
        except Exception as err:
            print(f"[Auto-save DRAFT warning]: {err}")

        assistant_reply = (
            f"Thank you for providing the required details! I have verified your grievance and generated a "
            f"formal structured complaint for the **{classification['department']}** (Priority: **{classification['urgency'].title()}**).\n\n"
            f"Your Grievance Reference ID is **{temp_id}**.\n\n"
            f"Please review the details in the review panel. You may edit, regenerate, or confirm the submission."
        )

    return {
        "text": text,
        "classification": classification,
        "extracted_entities": extracted,
        "completeness": completeness,
        "workflow_stage": stage,
        "assistant_reply": assistant_reply,
        "structured_grievance": structured_grievance,
        "reference_id": temp_id
    }
