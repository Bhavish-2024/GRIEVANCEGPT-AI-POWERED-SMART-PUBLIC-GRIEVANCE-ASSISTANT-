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
from src.nlp.intent import validate_intent
from src.rag.qa import answer_civic_question

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
    1. Intent Validation (Gateway: Greeting, Ambiguous, Civic Question, Complaint, Irrelevant)
    2. RAG Retrieval & Grounded Answering (for Civic Questions)
    3. ML Classification & Extraction (for Genuine Civic Complaints)
    4. Completeness Evaluation & Follow-up Questioning
    5. Structured Grievance Draft Generation (when complete)
    """
    # 1. Intent Validation Gate
    intent_info = validate_intent(
        text=text,
        accumulated_data=accumulated_data,
        session_history=session_history
    )
    detected_intent = intent_info["intent"]

    # 1A. GREETING / EMPTY INPUT HANDLER: Friendly response without classification or RAG retrieval
    if detected_intent == "greeting":
        lang = intent_info.get("language", "english")
        if intent_info.get("reason") == "empty_input":
            reply = (
                "Please enter your message or describe the civic issue you are experiencing.\n\n"
                "You can report problems such as drinking water supply disruptions, damaged roads/potholes, "
                "uncollected garbage, or non-functioning streetlights, or ask questions about municipal procedures."
            )
        else:
            if lang in ("tamil", "tanglish", "code_mixed") or any(w in text.lower() for w in ["vanakkam", "வணக்கம்"]):
                reply = (
                    "வணக்கம்! Vanakkam! GrievanceGPT-க்கு வரவேற்கிறோம். உங்களுக்கு எப்படி உதவ முடியும்?\n\n"
                    "குடிநீர் விநியோகத் தடை, சாலை பள்ளங்கள், குப்பை மேடு, அல்லது தெருவிளக்கு பிரச்சனை போன்ற பொதுக் குறைகளை "
                    "பதிவு செய்யலாம், அல்லது நகராட்சி சேவைகள் பற்றிய கேள்விகளை கேட்கலாம்."
                )
            else:
                reply = (
                    "Hello! Vanakkam! Welcome to GrievanceGPT, your AI-powered smart public grievance assistant.\n\n"
                    "How can I assist you today? You can:\n"
                    "• **Report a Civic Issue**: Describe problems like drinking water supply disruptions, damaged roads/potholes, uncollected garbage, or streetlights.\n"
                    "• **Ask Civic Questions**: Inquire about municipal departments, grievance procedures, or civic guidelines."
                )
        return {
            "text": text,
            "classification": {
                "intent": "greeting",
                "intent_confidence": intent_info.get("confidence", 0.99),
                "category": None,
                "category_confidence": 0.0,
                "department": None,
                "department_confidence": 0.0,
                "urgency": None,
                "urgency_confidence": 0.0,
                "language": lang,
                "normalized_text": intent_info.get("normalized_text", text),
                "model_source": "intent_validator"
            },
            "extracted_entities": {},
            "completeness": {"score": 0, "is_complete": False, "missing_fields": [], "field_status": {}},
            "workflow_stage": "GREETING",
            "assistant_reply": reply,
            "structured_grievance": None,
            "reference_id": None
        }

    # 1B. AMBIGUOUS INPUT HANDLER: Ask clarification question before complaint retrieval
    if detected_intent == "ambiguous":
        lang = intent_info.get("language", "english")
        reply = (
            "I would be glad to help you resolve this issue. Could you please specify what kind of civic problem you are experiencing?\n\n"
            "For example:\n"
            "• **Drinking Water**: Pipeline rupture, contamination, or supply interruption\n"
            "• **Roads & Pavements**: Hazardous potholes, broken asphalt, or road cave-ins\n"
            "• **Street Lighting**: Non-functioning streetlights or dark streets\n"
            "• **Sanitation**: Uncollected garbage heaps or overflowing dustbins\n"
            "• **Drainage / Sewage**: Overflowing manholes or blocked drains\n"
            "• **Electricity**: Power cuts, voltage fluctuations, or hanging live wires"
        )
        return {
            "text": text,
            "classification": {
                "intent": "ambiguous",
                "intent_confidence": intent_info.get("confidence", 0.90),
                "category": None,
                "category_confidence": 0.0,
                "department": None,
                "department_confidence": 0.0,
                "urgency": None,
                "urgency_confidence": 0.0,
                "language": lang,
                "normalized_text": intent_info.get("normalized_text", text),
                "model_source": "intent_validator"
            },
            "extracted_entities": {},
            "completeness": {"score": 0, "is_complete": False, "missing_fields": [], "field_status": {}},
            "workflow_stage": "CLARIFICATION",
            "assistant_reply": reply,
            "structured_grievance": None,
            "reference_id": None
        }

    # 1C. CIVIC QUESTION HANDLER: RAG retrieval over authoritative knowledge base
    if detected_intent == "civic_question":
        # Pass user's actual question preserving context as retrieval query
        rag_res = answer_civic_question(query=text)
        return {
            "text": text,
            "classification": {
                "intent": "civic_question",
                "intent_confidence": intent_info.get("confidence", 0.92),
                "category": "civic_information",
                "category_confidence": 0.92,
                "department": None,
                "department_confidence": 0.0,
                "urgency": "low",
                "urgency_confidence": 0.90,
                "language": intent_info.get("language", "english"),
                "normalized_text": intent_info.get("normalized_text", text),
                "model_source": "rag_knowledge_base"
            },
            "extracted_entities": {},
            "completeness": {"score": 100, "is_complete": False, "missing_fields": [], "field_status": {}},
            "workflow_stage": "CIVIC_INFO",
            "assistant_reply": rag_res["answer"],
            "structured_grievance": None,
            "reference_id": None,
            "rag": rag_res
        }

    # 1D. IRRELEVANT / OUT-OF-DOMAIN HANDLER
    if detected_intent == "irrelevant":
        reply = (
            "I am GrievanceGPT, an AI assistant dedicated to public municipal civic issues and services "
            "(such as water supply, roads, sanitation, streetlights, and drainage).\n\n"
            "Please describe a civic problem you would like to report, or ask a question regarding municipal procedures and guidelines."
        )
        return {
            "text": text,
            "classification": {
                "intent": "irrelevant",
                "intent_confidence": intent_info.get("confidence", 0.85),
                "category": None,
                "category_confidence": 0.0,
                "department": None,
                "department_confidence": 0.0,
                "urgency": None,
                "urgency_confidence": 0.0,
                "language": intent_info.get("language", "english"),
                "normalized_text": intent_info.get("normalized_text", text),
                "model_source": "intent_validator"
            },
            "extracted_entities": {},
            "completeness": {"score": 0, "is_complete": False, "missing_fields": [], "field_status": {}},
            "workflow_stage": "IRRELEVANT",
            "assistant_reply": reply,
            "structured_grievance": None,
            "reference_id": None
        }

    # 2. GENUINE CIVIC COMPLAINT PIPELINE
    # ML Classification (Intent, Category, Department, Urgency)
    classification = classifier.predict(text)
    
    # Entity Extraction
    extracted = extract_grievance_entities(text)

    
    vague_locations_set = {
        "this place", "this area", "here", "my area", "inga", "enga area", "our area",
        "nearby", "same place", "same area", "place", "area", "this issue",
        "not specified", "unspecified", "pending confirmation"
    }

    # Maintain existing reference token if present in session history
    temp_id = (accumulated_data.get("reference_id") if accumulated_data else None) or generate_grievance_id()

    # Merge with accumulated data from multi-turn conversation if present
    if accumulated_data:
        prior_problem = accumulated_data.get("problem")
        prior_category = accumulated_data.get("category")
        prior_department = accumulated_data.get("department")
        prior_urgency = accumulated_data.get("urgency")

        # Check if the new message is primarily providing location, district, or duration follow-up
        new_loc = extracted.get("location")
        is_new_location = bool(new_loc and new_loc.lower() not in vague_locations_set)
        
        is_followup_info = (
            is_new_location or
            extracted.get("duration") != "Unspecified" or 
            extracted.get("district") != "Not specified" or
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
            if k == "location":
                if is_new_location:
                    extracted["location"] = new_loc
                elif v and str(v).lower() not in vague_locations_set:
                    extracted["location"] = v
            else:
                if (not curr_val or curr_val in ("Unspecified", "Not specified")) and v:
                    extracted[k] = v
                elif curr_val and curr_val not in ("Unspecified", "Not specified"):
                    extracted[k] = curr_val

    # Evaluate completeness
    completeness = evaluate_completeness(extracted)

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
