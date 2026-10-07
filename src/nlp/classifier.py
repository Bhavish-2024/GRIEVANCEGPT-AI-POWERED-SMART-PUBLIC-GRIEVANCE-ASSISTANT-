"""
Classifier interface for GrievanceGPT.
Loads trained CivicDex machine learning models from disk and performs
real-time inference for Intent, Category, Department, and Urgency.
"""
import joblib
import logging
from typing import Dict, Any, Optional
from src.config import (
    INTENT_MODEL_PATH,
    CATEGORY_MODEL_PATH,
    DEPARTMENT_MODEL_PATH,
    URGENCY_MODEL_PATH
)
from src.nlp.preprocessing import preprocess_for_classification, detect_language, normalize_text

logger = logging.getLogger("GrievanceGPT.Classifier")

def synchronize_classification(text: str, category: str, department: str, urgency: str) -> Dict[str, str]:
    """Ensure category and department never contradict each other and detect compound multi-utility complaints."""
    lowered = text.lower()
    has_water = any(w in lowered for w in ["water", "தண்ணீர்", "குடிநீர்", "thanni", "pipeline", "tap"])
    has_elec = any(w in lowered for w in ["electricity", "power", "current", "voltage", "transformer", "electric", "elctric"])
    has_road = any(w in lowered for w in ["road", "pothole", "சாலை", "குழி", "kuzhi"])
    has_light = any(w in lowered for w in ["light", "streetlight", "விளக்கு"])
    has_garbage = any(w in lowered for w in ["garbage", "kuppai", "குப்பை", "waste"])
    has_drainage = any(w in lowered for w in ["drain", "sewage", "saakadai", "சாக்கடை"])

    # 1. Multi-utility compound complaints
    if has_water and has_elec:
        return {
            "category": "water_supply & electricity",
            "department": "Water Supply & Electricity Board",
            "urgency": "high"
        }
    if has_road and has_light:
        return {
            "category": "roads & street_lighting",
            "department": "Roads & Street Lighting Department",
            "urgency": urgency or "high"
        }
    if has_water and has_drainage:
        return {
            "category": "water_supply & drainage",
            "department": "Water Supply & Sewerage Board",
            "urgency": "high"
        }

    # 2. Single-utility keyword alignment
    if has_elec and not has_water:
        return {
            "category": "electricity",
            "department": "Electricity Board (TANGEDCO)",
            "urgency": urgency or "high"
        }
    if has_water and not has_elec:
        return {
            "category": "water_supply",
            "department": "Water Supply Department",
            "urgency": urgency or "high"
        }
    if has_road:
        return {
            "category": "road_maintenance",
            "department": "Roads & Infrastructure Department",
            "urgency": urgency or "high"
        }
    if has_garbage:
        return {
            "category": "garbage_collection",
            "department": "Sanitation & Waste Management",
            "urgency": urgency or "medium"
        }
    if has_drainage:
        return {
            "category": "drainage_sewage",
            "department": "Sanitation & Waste Management",
            "urgency": urgency or "high"
        }
    if has_light:
        return {
            "category": "street_lighting",
            "department": "Civic Maintenance & Street Lighting",
            "urgency": urgency or "medium"
        }

    # 3. Synchronize department to category if no keyword matched
    if department in ("Water Supply Department", "Water Board"):
        category = "water_supply"
    elif department in ("Electricity Department", "Electricity Board (TANGEDCO)"):
        category = "electricity"
    elif department in ("Roads Department", "Roads & Infrastructure Department"):
        category = "road_maintenance"
    elif department in ("Sanitation Department", "Sanitation & Waste Management"):
        category = "garbage_collection"
    elif department in ("Drainage Department", "Drainage & Sewerage Board"):
        category = "drainage_sewage"

    return {
        "category": category,
        "department": department,
        "urgency": urgency
    }


class GrievanceClassifier:
    def __init__(self):
        self.intent_model = None
        self.category_model = None
        self.department_model = None
        self.urgency_model = None
        self._load_models()

    def _load_models(self):
        """Load trained models from disk if available."""
        try:
            if INTENT_MODEL_PATH.exists():
                self.intent_model = joblib.load(INTENT_MODEL_PATH)
            if CATEGORY_MODEL_PATH.exists():
                self.category_model = joblib.load(CATEGORY_MODEL_PATH)
            if DEPARTMENT_MODEL_PATH.exists():
                self.department_model = joblib.load(DEPARTMENT_MODEL_PATH)
            if URGENCY_MODEL_PATH.exists():
                self.urgency_model = joblib.load(URGENCY_MODEL_PATH)
        except Exception as e:
            logger.warning(f"Error loading trained models: {e}. Will use heuristic fallback.")

    @property
    def is_ready(self) -> bool:
        """Check if all 4 ML models are loaded."""
        return all([
            self.intent_model is not None,
            self.category_model is not None,
            self.department_model is not None,
            self.urgency_model is not None
        ])

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Classify input text using trained CivicDex models with probability scores.
        """
        norm_text = normalize_text(text)
        prep_text = preprocess_for_classification(text)
        language = detect_language(text)

        # Fallback if models are not yet trained
        if not self.is_ready:
            self._load_models()
            if not self.is_ready:
                return self._heuristic_predict(text, language)

        try:
            # Intent
            intent = self.intent_model.predict([prep_text])[0]
            intent_proba = max(self.intent_model.predict_proba([prep_text])[0])

            # Category
            category = self.category_model.predict([prep_text])[0]
            cat_proba = max(self.category_model.predict_proba([prep_text])[0])

            # Department
            department = self.department_model.predict([prep_text])[0]
            dept_proba = max(self.department_model.predict_proba([prep_text])[0])

            # Urgency
            urgency = self.urgency_model.predict([prep_text])[0]
            urg_proba = max(self.urgency_model.predict_proba([prep_text])[0])

            # Synchronize category and department to prevent contradictions
            synced = synchronize_classification(text, category, department, urgency)

            return {
                "intent": intent,
                "intent_confidence": round(float(intent_proba), 4),
                "category": synced["category"],
                "category_confidence": round(float(cat_proba), 4),
                "department": synced["department"],
                "department_confidence": round(float(dept_proba), 4),
                "urgency": synced["urgency"],
                "urgency_confidence": round(float(urg_proba), 4),
                "language": language,
                "normalized_text": norm_text,
                "model_source": "civicdex_ml"
            }
        except Exception as e:
            logger.error(f"Prediction error: {e}. Using fallback classifier.")
            return self._heuristic_predict(text, language)

    def _heuristic_predict(self, text: str, language: str) -> Dict[str, Any]:
        """Heuristic fallback classifier if ML pipelines are missing or encountered error."""
        synced = synchronize_classification(text, "civic_maintenance", "Municipal Administration", "high")

        return {
            "intent": "complaint",
            "intent_confidence": 0.85,
            "category": synced["category"],
            "category_confidence": 0.90,
            "department": synced["department"],
            "department_confidence": 0.90,
            "urgency": synced["urgency"],
            "urgency_confidence": 0.85,
            "language": language,
            "normalized_text": normalize_text(text),
            "model_source": "heuristic_fallback"
        }

# Global singleton
classifier = GrievanceClassifier()
