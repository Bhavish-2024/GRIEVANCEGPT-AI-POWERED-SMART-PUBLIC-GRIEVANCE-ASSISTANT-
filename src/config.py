"""
Configuration settings for GrievanceGPT.
Reads environment variables with sensible defaults for both local development and Render cloud deployment.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env if present
load_dotenv()

# Directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
VECTOR_STORE_DIR = MODELS_DIR / "vector_store"
KNOWLEDGE_BASE_DIR = BASE_DIR / "knowledge_base"
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

# Ensure runtime directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
KNOWLEDGE_BASE_DIR.mkdir(parents=True, exist_ok=True)

# Datasets
CIVICDEX_CSV = DATA_DIR / "civicdex.csv"
TRAIN_CSV = DATA_DIR / "train.csv"
VAL_CSV = DATA_DIR / "validation.csv"
TEST_CSV = DATA_DIR / "test.csv"

# Model Paths
INTENT_MODEL_PATH = MODELS_DIR / "intent_model.pkl"
CATEGORY_MODEL_PATH = MODELS_DIR / "category_model.pkl"
DEPARTMENT_MODEL_PATH = MODELS_DIR / "department_model.pkl"
URGENCY_MODEL_PATH = MODELS_DIR / "urgency_model.pkl"
RAG_INDEX_PATH = VECTOR_STORE_DIR / "rag_index.pkl"

# Database
DATABASE_PATH = DATA_DIR / "grievance_gpt.db"

# Server Settings
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))
DEBUG = os.getenv("DEBUG", "False").lower() in ("true", "1", "yes")

# Ollama / Local LLM Settings
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:8b")
OLLAMA_FALLBACK_MODEL = os.getenv("OLLAMA_FALLBACK_MODEL", "qwen3:4b")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "qwen3-embedding:0.6b")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", 30))

# Remote / Cloud LLM (Optional for Render)
REMOTE_LLM_URL = os.getenv("REMOTE_LLM_URL", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")

# Admin Portal Authentication
ADMIN_PIN = os.getenv("ADMIN_PIN", "admin@2026")
