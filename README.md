# GrievanceGPT – AI-Powered Smart Public Grievance Assistant

**GrievanceGPT** is an intelligent public grievance intake, analysis, and preparation platform. It functions as an accessible interface between citizens and public authorities, combining machine learning classifiers trained on the **CivicDex** multilingual dataset with an open-source local Large Language Model (**Ollama Qwen3**) and **Grounded RAG** knowledge retrieval.

Designed for modern deployment on **Render** (free/starter web service) as well as local environments, the platform features a responsive **Government-Grade Web Dashboard** powered by **FastAPI**, **Tailwind CSS**, and **Chart.js**, operating at a minimal memory footprint (~40MB RAM) without any dependency on paid cloud LLMs (no Gemini API, no OpenAI API).

---

## 🏛️ Project Architecture

```
CITIZEN (English / தமிழ் / Tanglish / Code-Mixed)
   │
   ▼
FastAPI Web Dashboard (Uvicorn / Render-Ready)
   │
   ├───────────────────────────────┬───────────────────────────────┐
   ▼                               ▼                               ▼
[Citizen Intake Dialogue]    [Completeness Reviewer]      [Civic FAQ (RAG)]
   │                               │                               │
   ▼                               ▼                               ▼
Multilingual Preprocessing   Civic Dexterity Auditor      TF-IDF / Vector Store
   │                               │                               │
   ├───────────────────────────────┤                               ▼
   ▼                               ▼                      Grounded Q&A (Qwen3)
CivicDex ML Classifiers     Completeness Scoring                   │
 ├── Intent Classifier       (0% - 100% Score)                     ▼
 ├── Category Classifier     Missing Field Alerts        Grounded Source Citations
 ├── Department Classifier   Targeted Follow-Ups         [e.g., water_supply.txt]
 └── Urgency / Priority            │
   │                               │
   └───────────────┬───────────────┘
                   ▼
       Structured Grievance Draft
       (Subject, Formal Letter, Locality, Token: GRV-2026-XXXXXX)
                   │
                   ▼
       Citizen Review & Editing [Edit] [Regenerate] [Confirm]
                   │
                   ▼
         SQLite Grievance Registry (CONFIRMED)
```

---

## ✨ Core Features

1. **Natural Multilingual & Code-Mixed Understanding**:
   - Seamlessly processes citizen complaints submitted in **English**, **Tamil (தமிழ்)**, **Tanglish** (Tamil in Latin script like *"enga area la water varala"*), and **Code-mixed** vernacular expressions.

2. **CivicDex Baseline ML Classifiers**:
   - Classical machine learning pipelines using character-level `(2, 5)` and word-level `(1, 2)` TF-IDF vectorization with balanced Logistic Regression:
     - **Intent Classification** (96.7% accuracy / 97.8% weighted F1).
     - **Department Classification** (83.3% accuracy / 81.0% weighted F1 across 9 departments).
     - **Category Classification** (76.7% accuracy / 74.6% weighted F1 across 10 categories).
     - **Urgency / Priority Estimation** (High / Medium / Low).

3. **Intelligent Missing-Information Detection**:
   - Evaluates complaints against core criteria (Problem 25%, Location 25%, Service/Department 20%, Description 10%, Duration 10%, Scope 10%).
   - If crucial details (like location or duration) are missing, the assistant asks targeted follow-up questions before generating the grievance.

4. **Structured Grievance Draft Generation**:
   - Formulates official grievance letters with subject, department routing, severity rating, and unique tracking token (`GRV-2026-XXXXXX`).
   - Allows citizens to **edit**, **regenerate**, and **confirm** their grievance before final submission.

5. **Grounded RAG Knowledge Base**:
   - 8 civic knowledge guides (`water_supply.txt`, `roads.txt`, `sanitation.txt`, `electricity.txt`, `public_health.txt`, `departments.txt`, `grievance_procedure.txt`, `faq.txt`).
   - Grounded Q&A with transparent source document citations.

6. **Government-Grade Responsive Web Dashboard**:
   - High-performance, lightweight FastAPI application replacing heavy legacy frameworks.
   - Interactive intake chat, live side-panel diagnostics, standalone completeness auditor, searchable SQLite grievance registry, and interactive Chart.js analytics.

7. **Zero Paid Cloud API Requirement**:
   - Uses local **Ollama Qwen3** (`qwen3:8b`, fallback `qwen3:4b`, embedding `qwen3-embedding:0.6b`).
   - Includes an intelligent adaptive fallback engine so the application runs 100% reliably anywhere, including on Render free tier without local GPU.

---

## 📁 Folder Structure

```
GrievanceGPT/
├── app.py                       # FastAPI application & dashboard server
├── Procfile                     # Render start command (uvicorn app:app)
├── render.yaml                  # Render cloud blueprint
├── requirements.txt             # Pinned, compatible dependencies
├── README.md                    # Comprehensive documentation
├── .env.example                 # Environment variable template
├── .gitignore                   # Git ignore rules
│
├── data/
│   ├── civicdex.csv             # Multilingual CivicDex dataset (660 samples)
│   ├── train.csv                # Training split (540 samples)
│   ├── validation.csv           # Validation split (60 samples)
│   ├── test.csv                 # Test split (60 samples)
│   └── grievance_gpt.db         # Persistent SQLite database
│
├── models/
│   ├── intent_model.pkl         # Trained TF-IDF + Logistic Regression
│   ├── category_model.pkl       # Category classifier
│   ├── department_model.pkl     # Department classifier
│   ├── urgency_model.pkl        # Urgency/Severity classifier
│   └── vector_store/
│       └── rag_index.pkl        # Chunked vector search index
│
├── src/
│   ├── config.py                # System settings and paths
│   ├── database.py              # SQLite CRUD, tokens, and analytics
│   ├── llm/
│   │   └── ollama_client.py     # Ollama client, health check, & fallbacks
│   ├── nlp/
│   │   ├── preprocessing.py     # Multilingual text normalization
│   │   ├── classifier.py        # ML classifier inference engine
│   │   ├── train_models.py      # Reproducible model training pipeline
│   │   └── extraction.py        # Entity extraction & heuristic fallback
│   ├── rag/
│   │   ├── ingest.py            # Knowledge base chunking & indexing
│   │   ├── retriever.py         # Semantic knowledge retriever
│   │   └── qa.py                # Grounded Q&A generator with citations
│   └── grievance/
│       ├── analyzer.py          # Unified conversational coordinator
│       ├── completeness.py      # Completeness scoring logic
│       └── generator.py         # Structured grievance draft formatter
│
├── knowledge_base/              # Civic procedure & department guides
│   ├── grievance_procedure.txt
│   ├── departments.txt
│   ├── water_supply.txt
│   ├── roads.txt
│   ├── sanitation.txt
│   ├── electricity.txt
│   ├── public_health.txt
│   └── faq.txt
│
├── static/
│   ├── css/custom.css           # Glassmorphism styling and custom themes
│   └── js/dashboard.js          # Interactive chat and dashboard controller
│
├── templates/
│   └── index.html               # Responsive Government-grade dashboard SPA
│
└── tests/                       # Automated test suite (28 tests, 100% pass)
    ├── test_api.py
    ├── test_classifiers.py
    ├── test_completeness.py
    ├── test_database.py
    ├── test_dataset.py
    ├── test_extraction.py
    ├── test_generator.py
    ├── test_preprocessing.py
    └── test_rag.py
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- **Python**: 3.11 or 3.12 installed.
- **Ollama** (Optional for local LLM acceleration): [Download Ollama](https://ollama.com/)

### 2. Setup Virtual Environment & Install Dependencies
```bash
# Clone the repository and navigate into folder
cd "GrievanceGPT AI-Powered Smart Public Grievance Assistant"

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Setup Local Ollama LLM (Optional)
If running Ollama locally:
```bash
# Pull primary LLM (Qwen3 8B)
ollama pull qwen3:8b

# Pull embedding model
ollama pull qwen3-embedding:0.6b

# Lightweight fallback model (optional)
ollama pull qwen3:4b

# Start Ollama server
ollama serve
```
*(Note: If Ollama is not installed or offline, GrievanceGPT automatically activates its built-in intelligent rule & ML fallback engine, ensuring the dashboard never crashes).*

### 4. Train CivicDex Models & Build RAG Index
The models and RAG vector store train automatically on first startup. You can also trigger them manually:
```bash
# Train ML Classifiers (Intent, Category, Department, Urgency)
python src/nlp/train_models.py

# Build RAG Knowledge Base Vector Store
python src/rag/ingest.py
```

### 5. Launch the Dashboard
```bash
python app.py
```
Open your browser and navigate to:
**`http://localhost:8000`**

---

## ☁️ Deployment on Render

This project is optimized for deployment on [Render](https://render.com/) with minimal resource usage (~40MB RAM):

### Method A: Blueprint Deployment (`render.yaml`)
1. Push your repository to GitHub.
2. In the Render Dashboard, click **New +** > **Blueprint**.
3. Connect your repository. Render will automatically read `render.yaml` and configure:
   - Build Command: `pip install -r requirements.txt && python src/nlp/train_models.py && python src/rag/ingest.py`
   - Start Command: `uvicorn app:app --host 0.0.0.0 --port $PORT`

### Method B: Manual Web Service
1. Click **New +** > **Web Service**.
2. Select your repository.
3. Set **Runtime**: `Python 3`.
4. Set **Build Command**: `pip install -r requirements.txt && python src/nlp/train_models.py && python src/rag/ingest.py`
5. Set **Start Command**: `uvicorn app:app --host 0.0.0.0 --port $PORT`
6. Click **Deploy**.

---

## 🧪 Running Automated Tests

Run the full automated test suite covering dataset integrity, classifiers, preprocessing, entity extraction, completeness checks, RAG, database, and FastAPI endpoints:
```bash
pytest tests/ -v
```
All **28 automated tests** will execute and report passing status.

---

## 🎯 Verification Scenario (Primary Test Case)

**Citizen Input (Tamil/Tanglish):**
> `"எங்கள் area ல 4 days ah water வரல."`

1. **Understanding & Detection**: System identifies language as `code_mixed` / `tamil`, detects drinking water disruption, and notes a `4 days` duration.
2. **Missing Information Identified**: System detects that exact street name or locality is missing.
3. **Follow-Up Question**: Assistant responds:
   > *"I understand your grievance regarding Drinking water supply disruption or shortage. To prepare an official grievance for the Water Supply Department, could you please clarify: 1. Which specific street, ward, or locality is facing this issue?"*
4. **Follow-Up Response**: Citizen replies:
   > *"5th Cross Road, Gandhi Nagar, Katpadi"*
5. **Grievance Generation**: Completeness reaches 100%. Generates formal complaint addressed to the **Water Supply Department** with Priority: **High**, Reference Token: `GRV-2026-000004`.
6. **Citizen Confirmation**: Citizen clicks **Confirm Grievance**; the verified record is saved in SQLite under status `CONFIRMED`.

---

## ⚖️ Scope & Disclaimer

> [!IMPORTANT]
> **Prototype Scope**: GrievanceGPT is an AI-powered preparation, categorization, and verification assistant designed to help citizens formulate actionable complaints. It **does not claim** to be integrated with CPGRAMS or dispatch government teams without authorized administrative clearance.
