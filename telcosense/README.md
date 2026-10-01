<div align="center">

# 📡 TelcoSense Platform
### *Enterprise-Grade Autonomous Telecom ML & GenAI Intelligence Platform*

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![MLflow](https://img.shields.io/badge/MLflow-2.10-0194E2?logo=mlflow&logoColor=white)](https://mlflow.org/)
[![Optuna](https://img.shields.io/badge/Optuna-3.5-blue?logo=optuna&logoColor=white)](https://optuna.org/)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector_DB-DC2626?logo=qdrant&logoColor=white)](https://qdrant.tech/)
[![Apache Airflow](https://img.shields.io/badge/Airflow-2.8-017CEE?logo=apache-airflow&logoColor=white)](https://airflow.apache.org/)
[![Evidently AI](https://img.shields.io/badge/Evidently_AI-Drift_Detection-FF6B6B)](https://www.evidentlyai.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## 🏛️ Executive Summary & Sunrise ADAO Alignment

**TelcoSense** is an enterprise-scale, production-ready Machine Learning and Generative AI platform designed for tier-1 telecommunication service providers (aligned with **Sunrise ADAO — Autonomous Data & Analytics Organization**). 

The platform unifies predictive customer intelligence, automated support operations, and secure enterprise knowledge retrieval with strict Swiss data privacy guardrails:

```
                                  ┌────────────────────────────────────────────────────────┐
                                  │            TelcoSense Enterprise Platform              │
                                  └──────────────────────────┬─────────────────────────────┘
                                                             │
                 ┌───────────────────────────────────────────┼────────────────────────────────────────────┐
                 ▼                                           ▼                                            ▼
   ┌───────────────────────────┐               ┌───────────────────────────┐                ┌───────────────────────────┐
   │  Predictive ML Engine     │               │ Support AI Engine         │                │  GenAI Vector RAG Engine  │
   ├───────────────────────────┤               ├───────────────────────────┤                ├───────────────────────────┤
   │ • XGBoost Churn Model     │               │ • LightGBM Multi-class    │                │ • Qdrant Vector Index     │
   │ • ROC-AUC ~ 0.84          │               │ • Priority & Escalation   │                │ • sentence-transformers   │
   │ • Optuna Bayesian HPO     │               │ • F1-Score ~ 1.00         │                │ • Swiss PII Redaction     │
   │ • Top-5 Driver Analysis   │               │ • Real-time SLA Routing   │                │ • Grounded Context Gen    │
   └─────────────┬─────────────┘               └─────────────┬─────────────┘                └─────────────┬─────────────┘
                 │                                           │                                            │
                 └───────────────────────────────────────────┼────────────────────────────────────────────┘
                                                             │
                                                             ▼
                                  ┌────────────────────────────────────────────────────────┐
                                  │   FastAPI Gateway (/predict, /classify, /rag/query)    │
                                  └──────────────────────────┬─────────────────────────────┘
                                                             │
                                                             ▼
                                  ┌────────────────────────────────────────────────────────┐
                                  │ Continuous Training & Drift Monitoring (Airflow + MLflow│
                                  └────────────────────────────────────────────────────────┘
```

---

## 🚀 Key Platform Capabilities

### 1. 🔮 Predictive Churn Engine (`src/models/train_churn.py`)
- **Algorithm**: Scaled XGBoost Classifier with 10-trial Optuna Bayesian Hyperparameter Optimization.
- **Performance**: **ROC-AUC ~ 0.84**, Precision ~ 0.65, Recall ~ 0.77.
- **Explainability**: Returns customer-level churn risk tiers (`Low`, `Medium`, `High`) and dynamically extracts the **top 5 feature drivers** from the tree model for agent actionability.

### 2. 🎫 NLP Ticket Classifier (`src/models/train_classifier.py`)
- **Algorithm**: TF-IDF Vectorizer (bi-grams, sublinear scaling) + LightGBM Multi-class Classifier.
- **Categories**: `Billing`, `Network`, `Hardware`.
- **Performance**: **Macro & Weighted F1-Score ~ 1.00** across all categories.
- **Escalation Risk**: Heuristic NLP scoring combining prediction uncertainty and critical domain keywords (`outage`, `urgent`, `down`, `critical`).

### 3. 🧠 GenAI RAG Vector Engine (`src/rag/ingest.py`, `src/rag/retriever.py`)
- **Vector Database**: Qdrant vector engine with persistent storage.
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense vectors).
- **Document Store**: Synthetic knowledge base of Swiss telecom SOPs (5G Router Troubleshooting, Swiss Tariff & Invoicing, Fiber Optics Manual, Outage Playbooks) + 2,000 embedded support tickets.
- **Swiss PII Redaction**: Lightweight regex sanitization layer protecting Swiss IBANs (`CH\d{2}...`), international/Swiss phone numbers (`+41...`), and customer email addresses before retrieval context emission.

### 4. 🔄 MLOps Continuous Retraining Pipeline (`dags/retraining_dag.py`)
- **Drift Detection**: Evidently AI statistical drift evaluation (Wasserstein distance, Kolmogorov-Smirnov $p < 0.05$).
- **Airflow Orchestration**: Conditional branching DAG that monitors data drift on incoming batches.
- **Champion-Challenger Gate**: Automatically compares newly trained Challenger model against the Production Champion. Automatically promotes Challenger to MLflow Stage="Production" if Challenger ROC-AUC > Champion ROC-AUC.

---

## 📁 Repository Structure

```
telcosense/
├── config/
│   └── config.yaml                     # Global platform hyperparameters & paths
├── dags/
│   └── retraining_dag.py               # Airflow Continuous Retraining & Champion Gate DAG
├── data/
│   ├── raw/
│   │   ├── ibm_churn.csv               # Raw customer churn records (7,043 rows)
│   │   ├── tickets.csv                 # Raw support tickets (2,000 rows)
│   │   └── kb_docs/                    # Synthetic Swiss Telecom KB manuals
│   └── processed/
│       ├── churn_processed.parquet     # Engineered feature store
│       ├── tickets_processed.parquet   # Engineered NLP feature store
│       └── qdrant_db/                  # Persistent Qdrant vector index
├── mlruns/                             # MLflow Tracking & Model Registry store
├── src/
│   ├── data/
│   │   ├── generator.py                # Synthetic dataset & ticket generator
│   │   └── feature_store.py            # Feature engineering & transformation store
│   ├── models/
│   │   ├── train_churn.py              # XGBoost + Optuna HPO training pipeline
│   │   └── train_classifier.py         # LightGBM + TF-IDF classifier training pipeline
│   ├── monitoring/
│   │   └── drift_detector.py           # Evidently AI feature & concept drift detector
│   ├── rag/
│   │   ├── ingest.py                   # Chunking & vectorization ingestion pipeline
│   │   └── retriever.py                # Semantic retriever with Swiss PII redaction
│   └── api/
│       └── main.py                     # Production FastAPI REST Gateway
├── tests/
│   └── test_evaluation.py              # 37-point Pytest evaluation test suite
├── Dockerfile                          # Multi-stage container definition
├── docker-compose.yml                  # Full stack orchestration definition
├── requirements.txt                    # Pinned production dependencies
└── README.md                           # Platform documentation
```

---

## ⚡ Quickstart Guide

### Prerequisites
- Python 3.11+
- Docker & Docker Compose (Optional for container stack)

### Option A: Run Standalone (Local Python)

```bash
# 1. Clone repository and set up virtual environment
cd telcosense
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
# source .venv/bin/activate

# 2. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3. Step 1: Generate Raw Data & Feature Store
python src/data/generator.py
python src/data/feature_store.py

# 4. Step 2: Train & Register ML Models with MLflow
python src/models/train_churn.py --trials 10 --cv 5
python src/models/train_classifier.py --trials 10 --cv 5

# 5. Step 3: Populate Qdrant Vector Index
python src/rag/ingest.py --mode all --recreate

# 6. Step 4: Run Complete Pytest Suite (37 Tests)
pytest tests/test_evaluation.py -v

# 7. Step 5: Launch FastAPI Gateway
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

---

### Option B: Run via Docker Compose

Launch the full distributed architecture with a single command:

```bash
docker-compose up --build -d
```

#### Service URLs:
| Service | URL | Description |
| :--- | :--- | :--- |
| **FastAPI Gateway** | `http://localhost:8000/docs` | Interactive Swagger API documentation |
| **MLflow Registry** | `http://localhost:5000` | Experiment tracking & model registry UI |
| **Airflow UI** | `http://localhost:8080` | DAG scheduler UI (`admin` / `admin`) |
| **Qdrant Dashboard**| `http://localhost:6333/dashboard` | Vector storage & collection explorer |
| **PostgreSQL** | `localhost:5432` | Backend metadata store |

---

## 📡 API Reference & Endpoints

### 1. `POST /predict/churn`
Predicts churn risk and isolates feature drivers.

```json
// Request Body
{
  "SeniorCitizen": 0,
  "Partner": 1,
  "Dependents": 0,
  "tenure": 12,
  "PhoneService": 1,
  "MultipleLines": 0,
  "InternetService": 1,
  "OnlineSecurity": 0,
  "OnlineBackup": 0,
  "DeviceProtection": 0,
  "TechSupport": 0,
  "StreamingTV": 1,
  "StreamingMovies": 1,
  "Contract": 0,
  "PaperlessBilling": 1,
  "PaymentMethod": 2,
  "MonthlyCharges": 85.50,
  "TotalCharges": 1026.00,
  "gender_encoded": 1,
  "charge_per_tenure": 85.50,
  "tenure_band": 1
}

// Response Body
{
  "churn_probability": 0.7421,
  "churn_predicted": true,
  "risk_level": "High",
  "top_feature_drivers": [
    { "feature": "Contract", "value": 0.0, "importance": 0.2841 },
    { "feature": "MonthlyCharges", "value": 85.5, "importance": 0.1623 },
    { "feature": "OnlineSecurity", "value": 0.0, "importance": 0.1195 },
    { "feature": "tenure", "value": 12.0, "importance": 0.1042 },
    { "feature": "TechSupport", "value": 0.0, "importance": 0.0874 }
  ]
}
```

### 2. `POST /classify/ticket`
Classifies support ticket text, infers priority, and computes escalation risk.

```json
// Request Body
{
  "ticket_text": "URGENT: Fibre internet is completely down in Zurich office since 2 hours!"
}

// Response Body
{
  "category": "Network",
  "confidence": 0.9642,
  "priority": "High",
  "escalation_risk_score": 0.0537
}
```

### 3. `POST /rag/query`
Sanitizes customer query PII, searches Qdrant knowledge base, and returns grounded context.

```json
// Request Body
{
  "query": "Customer with phone +41 79 123 4567 asking how to fix amber LED on 5G router",
  "top_k": 3,
  "category_filter": "Network",
  "mask_pii": true
}

// Response Body
{
  "query_masked": "Customer with phone [PHONE REDACTED] asking how to fix amber LED on 5G router",
  "results": [
    {
      "chunk_id": "c71e2474-0f2c-4903-8d07-a3f2d26f7a77",
      "text": "1. No Internet Connectivity Symptoms: Router shows solid amber WAN LED...",
      "score": 0.8842,
      "category": "Network",
      "source": "5G_Router_Troubleshooting_Guide.txt",
      "title": "5G Router Troubleshooting Guide"
    }
  ],
  "context": "[1] (Network) 5G Router Troubleshooting Guide: 1. No Internet Connectivity...",
  "sources": [
    "5G_Router_Troubleshooting_Guide.txt"
  ]
}
```

### 4. `GET /health`
Returns gateway liveness, connected models, and vector engine status.

---

## 🧪 Comprehensive Evaluation Suite

Run the full end-to-end evaluation suite:

```bash
pytest tests/test_evaluation.py -v
```

```
============================== 37 passed in 50.79s ==============================
```

Covering:
- **Data Integrity**: Schema conformance, categorical bounds, null safety.
- **Feature Store**: Derived ratio validity (`charge_per_tenure`), quantile binning.
- **Model Inferences**: Probability range $[0, 1]$, risk tier assignment, multi-class distribution.
- **Monitoring**: Evidently AI zero-drift baseline invariance.
- **GenAI Retrieval**: Chunking overlap, Qdrant in-memory indexing, semantic top-$k$ recall.
- **Security Guardrails**: Multi-pattern Swiss PII redaction verification.
- **API Endpoints**: 503 circuit-breaker behavior, schema validation, mock model injection.

---

## 🛡️ License
Distributed under the MIT License. Enterprise-grade open source architecture.
