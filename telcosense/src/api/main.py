"""
TelcoSense - FastAPI Gateway
=============================
REST API exposing churn prediction, ticket classification, RAG search,
and system health endpoints.

Start:
  uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

Endpoints:
  POST /predict/churn     -> churn probability, risk level, top feature drivers
  POST /classify/ticket   -> category, priority, escalation risk score
  POST /rag/query         -> PII-masked semantic KB search with grounded context
  GET  /health            -> service and model registry status
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Final

import mlflow.sklearn
import numpy as np
import pandas as pd
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logger = logging.getLogger("telcosense.api")

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------
_PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
MLRUNS_DIR: Final[Path] = _PROJECT_ROOT / "mlruns"

# Local MLflow model URIs (Model Registry aliases registered during training)
MODEL_URI_CHURN: Final[str] = "models:/telcosense-churn-model/latest"
MODEL_URI_CLASSIFIER: Final[str] = "models:/telcosense-ticket-classifier/latest"

FEATURE_COLS: Final[list[str]] = [
    "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges",
    "gender_encoded", "charge_per_tenure", "tenure_band",
]

TICKET_CATEGORIES: Final[list[str]] = ["Billing", "Network", "Hardware"]

# ---------------------------------------------------------------------------
# Application state (models loaded once at startup)
# ---------------------------------------------------------------------------


class AppState:
    churn_model: Any = None
    ticket_classifier: Any = None
    retriever: Any = None  # TelcoRetriever | None


_state = AppState()


def _load_mlflow_model(uri: str) -> Any:
    """Try multiple URI strategies to load a local MLflow model."""
    mlflow.set_tracking_uri(str(MLRUNS_DIR))

    # Strategy 1: registry alias (latest version)
    try:
        return mlflow.sklearn.load_model(uri)
    except Exception:
        pass

    # Strategy 2: resolve latest run artifact directly
    client = mlflow.tracking.MlflowClient(str(MLRUNS_DIR))
    model_name = uri.split("/")[1]
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
        if versions:
            latest = sorted(versions, key=lambda v: int(v.version), reverse=True)[0]
            run_uri = f"runs:/{latest.run_id}/{model_name.replace('-', '_')}"
            return mlflow.sklearn.load_model(run_uri)
    except Exception:
        pass

    # Strategy 3: scan mlruns for the artifact
    for artifact_dir in MLRUNS_DIR.glob(f"**/artifacts/{model_name.replace('-', '_')}/MLmodel"):
        model_dir = artifact_dir.parent
        try:
            return mlflow.sklearn.load_model(str(model_dir))
        except Exception:
            continue

    raise RuntimeError(f"Could not load model from URI: {uri}")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load ML models and retriever at startup; release on shutdown."""
    mlflow.set_tracking_uri(str(MLRUNS_DIR))
    logger.info("MLflow tracking URI: %s", MLRUNS_DIR)

    logger.info("Loading churn model from MLflow registry ...")
    try:
        _state.churn_model = _load_mlflow_model(MODEL_URI_CHURN)
        logger.info("Churn model loaded successfully.")
    except Exception as exc:
        logger.warning("Churn model not available: %s", exc)

    logger.info("Loading ticket classifier from MLflow registry ...")
    try:
        _state.ticket_classifier = _load_mlflow_model(MODEL_URI_CLASSIFIER)
        logger.info("Ticket classifier loaded successfully.")
    except Exception as exc:
        logger.warning("Ticket classifier not available: %s", exc)

    logger.info("Initialising RAG retriever ...")
    try:
        from src.rag.retriever import TelcoRetriever, RetrieverConfig
        _state.retriever = TelcoRetriever(RetrieverConfig())
        logger.info("RAG retriever initialised (collection: %s).", _state.retriever.config.collection_name)
    except Exception as exc:
        logger.warning("Retriever not available: %s", exc)

    yield

    logger.info("Shutting down TelcoSense API.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="TelcoSense API",
    description="Enterprise Telecom ML & GenAI Platform - REST Gateway",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class CustomerFeatures(BaseModel):
    """Feature vector for a single customer churn prediction."""

    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: int = Field(ge=0, le=1)
    Dependents: int = Field(ge=0, le=1)
    tenure: int = Field(ge=0)
    PhoneService: int = Field(ge=0, le=1)
    MultipleLines: int = Field(ge=0)
    InternetService: int = Field(ge=0)
    OnlineSecurity: int = Field(ge=0)
    OnlineBackup: int = Field(ge=0)
    DeviceProtection: int = Field(ge=0)
    TechSupport: int = Field(ge=0)
    StreamingTV: int = Field(ge=0)
    StreamingMovies: int = Field(ge=0)
    Contract: int = Field(ge=0)
    PaperlessBilling: int = Field(ge=0, le=1)
    PaymentMethod: int = Field(ge=0)
    MonthlyCharges: float = Field(ge=0.0)
    TotalCharges: float = Field(ge=0.0)
    gender_encoded: int = Field(ge=0, le=1)
    charge_per_tenure: float
    tenure_band: int = Field(ge=0, le=4)


class FeatureDriver(BaseModel):
    feature: str
    value: float
    importance: float


class ChurnPredictionResponse(BaseModel):
    churn_probability: float
    churn_predicted: bool
    risk_level: str  # Low / Medium / High
    top_feature_drivers: list[FeatureDriver] = Field(default_factory=list)


class TicketClassifyRequest(BaseModel):
    ticket_text: str = Field(min_length=5)


class TicketClassifyResponse(BaseModel):
    category: str
    confidence: float
    priority: str
    escalation_risk_score: float


class RAGQueryRequest(BaseModel):
    query: str = Field(min_length=3, description="Customer query text")
    top_k: int = Field(default=5, ge=1, le=20)
    category_filter: str | None = Field(default=None, description="Billing / Network / Hardware")
    mask_pii: bool = Field(default=True, description="Redact PII from retrieved context")


class RAGQueryResponse(BaseModel):
    query_masked: str
    results: list[dict]
    context: str
    sources: list[str]


# Legacy search endpoint schema (backward compatible)
class RAGSearchRequest(BaseModel):
    query: str = Field(min_length=3)
    top_k: int = Field(default=5, ge=1, le=20)
    category_filter: str | None = None
    priority_filter: str | None = None


class RAGSearchResponse(BaseModel):
    results: list[dict]
    context: str


class HealthResponse(BaseModel):
    status: str
    churn_model_loaded: bool
    classifier_loaded: bool
    retriever_connected: bool
    mlflow_uri: str


# ---------------------------------------------------------------------------
# Dependency helpers
# ---------------------------------------------------------------------------


def get_churn_model() -> Any:
    if _state.churn_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Churn model not loaded. Run the training pipeline first.",
        )
    return _state.churn_model


def get_classifier() -> Any:
    if _state.ticket_classifier is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Ticket classifier not loaded.",
        )
    return _state.ticket_classifier


def get_retriever() -> Any:
    if _state.retriever is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG retriever not available. Run ingest pipeline first.",
        )
    return _state.retriever


# ---------------------------------------------------------------------------
# Helper: top feature drivers from XGBoost pipeline
# ---------------------------------------------------------------------------


def _get_top_drivers(model: Any, row: pd.DataFrame, top_n: int = 5) -> list[FeatureDriver]:
    """Extract top N feature importances from the XGBoost pipeline."""
    drivers: list[FeatureDriver] = []
    try:
        # XGBoost is typically the last step in a sklearn Pipeline
        xgb_step = model
        if hasattr(model, "steps"):
            xgb_step = model.steps[-1][1]
        importances = xgb_step.feature_importances_
        feature_names = FEATURE_COLS
        paired = sorted(
            zip(feature_names, row.iloc[0].tolist(), importances),
            key=lambda x: x[2], reverse=True,
        )[:top_n]
        drivers = [
            FeatureDriver(feature=f, value=float(v), importance=float(imp))
            for f, v, imp in paired
        ]
    except Exception:
        pass
    return drivers


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health_check() -> HealthResponse:
    """Liveness and readiness check with model registry status."""
    return HealthResponse(
        status="ok",
        churn_model_loaded=_state.churn_model is not None,
        classifier_loaded=_state.ticket_classifier is not None,
        retriever_connected=_state.retriever is not None,
        mlflow_uri=str(MLRUNS_DIR),
    )


@app.post(
    "/predict/churn",
    response_model=ChurnPredictionResponse,
    tags=["predictions"],
    summary="Predict customer churn probability with feature drivers",
)
async def predict_churn(
    features: CustomerFeatures,
    model: Any = Depends(get_churn_model),
) -> ChurnPredictionResponse:
    """
    Returns churn probability, risk tier (Low/Medium/High), and the
    top 5 most influential feature drivers for interpretability.
    """
    row = pd.DataFrame([features.model_dump()])[FEATURE_COLS]
    prob = float(model.predict_proba(row)[0, 1])
    predicted = prob >= 0.5
    risk = "High" if prob >= 0.7 else ("Medium" if prob >= 0.4 else "Low")
    drivers = _get_top_drivers(model, row)
    return ChurnPredictionResponse(
        churn_probability=round(prob, 4),
        churn_predicted=predicted,
        risk_level=risk,
        top_feature_drivers=drivers,
    )


@app.post(
    "/classify/ticket",
    response_model=TicketClassifyResponse,
    tags=["predictions"],
    summary="Classify support ticket and assess escalation risk",
)
async def classify_ticket(
    request: TicketClassifyRequest,
    clf: Any = Depends(get_classifier),
) -> TicketClassifyResponse:
    """
    Classify a raw support ticket into Billing / Network / Hardware,
    infer priority (High if escalation keywords found), and compute
    an escalation risk score from the prediction confidence.
    """
    proba = clf.predict_proba([request.ticket_text])[0]
    predicted_class = str(clf.classes_[int(np.argmax(proba))])
    confidence = float(np.max(proba))

    # Heuristic priority and escalation risk
    text_lower = request.ticket_text.lower()
    high_priority_keywords = {"urgent", "outage", "down", "cannot", "broken", "fail", "critical", "emergency"}
    priority = "High" if any(kw in text_lower for kw in high_priority_keywords) else "Low"
    # Escalation risk: high-priority + low-confidence = more uncertain = higher risk
    escalation_risk = round((1.0 - confidence) * (1.5 if priority == "High" else 1.0), 4)
    escalation_risk = min(escalation_risk, 1.0)

    return TicketClassifyResponse(
        category=predicted_class,
        confidence=round(confidence, 4),
        priority=priority,
        escalation_risk_score=escalation_risk,
    )


@app.post(
    "/rag/query",
    response_model=RAGQueryResponse,
    tags=["rag"],
    summary="Semantic KB query with PII masking and grounded context",
)
async def rag_query(
    request: RAGQueryRequest,
    retriever: Any = Depends(get_retriever),
) -> RAGQueryResponse:
    """
    Accepts a customer query, optionally masks PII, retrieves relevant
    chunks from the Qdrant KB, and returns a grounded context string
    suitable for LLM augmentation.
    """
    from src.rag.retriever import mask_pii_text

    # Mask PII in the incoming query before logging / returning
    safe_query = mask_pii_text(request.query) if request.mask_pii else request.query

    chunks = retriever.search(
        query=request.query,  # embed original query for better recall
        top_k=request.top_k,
        category_filter=request.category_filter,
        mask_pii=request.mask_pii,
    )

    context = retriever.format_context(chunks)
    sources = list({c.source or c.title for c in chunks if c.source or c.title})

    return RAGQueryResponse(
        query_masked=safe_query,
        results=[c.model_dump() for c in chunks],
        context=context,
        sources=sources,
    )


@app.post(
    "/rag/search",
    response_model=RAGSearchResponse,
    tags=["rag"],
    summary="[Legacy] Semantic search over ticket knowledge base",
)
async def rag_search(
    request: RAGSearchRequest,
    retriever: Any = Depends(get_retriever),
) -> RAGSearchResponse:
    """Backward-compatible semantic search endpoint (no PII masking)."""
    chunks = retriever.search(
        query=request.query,
        top_k=request.top_k,
        category_filter=request.category_filter,
        priority_filter=request.priority_filter,
    )
    context = retriever.format_context(chunks)
    return RAGSearchResponse(
        results=[c.model_dump() for c in chunks],
        context=context,
    )


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
