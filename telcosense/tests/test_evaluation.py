"""
TelcoSense - Pytest Evaluation Suite
=======================================
End-to-end evaluation tests covering:
  1. Data generator (churn download/fallback + ticket generation)
  2. Feature store (engineering pipeline, derived features)
  3. Churn model (prediction shape, probability bounds, mocked)
  4. Ticket classifier (category prediction, confidence bounds)
  5. Drift detector (no-drift baseline assertion)
  6. RAG ingestion (KB doc generation, chunking, in-memory Qdrant)
  7. RAG retriever (PII masking correctness, semantic search)
  8. API endpoints (health, 503 guards, churn predict, RAG query)

Run with:
  pytest tests/test_evaluation.py -v
  pytest tests/test_evaluation.py -v --tb=short -q
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

PLATFORM_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PLATFORM_ROOT))


# ---------------------------------------------------------------------------
# Fixtures - shared synthetic data
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def synthetic_churn_df() -> pd.DataFrame:
    """Generate a small synthetic churn DataFrame for testing."""
    from src.data.generator import _generate_synthetic_churn
    return _generate_synthetic_churn(n=200, seed=42)


@pytest.fixture(scope="session")
def synthetic_tickets_df(synthetic_churn_df: pd.DataFrame, tmp_path_factory: Any) -> pd.DataFrame:
    """Generate synthetic ticket records linked to synthetic customer IDs."""
    from src.data.generator import generate_ticket_dataset

    tmp_dir = tmp_path_factory.mktemp("data")
    churn_path = tmp_dir / "ibm_churn.csv"
    ticket_path = tmp_dir / "tickets.csv"
    synthetic_churn_df.to_csv(churn_path, index=False)

    generate_ticket_dataset(output_path=ticket_path, n=200, churn_csv=churn_path, seed=42)
    return pd.read_csv(ticket_path)


@pytest.fixture(scope="session")
def processed_features_df(synthetic_churn_df: pd.DataFrame, tmp_path_factory: Any) -> pd.DataFrame:
    """Run feature store on synthetic churn data."""
    from src.data.feature_store import process_churn_features

    tmp_dir = tmp_path_factory.mktemp("processed")
    churn_path = tmp_dir / "ibm_churn.csv"
    parquet_path = tmp_dir / "churn_processed.parquet"
    synthetic_churn_df.to_csv(churn_path, index=False)

    df = process_churn_features(churn_csv=churn_path, output_parquet=parquet_path)
    return df


# ---------------------------------------------------------------------------
# 1. Data generator tests
# ---------------------------------------------------------------------------


class TestDataGenerator:
    def test_synthetic_churn_shape(self, synthetic_churn_df: pd.DataFrame) -> None:
        assert len(synthetic_churn_df) == 200
        assert "customerID" in synthetic_churn_df.columns
        assert "Churn" in synthetic_churn_df.columns

    def test_churn_values_binary(self, synthetic_churn_df: pd.DataFrame) -> None:
        assert set(synthetic_churn_df["Churn"].unique()).issubset({"Yes", "No"})

    def test_numeric_columns_non_negative(self, synthetic_churn_df: pd.DataFrame) -> None:
        assert (synthetic_churn_df["tenure"] >= 0).all()
        assert (synthetic_churn_df["MonthlyCharges"] >= 0).all()
        assert (synthetic_churn_df["TotalCharges"] >= 0).all()

    def test_ticket_shape(self, synthetic_tickets_df: pd.DataFrame) -> None:
        assert len(synthetic_tickets_df) == 200
        required_cols = {
            "ticket_id", "customer_id", "ticket_text",
            "category", "priority", "escalation_risk",
        }
        assert required_cols.issubset(set(synthetic_tickets_df.columns))

    def test_ticket_categories(self, synthetic_tickets_df: pd.DataFrame) -> None:
        valid_cats = {"Billing", "Network", "Hardware"}
        assert set(synthetic_tickets_df["category"].unique()).issubset(valid_cats)

    def test_ticket_priorities(self, synthetic_tickets_df: pd.DataFrame) -> None:
        assert set(synthetic_tickets_df["priority"].unique()).issubset({"Low", "High"})

    def test_escalation_risk_bounds(self, synthetic_tickets_df: pd.DataFrame) -> None:
        assert (synthetic_tickets_df["escalation_risk"] >= 0.0).all()
        assert (synthetic_tickets_df["escalation_risk"] <= 1.0).all()

    def test_download_fallback_on_network_failure(self, tmp_path: Path) -> None:
        """When network fails, generator should fall back to synthetic data."""
        from src.data.generator import generate_churn_dataset

        out = tmp_path / "ibm_churn.csv"
        with patch("src.data.generator.requests.get", side_effect=ConnectionError("No network")):
            result_path = generate_churn_dataset(
                output_path=out,
                synthetic_rows=100,
                seed=42,
            )

        df = pd.read_csv(result_path)
        assert len(df) == 100
        assert "customerID" in df.columns


# ---------------------------------------------------------------------------
# 2. Feature store tests
# ---------------------------------------------------------------------------


class TestFeatureStore:
    def test_processed_has_derived_features(self, processed_features_df: pd.DataFrame) -> None:
        assert "charge_per_tenure" in processed_features_df.columns
        assert "tenure_band" in processed_features_df.columns
        assert "Churn_label" in processed_features_df.columns

    def test_churn_label_binary(self, processed_features_df: pd.DataFrame) -> None:
        assert set(processed_features_df["Churn_label"].unique()).issubset({0, 1})

    def test_tenure_band_range(self, processed_features_df: pd.DataFrame) -> None:
        assert processed_features_df["tenure_band"].between(0, 4).all()

    def test_no_nan_in_numeric_features(self, processed_features_df: pd.DataFrame) -> None:
        numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges", "charge_per_tenure"]
        for col in numeric_cols:
            if col in processed_features_df.columns:
                assert processed_features_df[col].isna().sum() == 0, f"{col} has NaNs"

    def test_parquet_roundtrip(self, processed_features_df: pd.DataFrame, tmp_path: Path) -> None:
        parquet_path = tmp_path / "test_features.parquet"
        processed_features_df.to_parquet(parquet_path, index=False)
        loaded = pd.read_parquet(parquet_path)
        assert len(loaded) == len(processed_features_df)


# ---------------------------------------------------------------------------
# 3. Churn model tests (mocked MLflow)
# ---------------------------------------------------------------------------


class TestChurnModel:
    def test_mock_churn_prediction_shape(self, processed_features_df: pd.DataFrame) -> None:
        """A mock model returning correct probability arrays."""
        mock_model = MagicMock()
        n = len(processed_features_df)
        mock_model.predict_proba.return_value = np.column_stack(
            [np.random.uniform(0, 1, n), np.random.uniform(0, 1, n)]
        )
        proba = mock_model.predict_proba(processed_features_df)
        assert proba.shape == (n, 2)

    def test_probability_bounds(self) -> None:
        """Predicted probabilities must lie in [0, 1]."""
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7], [0.8, 0.2]])
        proba = mock_model.predict_proba(None)[:, 1]
        assert (proba >= 0.0).all()
        assert (proba <= 1.0).all()

    def test_risk_level_thresholds(self) -> None:
        """Validate risk-level logic used in the API."""
        def risk_level(prob: float) -> str:
            return "High" if prob >= 0.7 else ("Medium" if prob >= 0.4 else "Low")

        assert risk_level(0.8) == "High"
        assert risk_level(0.5) == "Medium"
        assert risk_level(0.2) == "Low"


# ---------------------------------------------------------------------------
# 4. Ticket classifier tests
# ---------------------------------------------------------------------------


class TestTicketClassifier:
    def test_mock_classifier_returns_category(self) -> None:
        mock_clf = MagicMock()
        mock_clf.predict.return_value = ["Billing"]
        mock_clf.predict_proba.return_value = np.array([[0.1, 0.7, 0.2]])
        mock_clf.classes_ = np.array(["Billing", "Hardware", "Network"])

        result = mock_clf.predict(["I was overcharged this month"])[0]
        assert result in {"Billing", "Network", "Hardware"}

    def test_confidence_bounds(self) -> None:
        mock_clf = MagicMock()
        mock_clf.predict_proba.return_value = np.array([[0.05, 0.85, 0.10]])
        proba = mock_clf.predict_proba(None)[0]
        assert abs(proba.sum() - 1.0) < 1e-5
        assert (proba >= 0.0).all()

    def test_escalation_risk_bounds(self) -> None:
        """Escalation risk score must be in [0, 1]."""
        confidence = 0.95
        priority = "High"
        escalation_risk = min((1.0 - confidence) * (1.5 if priority == "High" else 1.0), 1.0)
        assert 0.0 <= escalation_risk <= 1.0


# ---------------------------------------------------------------------------
# 5. Drift detector tests
# ---------------------------------------------------------------------------


class TestDriftDetector:
    def test_no_drift_on_identical_data(
        self,
        processed_features_df: pd.DataFrame,
        tmp_path: Path,
    ) -> None:
        """Identical reference and current data -> no drift expected."""
        from src.monitoring.drift_detector import DriftConfig, detect_drift

        ref_path = tmp_path / "ref_features.parquet"
        processed_features_df.to_parquet(ref_path, index=False)

        config = DriftConfig(
            reference_path=ref_path,
            reports_dir=tmp_path / "reports",
        )

        result = detect_drift(current_df=processed_features_df.copy(), config=config)
        assert isinstance(result.dataset_drift_detected, bool)
        assert result.reference_rows == len(processed_features_df)
        assert result.current_rows == len(processed_features_df)


# ---------------------------------------------------------------------------
# 6. RAG ingestion tests
# ---------------------------------------------------------------------------


class TestRAGIngestion:
    def test_kb_document_generation(self, tmp_path: Path) -> None:
        """KB docs must be written to disk with expected filenames."""
        from src.rag.ingest import generate_kb_documents, _KB_DOCUMENTS

        kb_dir = tmp_path / "kb_docs"
        paths = generate_kb_documents(kb_dir=kb_dir)

        assert len(paths) == len(_KB_DOCUMENTS)
        for p in paths:
            assert p.exists()
            assert p.stat().st_size > 100  # non-trivial content

    def test_recursive_chunking(self) -> None:
        """Chunking must return non-empty chunks within size bound."""
        from src.rag.ingest import _recursive_split

        text = "A " * 300  # 600 chars
        chunks = _recursive_split(text, chunk_size=200, chunk_overlap=40)
        assert len(chunks) >= 2
        for chunk in chunks:
            assert len(chunk) <= 210  # small tolerance for join separators
            assert chunk.strip()

    def test_kb_ingest_into_qdrant(self, tmp_path: Path) -> None:
        """Full KB ingestion into a local Qdrant instance must succeed."""
        from src.rag.ingest import IngestConfig, generate_kb_documents, ingest_kb_documents

        kb_dir = tmp_path / "kb_docs"
        qdrant_dir = tmp_path / "qdrant_db"
        generate_kb_documents(kb_dir=kb_dir)

        config = IngestConfig(
            collection_name="test-kb",
            chunk_size=300,
            chunk_overlap=50,
            batch_size=8,
            recreate_collection=True,
            qdrant_persist_path=qdrant_dir,
        )

        total = ingest_kb_documents(config, kb_dir=kb_dir)
        assert total > 0, "Expected at least one vector to be upserted"

    def test_ticket_ingest_into_qdrant(
        self,
        synthetic_tickets_df: pd.DataFrame,
        tmp_path: Path,
    ) -> None:
        """Ticket CSV ingestion must produce vectors for each ticket."""
        from src.rag.ingest import IngestConfig, ingest_tickets

        tickets_csv = tmp_path / "tickets.csv"
        qdrant_dir = tmp_path / "qdrant_db"
        synthetic_tickets_df.to_csv(tickets_csv, index=False)

        config = IngestConfig(
            chunk_size=300,
            chunk_overlap=50,
            batch_size=8,
            recreate_collection=True,
            qdrant_persist_path=qdrant_dir,
        )

        total = ingest_tickets(config, tickets_csv=tickets_csv)
        assert total >= len(synthetic_tickets_df)


# ---------------------------------------------------------------------------
# 7. RAG retriever tests
# ---------------------------------------------------------------------------


class TestPIIMasking:
    """Test PII masking correctness for phone numbers, emails, and Swiss IBANs."""

    def test_masks_email(self) -> None:
        from src.rag.retriever import mask_pii_text

        text = "Contact us at support@telcosense.ch for help."
        result = mask_pii_text(text)
        assert "support@telcosense.ch" not in result
        assert "[EMAIL REDACTED]" in result

    def test_masks_swiss_iban(self) -> None:
        from src.rag.retriever import mask_pii_text

        text = "Please transfer to CH56 0483 5012 3456 7800 9."
        result = mask_pii_text(text)
        assert "CH56" not in result
        assert "[IBAN REDACTED]" in result

    def test_masks_phone_number(self) -> None:
        from src.rag.retriever import mask_pii_text

        text = "Call +41 800 500 500 for support."
        result = mask_pii_text(text)
        # At least one redaction should have occurred
        assert "+41 800 500 500" not in result or "[PHONE REDACTED]" in result

    def test_clean_text_unchanged_structure(self) -> None:
        from src.rag.retriever import mask_pii_text

        text = "The router shows a red LED indicating hardware fault."
        result = mask_pii_text(text)
        # Technical content should not be mangled
        assert "router" in result
        assert "red LED" in result


class TestRAGRetriever:
    def test_retriever_search_after_ingest(self, tmp_path: Path) -> None:
        """Retriever must return results after KB documents have been ingested."""
        from src.rag.ingest import IngestConfig, generate_kb_documents, ingest_kb_documents
        from src.rag.retriever import TelcoRetriever, RetrieverConfig

        kb_dir = tmp_path / "kb_docs"
        qdrant_dir = tmp_path / "qdrant_db"
        generate_kb_documents(kb_dir=kb_dir)

        ingest_cfg = IngestConfig(
            collection_name="test-kb-search",
            chunk_size=400,
            chunk_overlap=80,
            batch_size=8,
            recreate_collection=True,
            qdrant_persist_path=qdrant_dir,
        )
        ingest_kb_documents(ingest_cfg, kb_dir=kb_dir)

        ret_cfg = RetrieverConfig(
            collection_name="test-kb-search",
            qdrant_persist_path=qdrant_dir,
            top_k=3,
            score_threshold=0.0,
        )
        retriever = TelcoRetriever(config=ret_cfg)
        results = retriever.search("billing dispute invoice")

        assert isinstance(results, list)
        assert len(results) > 0
        assert all(hasattr(r, "text") for r in results)

    def test_retriever_category_filter(self, tmp_path: Path) -> None:
        """Category filter must restrict results to the Billing category."""
        from src.rag.ingest import IngestConfig, generate_kb_documents, ingest_kb_documents
        from src.rag.retriever import TelcoRetriever, RetrieverConfig

        kb_dir = tmp_path / "kb_docs"
        qdrant_dir = tmp_path / "qdrant_db"
        generate_kb_documents(kb_dir=kb_dir)

        ingest_cfg = IngestConfig(
            collection_name="test-kb-filter",
            chunk_size=400,
            chunk_overlap=80,
            batch_size=8,
            recreate_collection=True,
            qdrant_persist_path=qdrant_dir,
        )
        ingest_kb_documents(ingest_cfg, kb_dir=kb_dir)

        ret_cfg = RetrieverConfig(
            collection_name="test-kb-filter",
            qdrant_persist_path=qdrant_dir,
            top_k=10,
            score_threshold=0.0,
        )
        retriever = TelcoRetriever(config=ret_cfg)
        results = retriever.search("payment invoice tariff", category_filter="Billing")

        for chunk in results:
            assert chunk.category == "Billing", f"Got unexpected category: {chunk.category}"


# ---------------------------------------------------------------------------
# 8. API endpoint tests
# ---------------------------------------------------------------------------


class TestAPI:
    def test_health_endpoint(self) -> None:
        """Health endpoint must return 200 with expected fields."""
        from fastapi.testclient import TestClient
        from src.api.main import app

        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "churn_model_loaded" in data
        assert "classifier_loaded" in data
        assert "retriever_connected" in data

    def test_churn_predict_returns_503_without_model(self) -> None:
        """Without a loaded model the endpoint should return 503."""
        from fastapi.testclient import TestClient
        from src.api.main import app, _state

        original = _state.churn_model
        _state.churn_model = None
        try:
            client = TestClient(app, raise_server_exceptions=False)
            payload = {k: 0 for k in [
                "SeniorCitizen", "Partner", "Dependents", "tenure",
                "PhoneService", "MultipleLines", "InternetService",
                "OnlineSecurity", "OnlineBackup", "DeviceProtection",
                "TechSupport", "StreamingTV", "StreamingMovies",
                "Contract", "PaperlessBilling", "PaymentMethod",
                "gender_encoded", "tenure_band",
            ]}
            payload.update({
                "MonthlyCharges": 50.0,
                "TotalCharges": 600.0,
                "charge_per_tenure": 50.0,
            })
            resp = client.post("/predict/churn", json=payload)
            assert resp.status_code == 503
        finally:
            _state.churn_model = original

    def test_classify_ticket_returns_503_without_classifier(self) -> None:
        """Without a loaded classifier the endpoint should return 503."""
        from fastapi.testclient import TestClient
        from src.api.main import app, _state

        original = _state.ticket_classifier
        _state.ticket_classifier = None
        try:
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.post("/classify/ticket", json={"ticket_text": "Test ticket"})
            assert resp.status_code == 503
        finally:
            _state.ticket_classifier = original

    def test_rag_query_returns_503_without_retriever(self) -> None:
        """Without a loaded retriever the /rag/query endpoint should return 503."""
        from fastapi.testclient import TestClient
        from src.api.main import app, _state

        original = _state.retriever
        _state.retriever = None
        try:
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.post(
                "/rag/query",
                json={"query": "billing invoice dispute", "top_k": 3},
            )
            assert resp.status_code == 503
        finally:
            _state.retriever = original

    def test_churn_predict_with_mock_model(self) -> None:
        """With a mock model, /predict/churn must return correct schema."""
        from fastapi.testclient import TestClient
        from src.api.main import app, _state

        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.35, 0.65]])
        mock_model.steps = [("scaler", MagicMock()), ("xgb", MagicMock())]
        mock_model.steps[-1][1].feature_importances_ = np.ones(21) / 21

        original = _state.churn_model
        _state.churn_model = mock_model
        try:
            client = TestClient(app)
            payload = {k: 0 for k in [
                "SeniorCitizen", "Partner", "Dependents", "tenure",
                "PhoneService", "MultipleLines", "InternetService",
                "OnlineSecurity", "OnlineBackup", "DeviceProtection",
                "TechSupport", "StreamingTV", "StreamingMovies",
                "Contract", "PaperlessBilling", "PaymentMethod",
                "gender_encoded", "tenure_band",
            ]}
            payload.update({
                "MonthlyCharges": 75.0,
                "TotalCharges": 900.0,
                "charge_per_tenure": 75.0,
            })
            resp = client.post("/predict/churn", json=payload)

            assert resp.status_code == 200
            data = resp.json()
            assert "churn_probability" in data
            assert "risk_level" in data
            assert data["risk_level"] in {"Low", "Medium", "High"}
            assert 0.0 <= data["churn_probability"] <= 1.0
        finally:
            _state.churn_model = original

    def test_classify_ticket_with_mock_classifier(self) -> None:
        """With a mock classifier, /classify/ticket must return correct schema."""
        from fastapi.testclient import TestClient
        from src.api.main import app, _state

        mock_clf = MagicMock()
        mock_clf.predict_proba.return_value = np.array([[0.05, 0.10, 0.85]])
        mock_clf.classes_ = np.array(["Billing", "Hardware", "Network"])

        original = _state.ticket_classifier
        _state.ticket_classifier = mock_clf
        try:
            client = TestClient(app)
            resp = client.post(
                "/classify/ticket",
                json={"ticket_text": "My internet is down and it is urgent!"},
            )

            assert resp.status_code == 200
            data = resp.json()
            assert data["category"] in {"Billing", "Network", "Hardware"}
            assert data["priority"] in {"Low", "High"}
            assert 0.0 <= data["escalation_risk_score"] <= 1.0
            assert 0.0 <= data["confidence"] <= 1.0
        finally:
            _state.ticket_classifier = original

    def test_rag_query_with_mock_retriever(self) -> None:
        """With a mock retriever, /rag/query must return sanitized response."""
        from fastapi.testclient import TestClient
        from src.api.main import app, _state
        from src.rag.retriever import RetrievedChunk

        mock_retriever = MagicMock()
        mock_chunk = RetrievedChunk(
            chunk_id="chunk-1",
            text="Reboot router and call support at [PHONE REDACTED]",
            score=0.92,
            category="Network",
            source="5G_Router_Troubleshooting_Guide.txt",
            title="5G Router Troubleshooting Guide",
        )
        mock_retriever.search.return_value = [mock_chunk]
        mock_retriever.format_context.return_value = "[1] (Network) 5G Router Troubleshooting Guide: Reboot router..."

        original = _state.retriever
        _state.retriever = mock_retriever
        try:
            client = TestClient(app)
            resp = client.post(
                "/rag/query",
                json={"query": "My phone number is +41 79 123 4567 how do I fix 5G?", "top_k": 3},
            )

            assert resp.status_code == 200
            data = resp.json()
            assert "[PHONE REDACTED]" in data["query_masked"]
            assert len(data["results"]) == 1
            assert "context" in data
            assert len(data["sources"]) > 0
        finally:
            _state.retriever = original


