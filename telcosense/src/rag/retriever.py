"""
TelcoSense - RAG Retriever
============================
Semantic search over the TelcoSense Qdrant vector store.
Supports metadata filtering by category/priority and includes
a PII masking layer that redacts phone numbers, email addresses,
and Swiss IBANs before returning retrieved chunks.

Usage:
  from src.rag.retriever import TelcoRetriever, RetrieverConfig
  retriever = TelcoRetriever()
  results = retriever.search("billing dispute late fee", top_k=5)
  safe_context = retriever.search_and_format("billing dispute", mask_pii=True)
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, ScoredPoint
from sentence_transformers import SentenceTransformer

logger = logging.getLogger("telcosense.rag.retriever")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
QDRANT_PERSIST_DIR: Final[Path] = _PROJECT_ROOT / "data" / "processed" / "qdrant_db"

DEFAULT_KB_COLLECTION: Final[str] = "telcosense-kb"
DEFAULT_TICKET_COLLECTION: Final[str] = "telco_tickets"
DEFAULT_EMBEDDING_MODEL: Final[str] = "all-MiniLM-L6-v2"
DEFAULT_TOP_K: Final[int] = 5

# ---------------------------------------------------------------------------
# PII Masking Patterns
# ---------------------------------------------------------------------------

# Swiss IBAN: CH followed by 2 check digits + 5 bank digits + 12 account digits
_SWISS_IBAN_RE = re.compile(
    r"\bCH\d{2}[\s]?\d{4}[\s]?\d{4}[\s]?\d{4}[\s]?\d{4}[\s]?\d{1}\b",
    re.IGNORECASE,
)
# International phone: +XX NNN NNN NNNN variants
_PHONE_RE = re.compile(
    r"(?<!\w)"
    r"(\+?\d{1,3}[\s.\-]?)?"
    r"\(?\d{2,4}\)?[\s.\-]?"
    r"\d{3,4}[\s.\-]?"
    r"\d{3,4}"
    r"(?!\w)",
)
# Email addresses
_EMAIL_RE = re.compile(
    r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"
)


def mask_pii(text: str) -> str:
    """
    Redact PII from *text* in-place.

    Replacements:
      - Swiss IBANs      → [IBAN REDACTED]
      - Phone numbers    → [PHONE REDACTED]
      - Email addresses  → [EMAIL REDACTED]
    """
    text = _SWISS_IBAN_RE.sub("[IBAN REDACTED]", text)
    text = _EMAIL_RE.sub("[EMAIL REDACTED]", text)
    text = _PHONE_RE.sub("[PHONE REDACTED]", text)
    return text


# ---------------------------------------------------------------------------
# Pydantic config and result schemas
# ---------------------------------------------------------------------------


class RetrieverConfig(BaseModel):
    """Configuration for the Qdrant retriever."""

    collection_name: str = DEFAULT_KB_COLLECTION
    qdrant_persist_path: Path = QDRANT_PERSIST_DIR
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=100)
    score_threshold: float = Field(default=0.0, ge=0.0, le=1.0)


class RetrievedChunk(BaseModel):
    """A single retrieved document chunk with optional PII masking."""

    source: str = ""
    title: str = ""
    category: str = ""
    priority: str = ""
    escalation_risk: float = 0.0
    doc_type: str = "knowledge_base"
    # ticket-specific (optional)
    ticket_id: str = ""
    customer_id: str = ""
    text: str
    score: float = Field(ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Retriever class
# ---------------------------------------------------------------------------


class TelcoRetriever:
    """
    Semantic retriever over the TelcoSense Qdrant knowledge base.

    Parameters
    ----------
    config:
        RetrieverConfig instance. Defaults to local on-disk Qdrant.
    """

    def __init__(self, config: RetrieverConfig | None = None) -> None:
        self.config = config or RetrieverConfig()
        self._client: QdrantClient | None = None
        self._model: SentenceTransformer | None = None

    # ---- Lazy initialisation -----------------------------------------------

    @property
    def client(self) -> QdrantClient:
        if self._client is None:
            persist_path = self.config.qdrant_persist_path
            persist_path.mkdir(parents=True, exist_ok=True)
            self._client = QdrantClient(path=str(persist_path))
            logger.info("Connected to Qdrant at %s.", persist_path)
        return self._client

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            logger.info("Loading embedding model '%s' ...", self.config.embedding_model)
            self._model = SentenceTransformer(self.config.embedding_model)
        return self._model

    # ---- Public API --------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int | None = None,
        category_filter: str | None = None,
        priority_filter: str | None = None,
        doc_type_filter: str | None = None,
        mask_pii: bool = False,
    ) -> list[RetrievedChunk]:
        """
        Perform semantic search with optional metadata filtering.

        Parameters
        ----------
        query:
            Free-text search query (PII is NOT masked in the embedding).
        top_k:
            Number of results to return.  Overrides config.top_k if provided.
        category_filter:
            Filter by category: Billing, Network, or Hardware.
        priority_filter:
            Filter by priority: Low or High (tickets only).
        doc_type_filter:
            Filter by doc_type: knowledge_base or ticket.
        mask_pii:
            If True, redact phone numbers, emails, and IBANs from returned text.

        Returns
        -------
        list[RetrievedChunk]
            Ranked list of retrieved document chunks.
        """
        k = top_k or self.config.top_k
        query_vector = self.model.encode(query, convert_to_numpy=True).tolist()

        # Build Qdrant filter
        must_conditions = []
        if category_filter:
            must_conditions.append(
                FieldCondition(key="category", match=MatchValue(value=category_filter))
            )
        if priority_filter:
            must_conditions.append(
                FieldCondition(key="priority", match=MatchValue(value=priority_filter))
            )
        if doc_type_filter:
            must_conditions.append(
                FieldCondition(key="doc_type", match=MatchValue(value=doc_type_filter))
            )

        qdrant_filter = Filter(must=must_conditions) if must_conditions else None

        try:
            results: list[ScoredPoint] = self.client.search(
                collection_name=self.config.collection_name,
                query_vector=query_vector,
                limit=k,
                score_threshold=self.config.score_threshold,
                query_filter=qdrant_filter,
                with_payload=True,
            )
        except Exception as exc:
            logger.error("Qdrant search failed: %s", exc)
            return []

        chunks: list[RetrievedChunk] = []
        for hit in results:
            payload = hit.payload or {}
            raw_text = payload.get("text", "")
            safe_text = mask_pii_text(raw_text) if mask_pii else raw_text
            chunks.append(
                RetrievedChunk(
                    source=payload.get("source", ""),
                    title=payload.get("title", ""),
                    category=payload.get("category", ""),
                    priority=payload.get("priority", ""),
                    escalation_risk=float(payload.get("escalation_risk", 0.0)),
                    doc_type=payload.get("doc_type", "knowledge_base"),
                    ticket_id=payload.get("ticket_id", ""),
                    customer_id=payload.get("customer_id", ""),
                    text=safe_text,
                    score=float(hit.score),
                )
            )

        logger.debug("Query '%s' returned %d results.", query, len(chunks))
        return chunks

    def format_context(self, chunks: list[RetrievedChunk]) -> str:
        """Format retrieved chunks into a single LLM-ready context string."""
        lines: list[str] = []
        for i, chunk in enumerate(chunks, start=1):
            doc_info = chunk.title or chunk.source or chunk.ticket_id
            lines.append(
                f"[{i}] [{doc_info} | {chunk.category} | {chunk.doc_type}]\n"
                f"{chunk.text}"
            )
        return "\n\n".join(lines)

    def search_and_format(
        self,
        query: str,
        top_k: int | None = None,
        mask_pii: bool = False,
        **kwargs: Any,
    ) -> str:
        """Convenience: search + format in one call for LLM context injection."""
        chunks = self.search(query, top_k=top_k, mask_pii=mask_pii, **kwargs)
        return self.format_context(chunks)

    def collection_info(self) -> dict[str, Any]:
        """Return basic stats about the configured collection."""
        try:
            info = self.client.get_collection(self.config.collection_name)
            return {
                "collection": self.config.collection_name,
                "vectors_count": info.vectors_count,
                "status": str(info.status),
            }
        except Exception as exc:
            logger.warning("Could not fetch collection info: %s", exc)
            return {"collection": self.config.collection_name, "error": str(exc)}


# ---------------------------------------------------------------------------
# Public PII masking alias (for use in API layer)
# ---------------------------------------------------------------------------

def mask_pii_text(text: str) -> str:
    """Alias for the module-level mask_pii function."""
    return mask_pii(text)
