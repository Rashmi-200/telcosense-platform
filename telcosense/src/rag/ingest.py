"""
TelcoSense - RAG Document Ingestion
=====================================
Generates synthetic Telecom KB docs, chunks them using a
RecursiveCharacterTextSplitter-style strategy, embeds with
sentence-transformers, and upserts into a local Qdrant collection
(persisted to data/processed/qdrant_db - no server required).

Usage:
  python -m src.rag.ingest                   # KB docs (default)
  python -m src.rag.ingest --mode tickets    # ticket CSV only
  python -m src.rag.ingest --mode all        # both
  python -m src.rag.ingest --recreate        # wipe and re-ingest
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import textwrap
import uuid
from pathlib import Path
from typing import Final, Iterator

# Suppress TensorFlow — must be set BEFORE sentence_transformers import
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import pandas as pd
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.models import PointStruct
from sentence_transformers import SentenceTransformer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("telcosense.rag.ingest")

# ---------------------------------------------------------------------------
# Path constants  (parents[2] = telcosense/ project root)
# ---------------------------------------------------------------------------
_PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
RAW_DIR: Final[Path] = _PROJECT_ROOT / "data" / "raw"
KB_DOCS_DIR: Final[Path] = RAW_DIR / "kb_docs"
TICKETS_CSV: Final[Path] = RAW_DIR / "tickets.csv"
QDRANT_PERSIST_DIR: Final[Path] = _PROJECT_ROOT / "data" / "processed" / "qdrant_db"

DEFAULT_KB_COLLECTION: Final[str] = "telcosense-kb"
DEFAULT_TICKET_COLLECTION: Final[str] = "telco_tickets"
DEFAULT_EMBEDDING_MODEL: Final[str] = "all-MiniLM-L6-v2"
EMBEDDING_DIM: Final[int] = 384
DEFAULT_CHUNK_SIZE: Final[int] = 400
DEFAULT_CHUNK_OVERLAP: Final[int] = 80
DEFAULT_BATCH_SIZE: Final[int] = 32


# ---------------------------------------------------------------------------
# Pydantic config
# ---------------------------------------------------------------------------


class IngestConfig(BaseModel):
    """Runtime configuration for the ingestion pipeline."""

    collection_name: str = DEFAULT_KB_COLLECTION
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    chunk_size: int = Field(default=DEFAULT_CHUNK_SIZE, ge=64)
    chunk_overlap: int = Field(default=DEFAULT_CHUNK_OVERLAP, ge=0)
    batch_size: int = Field(default=DEFAULT_BATCH_SIZE, ge=1)
    recreate_collection: bool = False
    qdrant_persist_path: Path = QDRANT_PERSIST_DIR


# ---------------------------------------------------------------------------
# Synthetic Telecom KB documents
# ---------------------------------------------------------------------------

_KB_DOCUMENTS: list[dict] = [
    {
        "filename": "5G_Router_Troubleshooting_Guide.txt",
        "category": "Network",
        "title": "5G Router Troubleshooting Guide",
        "content": textwrap.dedent("""\
            5G Router Troubleshooting Guide - TelcoSense Support

            Overview
            This guide covers common 5G home-router issues and step-by-step
            resolution procedures for TelcoSense field technicians and Tier-2
            support agents.

            1. No Internet Connectivity
            Symptoms: Router shows solid amber WAN LED, devices cannot browse.
            Resolution steps:
              Step 1. Reboot the router by pressing the reset button for 10 seconds.
              Step 2. Verify the SIM card is seated correctly in the router slot.
              Step 3. Check signal bars. Fewer than 2 bars indicates poor 5G coverage.
                 Reposition the router near a window or exterior wall.
              Step 4. Confirm APN settings: navigate to 192.168.1.1, Mobile Network,
                 APN and set to telcosense.5g with username blank and password blank.
              Step 5. If issues persist, factory-reset and re-provision via the
                 TelcoSense Mobile App under Devices, Add New Device.

            2. Slow Data Speeds
            Symptoms: Speed test shows low throughput despite full signal.
            Checklist:
              Ensure the router firmware is up-to-date via Settings, Firmware.
              Switch from NSA (Non-Standalone) to SA (Standalone) 5G if available.
              Check for local cell congestion: performance may degrade during
              peak hours from 18:00 to 22:00 local time.
              Disable 2.4 GHz Wi-Fi if not needed to reduce interference.
              Set QoS priority to Gaming or Streaming for improved latency.

            3. Intermittent Disconnections
            Root causes: Weak 5G signal with frequent handoff to 4G LTE fallback.
            Router overheating. Ensure 10 cm clearance on all sides.
            ISP-side load balancing events typically resolve within 2 hours.
            Resolution: Enable Signal Lock in advanced settings to pin to 5G NR band n78.
            Upgrade to the TelcoSense Pro 5G mesh extender for seamless roaming.

            4. Router LED Status Reference
            Green Solid: Healthy, 5G connected.
            Blue Solid: Connected via 4G LTE fallback.
            Amber Solid: WAN link failure.
            Red Solid: Hardware fault, contact support.
            White Pulsing: Firmware update in progress.

            5. Escalation Path
            Tier-1 attempts remote diagnostics via the TelcoSense NOC portal.
            If unresolved after 30 minutes, raise a P1 ticket and dispatch a
            field engineer. SLA for 5G outages: 4-hour response, 8-hour resolution.
            Support Hotline: +41 800 500 500. NOC Email: noc@telcosense.ch
        """),
    },
    {
        "filename": "Swiss_Billing_and_Tariff_Policy.txt",
        "category": "Billing",
        "title": "Swiss Billing and Tariff Policy",
        "content": textwrap.dedent("""\
            TelcoSense Swiss Billing and Tariff Policy v4.2 (2026)

            1. Invoicing Cycle
            Invoices are generated on the 1st of each month and due within
            30 days of issuance. Late payment incurs a CHF 15 administrative fee
            per outstanding invoice.

            2. Tariff Plans

            2.1 Basic Mobile CHF 29 per month
            Includes 5 GB data throttled to 1 Mbps after cap.
            200 national call minutes. 100 SMS included. No international roaming.

            2.2 Standard Mobile CHF 49 per month
            Includes 25 GB data throttled to 5 Mbps after cap.
            Unlimited national calls and SMS.
            EU zone roaming included with 15 GB. Voicemail included.

            2.3 Premium 5G CHF 89 per month
            Unlimited data with no throttling up to fair-use limit of 300 GB.
            Unlimited calls and SMS for national and EU destinations.
            Global roaming in 80+ countries.
            5G priority network access. Hotspot tethering up to 50 Mbps.

            2.4 Business Unlimited CHF 149 per month per line
            All Premium 5G benefits plus dedicated account manager.
            SLA-backed uptime guarantee of 99.9 percent. Priority NOC support.

            3. Dispute Resolution
            Customers must raise billing disputes within 60 days of invoice date.
            Disputes submitted via my.telcosense.ch, Billing, Dispute Invoice.
            Or by phone at +41 800 200 100 Monday to Friday 08:00 to 18:00.
            Resolution target: 10 business days. Credit notes issued within 5 days.

            4. Payment Methods Accepted
            Swiss IBAN bank transfer. Credit card (Visa, Mastercard, AmEx).
            PostFinance eBill. TelcoSense account credit (top-up via app).

            5. Roaming Charges outside EU zone
            USA and Canada: CHF 0.05 per MB data, CHF 1.20 per minute voice.
            Asia Pacific: CHF 0.08 per MB data, CHF 1.80 per minute voice.
            Africa and Rest: CHF 0.15 per MB data, CHF 2.50 per minute voice.

            6. Contract Cancellation
            Minimum contract term 12 months. Early termination fee:
            CHF 50 plus remaining monthly fees times 0.5. No fee after 24 months.

            7. VAT
            All prices include 8.1% Swiss VAT as required by MWSTG Art. 25.
        """),
    },
    {
        "filename": "Fibre_Optic_Connection_Manual.txt",
        "category": "Network",
        "title": "Fibre Optic Connection Installation Manual",
        "content": textwrap.dedent("""\
            Fibre Optic Connection Manual - TelcoSense Field Engineering

            1. Pre-Installation Checklist
            Confirm ONT model: TelcoSense FibreBox v3.
            Verify fibre drop cable is undamaged and properly labelled.
            Confirm customer splice point is accessible (building riser or MDF).
            Tools required: OTDR, fibre cleaver, fusion splicer, optical power meter.

            2. Physical Installation

            2.1 Fibre Entry Point
            Route the drop cable from the street cabinet to the building entry.
            Install a protective conduit at wall penetration points.
            Leave 1.5 m slack coiled at the ONT mounting location.

            2.2 Fusion Splicing
            Strip 2 cm of outer jacket from both fibres.
            Clean with 99% isopropyl alcohol wipe.
            Cleave each fibre to 1 degree angle (verify with cleaver scale).
            Splice using arc fusion: target splice loss below 0.05 dB.
            Seal splice with heat-shrink protection sleeve.

            2.3 ONT Configuration
            Connect pigtail to ONT SC/APC port.
            Power on ONT. LOS LED should go OFF within 30 seconds.
            Log into ONT admin at http://192.168.100.1 using admin and telco2026.
            Set GPON SN (serial number from device label).
            Confirm PON registration: Status, PON, Registration shows O5 Active.

            3. Testing and Certification

            3.1 OTDR Test
            Launch OTDR from building MDF end.
            Acceptable total insertion loss: below 28 dB (GPON class B+).
            Any event above 0.3 dB reflection loss indicates a problem splice.

            3.2 Speed Test
            Connect test laptop to ONT LAN port 1 (gigabit Ethernet).
            Expected speeds: Download 900 Mbps or higher. Upload 500 Mbps or higher.
            Use speedtest.telcosense.ch for calibrated results.

            4. Common Faults and Resolution
            No PON signal with LOS solid red: Fibre break or dirty connector.
            Run OTDR trace then clean or replace connector.
            High BER with LOS blinking: Splice loss too high.
            Re-splice targeting below 0.1 dB.
            ONT not registering with Auth LED off: Wrong GPON SN in provisioning.
            Correct SN in OSS portal.
            Slow speed with Speed LED amber: Bandwidth contention.
            Escalate to NOC for DSL port check.

            5. Escalation
            Fibre faults unresolved after 1 hour: raise P2 ticket in ServiceNow.
            Include OTDR trace file (.sor) and optical power measurement screenshot.
            SLA: residential 24 h restore; business 4 h restore.
        """),
    },
    {
        "filename": "Customer_Onboarding_and_SIM_Activation_Guide.txt",
        "category": "Hardware",
        "title": "Customer Onboarding and SIM Activation Guide",
        "content": textwrap.dedent("""\
            Customer Onboarding and SIM Activation Guide - TelcoSense

            1. SIM Card Types
            TelcoSense offers three physical formats and eSIM.
            Standard SIM (2FF) for legacy devices.
            Micro-SIM (3FF) for older smartphones.
            Nano-SIM (4FF) for current smartphones.
            eSIM (eUICC) for iPhone XS+, Samsung S20+, Pixel 3+ and later.

            2. Physical SIM Activation
            Step 1. Insert SIM into device tray (ensure correct orientation).
            Step 2. Power on device. TelcoSense should appear as carrier.
            Step 3. Open TelcoSense App, go to Account, Activate SIM.
            Enter the 20-digit ICCID from SIM card printed below barcode.
            Step 4. Await confirmation SMS within 5 minutes.
            Step 5. Restart device to finalize network registration.

            3. eSIM Activation
            In TelcoSense App go to Activate, eSIM, Download Profile.
            Follow device-specific prompts to install eSIM profile.
            Set TelcoSense as default data line.
            Activation code EID format is 32 hexadecimal digits.

            4. Number Porting (MNP)
            Submit MNP request at least 3 business days before desired date.
            Required documents: existing carrier account number and last invoice copy.
            Port completion within 24 hours of confirmed request.
            Service interruption window is 15 minutes or less during port.

            5. Troubleshooting Activation

            SIM Not Recognised:
            Try in another device to isolate hardware vs SIM fault.
            Clean contacts with dry cloth. Avoid isopropyl on nano-SIM.
            Replace SIM at any TelcoSense retail point free of charge.

            No Network Registration:
            Manual network selection: Settings, Mobile, Network, TelcoSense.
            If no TelcoSense network shown check device band compatibility.
            Required bands: B1/B3/B7/B28 for 4G and n78/n1 for 5G.

            eSIM Profile Fails to Download:
            Ensure Wi-Fi connectivity before download.
            QR code valid for 24 hours. Request new code via app if expired.
            Factory reset eSIM: Settings, General, Transfer Reset, Erase eSIM.

            6. Data APN Settings (manual)
            APN: telcosense.ch
            Username: blank
            Password: blank
            MCC: 228
            MNC: 08
            Authentication: None

            7. Support
            For activation failures contact +41 800 300 300 or
            support@telcosense.ch. 24/7 chat available in the app.
        """),
    },
    {
        "filename": "Network_Outage_Incident_Playbook.txt",
        "category": "Network",
        "title": "Network Outage Incident Management Playbook",
        "content": textwrap.dedent("""\
            Network Outage Incident Management Playbook - TelcoSense NOC

            1. Severity Classification
            P1 Critical: More than 10000 customers affected.
            Max response 15 minutes. Max resolution 2 hours.
            P2 Major: 1000 to 10000 customers affected.
            Max response 30 minutes. Max resolution 4 hours.
            P3 Moderate: 100 to 1000 customers affected.
            Max response 1 hour. Max resolution 8 hours.
            P4 Minor: Fewer than 100 customers affected.
            Max response 4 hours. Max resolution 24 hours.

            2. P1 Incident Procedure
            Step 1. NOC engineer declares P1 via PagerDuty alert to on-call team.
            Step 2. Incident Commander assigned within 5 minutes.
            Step 3. War-room bridge opened at teams.telcosense.ch/noc-p1.
            Step 4. Customer communications: status.telcosense.ch updated every 15 minutes.
            Step 5. Executive notification (CTO/CEO) if outage exceeds 30 minutes.

            3. Diagnostic Toolkit
            SCOM Dashboard: Real-time RAN KPIs including PRB utilization, CQI, SINR.
            Kibana Logs: Filter by service core-network AND level ERROR.
            Grafana: Panels, 5G Core, Session Success Rate target above 99.5%.
            CLI Tools: telco-cli ping-node checks ICMP reachability.
            telco-cli dump-alarms shows critical alerts.
            telco-cli show-ran-stats shows RAN statistics per cell.

            4. Common Outage Root Causes

            Core Network AMF SMF UPF:
            Container pod crash. Check with kubectl get pods -n 5g-core.
            Mitigation: kubectl rollout restart deployment/amf.

            Radio gNB:
            Overheating: check cabinet temperature sensors. Alarm threshold above 55 degrees C.
            Software bug: rollback to previous gNB build via OSS GUI.

            Transport IP MPLS:
            BGP session drop: verify with show bgp neighbors grep Idle.
            Reroute traffic via secondary MPLS LSP.

            5. Post-Incident Review PIR
            PIR required for all P1 and P2 incidents within 5 business days.
            Template in Confluence NOC PIR Template v3.
            Sections: Timeline, Root Cause, Impact, Preventive Actions.

            6. Escalation Contacts
            NOC Lead: noc-lead@telcosense.ch available 24/7.
            Network Director: nd@telcosense.ch during business hours.
            Vendor TAC Ericsson: +46 10 719 0000 available 24/7 for P1.
            Vendor TAC Nokia: +1 844 665 2001 available 24/7 for P1.
        """),
    },
]


# ---------------------------------------------------------------------------
# RecursiveCharacterTextSplitter-style chunking
# ---------------------------------------------------------------------------

_SEPARATORS: list[str] = ["\n\n", "\n", ". ", " ", ""]


def _recursive_split(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    separators: list[str] | None = None,
) -> list[str]:
    """
    Mimics LangChain RecursiveCharacterTextSplitter without external dep.
    Tries each separator in order; falls back to character-level splitting.
    Returns a list of non-empty string chunks.
    """
    seps = separators if separators is not None else _SEPARATORS
    final_chunks: list[str] = []

    sep_used = ""
    remaining_seps: list[str] = []
    for i, sep in enumerate(seps):
        if sep == "" or sep in text:
            sep_used = sep
            remaining_seps = seps[i + 1 :]
            break

    raw_splits = text.split(sep_used) if sep_used else [text]
    join_sep = sep_used if sep_used not in ("", " ") else " "

    current: list[str] = []
    current_len = 0

    for split in raw_splits:
        split_len = len(split)
        if split_len > chunk_size:
            if current:
                merged = join_sep.join(current).strip()
                if merged:
                    final_chunks.append(merged)
                current, current_len = [], 0
            sub_chunks = _recursive_split(split, chunk_size, chunk_overlap, remaining_seps)
            final_chunks.extend(sub_chunks)
            continue

        if current_len + split_len + len(join_sep) > chunk_size and current:
            merged = join_sep.join(current).strip()
            if merged:
                final_chunks.append(merged)
            while current and current_len > chunk_overlap:
                removed = current.pop(0)
                current_len -= len(removed) + len(join_sep)

        current.append(split)
        current_len += split_len + len(join_sep)

    if current:
        merged = join_sep.join(current).strip()
        if merged:
            final_chunks.append(merged)

    return [c for c in final_chunks if c.strip()]


def _batch(iterable: list, size: int) -> Iterator[list]:
    for i in range(0, len(iterable), size):
        yield iterable[i : i + size]


# ---------------------------------------------------------------------------
# KB document generation
# ---------------------------------------------------------------------------


def generate_kb_documents(kb_dir: Path = KB_DOCS_DIR) -> list[Path]:
    """Write synthetic KB .txt documents to kb_dir and return their paths."""
    kb_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for doc in _KB_DOCUMENTS:
        p = kb_dir / doc["filename"]
        p.write_text(doc["content"], encoding="utf-8")
        written.append(p)
        logger.info("KB doc written: %s", p.name)
    logger.info("Generated %d KB documents in %s.", len(written), kb_dir)
    return written


# ---------------------------------------------------------------------------
# Qdrant helpers (local on-disk persistence)
# ---------------------------------------------------------------------------


def _get_qdrant_client(persist_path: Path) -> QdrantClient:
    """Return a Qdrant client backed by on-disk storage."""
    persist_path.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(persist_path))


def _ensure_collection(client: QdrantClient, name: str, recreate: bool) -> None:
    existing = {c.name for c in client.get_collections().collections}
    if recreate and name in existing:
        client.delete_collection(name)
        existing.discard(name)
    if name not in existing:
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
        logger.info("Created Qdrant collection '%s'.", name)
    else:
        logger.info("Qdrant collection '%s' already exists - appending.", name)


# ---------------------------------------------------------------------------
# Core ingestion: KB documents
# ---------------------------------------------------------------------------


def ingest_kb_documents(
    config: IngestConfig,
    kb_dir: Path = KB_DOCS_DIR,
    model: SentenceTransformer | None = None,
) -> int:
    """Chunk, embed, and upsert KB .txt documents into Qdrant."""
    doc_paths = sorted(kb_dir.glob("*.txt"))
    if not doc_paths:
        logger.warning("No .txt files found in %s - generating first.", kb_dir)
        doc_paths = generate_kb_documents(kb_dir)

    logger.info("Found %d KB documents to ingest.", len(doc_paths))
    meta_map: dict[str, dict] = {d["filename"]: d for d in _KB_DOCUMENTS}

    points: list[PointStruct] = []
    for doc_path in doc_paths:
        text = doc_path.read_text(encoding="utf-8")
        meta = meta_map.get(doc_path.name, {})
        chunks = _recursive_split(text, config.chunk_size, config.chunk_overlap)
        for chunk in chunks:
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    payload={
                        "source": doc_path.name,
                        "title": meta.get("title", doc_path.stem),
                        "category": meta.get("category", "General"),
                        "doc_type": "knowledge_base",
                        "text": chunk,
                    },
                    vector=[],
                )
            )

    logger.info("Generated %d text chunks from %d KB docs.", len(points), len(doc_paths))

    client = _get_qdrant_client(config.qdrant_persist_path)
    _ensure_collection(client, config.collection_name, config.recreate_collection)

    _model = model or SentenceTransformer(config.embedding_model)
    logger.info("Embedding model '%s' ready.", config.embedding_model)

    total = 0
    for batch_pts in _batch(points, config.batch_size):
        texts = [p.payload["text"] for p in batch_pts]
        embeddings = _model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        for j, pt in enumerate(batch_pts):
            pt.vector = embeddings[j].tolist()
        client.upsert(collection_name=config.collection_name, points=batch_pts)
        total += len(batch_pts)

    logger.info(
        "KB ingestion complete - %d vectors upserted into '%s'.",
        total, config.collection_name,
    )
    return total


# ---------------------------------------------------------------------------
# Core ingestion: Support tickets
# ---------------------------------------------------------------------------


def ingest_tickets(
    config: IngestConfig,
    tickets_csv: Path = TICKETS_CSV,
    model: SentenceTransformer | None = None,
) -> int:
    """Chunk, embed, and upsert support ticket CSV into Qdrant."""
    logger.info("Loading tickets from %s ...", tickets_csv)
    df = pd.read_csv(tickets_csv)
    logger.info("Loaded %d tickets.", len(df))

    ticket_collection = DEFAULT_TICKET_COLLECTION
    client = _get_qdrant_client(config.qdrant_persist_path)
    _ensure_collection(client, ticket_collection, config.recreate_collection)

    _model = model or SentenceTransformer(config.embedding_model)

    points: list[PointStruct] = []
    for _, row in df.iterrows():
        chunks = _recursive_split(
            str(row["ticket_text"]), config.chunk_size, config.chunk_overlap
        )
        for chunk in chunks:
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    payload={
                        "ticket_id": str(row["ticket_id"]),
                        "customer_id": str(row["customer_id"]),
                        "category": str(row["category"]),
                        "priority": str(row["priority"]),
                        "escalation_risk": float(row["escalation_risk"]),
                        "doc_type": "ticket",
                        "text": chunk,
                    },
                    vector=[],
                )
            )

    logger.info("Generated %d chunks from %d tickets.", len(points), len(df))
    total = 0
    for batch_pts in _batch(points, config.batch_size):
        texts = [p.payload["text"] for p in batch_pts]
        embeddings = _model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        for j, pt in enumerate(batch_pts):
            pt.vector = embeddings[j].tolist()
        client.upsert(collection_name=ticket_collection, points=batch_pts)
        total += len(batch_pts)

    logger.info(
        "Ticket ingestion complete - %d vectors upserted into '%s'.",
        total, ticket_collection,
    )
    return total


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TelcoSense RAG ingestion pipeline.")
    parser.add_argument(
        "--mode", choices=["kb", "tickets", "all"], default="kb",
        help="Ingestion mode: kb (default), tickets, or all.",
    )
    parser.add_argument("--recreate", action="store_true", help="Recreate Qdrant collections.")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    config = IngestConfig(
        recreate_collection=args.recreate,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        batch_size=args.batch_size,
    )

    generate_kb_documents()

    model = SentenceTransformer(config.embedding_model)
    logger.info("Loaded embedding model: %s", config.embedding_model)

    if args.mode in ("kb", "all"):
        kb_total = ingest_kb_documents(config, model=model)
        logger.info("KB ingestion done - %d vectors.", kb_total)

    if args.mode in ("tickets", "all"):
        ticket_total = ingest_tickets(config, model=model)
        logger.info("Ticket ingestion done - %d vectors.", ticket_total)

    logger.info("=" * 60)
    logger.info("RAG ingestion pipeline complete.")
    logger.info("  Collection : %s", config.collection_name)
    logger.info("  Qdrant DB  : %s", config.qdrant_persist_path)
    logger.info("=" * 60)


if __name__ == "__main__":
    main(sys.argv[1:])
