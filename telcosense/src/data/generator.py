"""
TelcoSense — Data Generator
============================
Responsibility:
  1. Download the public IBM Telco Customer Churn dataset (with a graceful
     synthetic fallback if the network is unavailable).
  2. Generate 2,000 synthetic support-ticket logs mapped to customer IDs.

Usage:
  python -m src.data.generator
  python -m src.data.generator --tickets-only
  python -m src.data.generator --churn-only
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
import time
from pathlib import Path
from typing import Final, Sequence

import numpy as np
import pandas as pd
import requests
from faker import Faker
from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("telcosense.generator")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IBM_CHURN_URL: Final[str] = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)
RAW_DIR: Final[Path] = Path(__file__).resolve().parents[2] / "data" / "raw"
CHURN_CSV: Final[Path] = RAW_DIR / "ibm_churn.csv"
TICKETS_CSV: Final[Path] = RAW_DIR / "tickets.csv"

SYNTHETIC_ROWS: Final[int] = 2_000
TICKET_ROWS: Final[int] = 2_000
RANDOM_SEED: Final[int] = 42

TICKET_CATEGORIES: Final[Sequence[str]] = ["Billing", "Network", "Hardware"]
TICKET_PRIORITIES: Final[Sequence[str]] = ["Low", "High"]

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class ChurnRecord(BaseModel):
    """Single row of the IBM Telco Churn dataset."""

    customerID: str
    gender: str
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: str
    Dependents: str
    tenure: int = Field(ge=0)
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float = Field(ge=0.0)
    TotalCharges: float = Field(ge=0.0)
    Churn: str

    @field_validator("Churn")
    @classmethod
    def churn_must_be_binary(cls, v: str) -> str:
        if v not in {"Yes", "No"}:
            raise ValueError(f"Churn must be 'Yes' or 'No', got: {v!r}")
        return v


class TicketRecord(BaseModel):
    """Single row of the synthetic support-ticket dataset."""

    ticket_id: str
    customer_id: str
    ticket_text: str
    category: str
    priority: str
    escalation_risk: float = Field(ge=0.0, le=1.0)

    @field_validator("category")
    @classmethod
    def category_valid(cls, v: str) -> str:
        if v not in TICKET_CATEGORIES:
            raise ValueError(f"category must be one of {TICKET_CATEGORIES}")
        return v

    @field_validator("priority")
    @classmethod
    def priority_valid(cls, v: str) -> str:
        if v not in TICKET_PRIORITIES:
            raise ValueError(f"priority must be one of {TICKET_PRIORITIES}")
        return v


# ---------------------------------------------------------------------------
# IBM Churn Dataset helpers
# ---------------------------------------------------------------------------


def _download_ibm_churn(url: str, timeout: int = 30) -> pd.DataFrame:
    """Attempt to download the IBM Telco Churn CSV from *url*."""
    logger.info("Downloading IBM Telco Churn CSV from %s …", url)
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    from io import StringIO

    df = pd.read_csv(StringIO(resp.text))
    logger.info("Downloaded %d rows, %d columns.", len(df), len(df.columns))
    return df


def _clean_ibm_churn(df: pd.DataFrame) -> pd.DataFrame:
    """Minimal cleaning: coerce TotalCharges and drop rows with nulls."""
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    before = len(df)
    df = df.dropna(subset=["TotalCharges"])
    dropped = before - len(df)
    if dropped:
        logger.info("Dropped %d rows with non-numeric TotalCharges.", dropped)
    df["Churn"] = df["Churn"].str.strip()
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Synthetic Churn Fallback
# ---------------------------------------------------------------------------

_YES_NO: Final[Sequence[str]] = ["Yes", "No"]
_CONTRACTS: Final[Sequence[str]] = ["Month-to-month", "One year", "Two year"]
_INTERNET: Final[Sequence[str]] = ["DSL", "Fiber optic", "No"]
_PAYMENT: Final[Sequence[str]] = [
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)",
]
_MULTIPLE_LINES: Final[Sequence[str]] = ["No phone service", "No", "Yes"]
_ONLINE_FEATURE: Final[Sequence[str]] = ["No internet service", "No", "Yes"]


def _generate_synthetic_churn(n: int, seed: int) -> pd.DataFrame:
    """Generate *n* synthetic rows that mirror IBM Telco Churn schema."""
    rng = np.random.default_rng(seed)
    fake = Faker()
    fake.seed_instance(seed)

    logger.info("Generating %d synthetic churn rows (seed=%d)…", n, seed)

    customer_ids: list[str] = [
        f"CUST-{str(i).zfill(6)}" for i in range(1, n + 1)
    ]
    tenure: np.ndarray = rng.integers(0, 73, size=n)
    monthly_charges: np.ndarray = rng.uniform(18.0, 120.0, size=n)
    total_charges: np.ndarray = monthly_charges * tenure + rng.uniform(0, 50, size=n)
    total_charges = np.clip(total_charges, 0.0, None)

    # Churn probability correlated with contract type
    contracts: list[str] = rng.choice(_CONTRACTS, size=n).tolist()  # type: ignore[assignment]
    churn_prob: np.ndarray = np.where(
        np.array(contracts) == "Month-to-month",
        rng.uniform(0.3, 0.6, size=n),
        rng.uniform(0.05, 0.2, size=n),
    )
    churn: list[str] = ["Yes" if p > rng.random() else "No" for p in churn_prob]

    records: list[dict] = []
    for i in range(n):
        records.append(
            {
                "customerID": customer_ids[i],
                "gender": rng.choice(["Male", "Female"]),
                "SeniorCitizen": int(rng.choice([0, 1], p=[0.84, 0.16])),
                "Partner": rng.choice(_YES_NO),
                "Dependents": rng.choice(_YES_NO),
                "tenure": int(tenure[i]),
                "PhoneService": rng.choice(_YES_NO),
                "MultipleLines": rng.choice(_MULTIPLE_LINES),
                "InternetService": rng.choice(_INTERNET),
                "OnlineSecurity": rng.choice(_ONLINE_FEATURE),
                "OnlineBackup": rng.choice(_ONLINE_FEATURE),
                "DeviceProtection": rng.choice(_ONLINE_FEATURE),
                "TechSupport": rng.choice(_ONLINE_FEATURE),
                "StreamingTV": rng.choice(_ONLINE_FEATURE),
                "StreamingMovies": rng.choice(_ONLINE_FEATURE),
                "Contract": contracts[i],
                "PaperlessBilling": rng.choice(_YES_NO),
                "PaymentMethod": rng.choice(_PAYMENT),
                "MonthlyCharges": round(float(monthly_charges[i]), 2),
                "TotalCharges": round(float(total_charges[i]), 2),
                "Churn": churn[i],
            }
        )

    return pd.DataFrame(records)


def generate_churn_dataset(
    output_path: Path = CHURN_CSV,
    url: str = IBM_CHURN_URL,
    synthetic_rows: int = SYNTHETIC_ROWS,
    seed: int = RANDOM_SEED,
    force_synthetic: bool = False,
) -> Path:
    """
    Download the IBM Telco Churn CSV.  If the download fails (network error,
    HTTP error, timeout), fall back to generating *synthetic_rows* rows.

    Parameters
    ----------
    output_path:
        Destination CSV file.
    url:
        Remote URL for the IBM dataset.
    synthetic_rows:
        Number of rows to generate when the download fails.
    seed:
        Random seed for reproducibility.
    force_synthetic:
        Skip network attempt and go straight to synthetic generation.

    Returns
    -------
    Path
        Absolute path to the saved CSV file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df: pd.DataFrame

    if not force_synthetic:
        try:
            df = _download_ibm_churn(url)
            df = _clean_ibm_churn(df)
            source = "IBM Telco (downloaded)"
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Download failed (%s). Falling back to synthetic generation.", exc
            )
            df = _generate_synthetic_churn(synthetic_rows, seed)
            source = "synthetic (fallback)"
    else:
        df = _generate_synthetic_churn(synthetic_rows, seed)
        source = "synthetic (forced)"

    df.to_csv(output_path, index=False)
    logger.info(
        "Churn dataset saved → %s  [source=%s, rows=%d]",
        output_path,
        source,
        len(df),
    )
    return output_path.resolve()


# ---------------------------------------------------------------------------
# Ticket Dataset
# ---------------------------------------------------------------------------

# Templates for each category — deterministic variety without LLMs
_BILLING_TEMPLATES: Final[list[str]] = [
    "My bill for {month} seems incorrect. I was charged ${amount} but my plan is ${plan}.",
    "I noticed a duplicate charge of ${amount} on my account dated {date}.",
    "I never received my invoice for {month}. Please resend to {email}.",
    "My auto-payment failed. Please confirm the status for ${amount} due on {date}.",
    "I want to dispute a late fee of ${amount} that appeared on my latest bill.",
    "Why does my bill show an extra data overage charge of ${amount} this cycle?",
    "I was promised a discount of ${amount}/month but my bill still shows full price.",
    "Please help me understand my itemised charges totalling ${amount}.",
]

_NETWORK_TEMPLATES: Final[list[str]] = [
    "I've been experiencing slow internet speeds in {area} for the past {days} days.",
    "My connection drops every {hours} hours. I've already restarted my router.",
    "Video calls keep freezing. Upload speed is only {speed} Mbps on a {plan} plan.",
    "Total outage in {area} since {date}. When will service be restored?",
    "My 5G signal is extremely weak even with full bars showing on my handset.",
    "Packet loss is at {loss}% causing issues with online gaming and work VPN.",
    "The ping to international servers is {ping}ms — far worse than usual.",
    "WiFi keeps disconnecting on all {devices} devices. ISP modem lights look normal.",
]

_HARDWARE_TEMPLATES: Final[list[str]] = [
    "My set-top box (model {model}) reboots randomly every {hours} hours.",
    "The optical port on my router is damaged and blinking {color}.",
    "I received a faulty ONT device. It's not syncing and LED shows red.",
    "My modem ({model}) isn't detected by the technician's mobile app.",
    "One of the ethernet ports on the router stopped working after a power cut.",
    "The battery backup on my router depletes in {minutes} minutes — it used to last hours.",
    "Screen on my managed WiFi extender is cracked after delivery. Need replacement.",
    "My SIM card appears to be defective — ICCID shows error on two different handsets.",
]

_TEMPLATES_BY_CATEGORY: Final[dict[str, list[str]]] = {
    "Billing": _BILLING_TEMPLATES,
    "Network": _NETWORK_TEMPLATES,
    "Hardware": _HARDWARE_TEMPLATES,
}


def _render_ticket_text(category: str, fake: Faker, rng: random.Random) -> str:
    """Fill in a template for the given *category* with fake values."""
    template = rng.choice(_TEMPLATES_BY_CATEGORY[category])
    replacements: dict[str, str] = {
        "{month}": fake.month_name(),
        "{amount}": str(rng.randint(10, 500)),
        "{plan}": str(rng.randint(30, 200)),
        "{date}": fake.date_this_year().strftime("%d %b %Y"),
        "{email}": fake.email(),
        "{area}": fake.city(),
        "{days}": str(rng.randint(1, 30)),
        "{hours}": str(rng.randint(1, 48)),
        "{speed}": str(rng.randint(1, 20)),
        "{loss}": str(rng.randint(5, 50)),
        "{ping}": str(rng.randint(100, 800)),
        "{devices}": str(rng.randint(2, 10)),
        "{model}": f"XG-{rng.randint(1000, 9999)}",
        "{color}": rng.choice(["amber", "red", "orange"]),
        "{minutes}": str(rng.randint(5, 30)),
    }
    text = template
    for placeholder, value in replacements.items():
        text = text.replace(placeholder, value)
    return text


def generate_ticket_dataset(
    output_path: Path = TICKETS_CSV,
    n: int = TICKET_ROWS,
    churn_csv: Path = CHURN_CSV,
    seed: int = RANDOM_SEED,
) -> Path:
    """
    Generate *n* synthetic support-ticket rows.

    Customer IDs are sampled from *churn_csv* when available; otherwise
    deterministic CUST-XXXXXX IDs are generated.

    Parameters
    ----------
    output_path:
        Destination CSV file.
    n:
        Number of ticket rows to generate.
    churn_csv:
        Path to the churn CSV to source customer IDs.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    Path
        Absolute path to the saved CSV file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rng_py = random.Random(seed)
    rng_np = np.random.default_rng(seed)
    fake = Faker()
    fake.seed_instance(seed)

    # Source customer IDs
    customer_ids: list[str]
    if churn_csv.exists():
        try:
            churn_df = pd.read_csv(churn_csv, usecols=["customerID"])
            customer_ids = churn_df["customerID"].tolist()
            logger.info(
                "Loaded %d customer IDs from %s.", len(customer_ids), churn_csv
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not read churn CSV (%s); generating IDs.", exc)
            customer_ids = [f"CUST-{str(i).zfill(6)}" for i in range(1, n + 1)]
    else:
        logger.info("Churn CSV not found; generating synthetic customer IDs.")
        customer_ids = [f"CUST-{str(i).zfill(6)}" for i in range(1, n + 1)]

    # Sample categories and priorities with realistic imbalance
    categories: list[str] = rng_py.choices(
        TICKET_CATEGORIES,
        weights=[0.40, 0.40, 0.20],  # Billing/Network dominant, Hardware less common
        k=n,
    )
    priorities: list[str] = rng_py.choices(
        TICKET_PRIORITIES,
        weights=[0.65, 0.35],  # More Low than High priority
        k=n,
    )

    # Escalation risk: higher for High priority & certain categories
    base_risk = rng_np.uniform(0.05, 0.4, size=n)
    priority_bump = np.array([0.3 if p == "High" else 0.0 for p in priorities])
    category_bump = np.array(
        [0.1 if c == "Network" else 0.05 for c in categories]
    )
    escalation_risk: np.ndarray = np.clip(
        base_risk + priority_bump + category_bump, 0.0, 1.0
    )

    records: list[dict] = []
    for i in range(n):
        cid = rng_py.choice(customer_ids)
        category = categories[i]
        ticket: dict = {
            "ticket_id": f"TKT-{str(i + 1).zfill(6)}",
            "customer_id": cid,
            "ticket_text": _render_ticket_text(category, fake, rng_py),
            "category": category,
            "priority": priorities[i],
            "escalation_risk": round(float(escalation_risk[i]), 4),
        }
        records.append(ticket)

    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    logger.info(
        "Ticket dataset saved → %s  [rows=%d]",
        output_path,
        len(df),
    )
    return output_path.resolve()


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def validate_churn_sample(df: pd.DataFrame, n_samples: int = 5) -> None:
    """Validate a sample of churn rows against the Pydantic schema."""
    errors: list[str] = []
    sample = df.sample(min(n_samples, len(df)), random_state=RANDOM_SEED)
    for idx, row in sample.iterrows():
        try:
            ChurnRecord(**row.to_dict())
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Row {idx}: {exc}")
    if errors:
        logger.warning("Churn validation issues:\n%s", "\n".join(errors))
    else:
        logger.info("Churn sample validation passed (%d rows).", len(sample))


def validate_ticket_sample(df: pd.DataFrame, n_samples: int = 5) -> None:
    """Validate a sample of ticket rows against the Pydantic schema."""
    errors: list[str] = []
    sample = df.sample(min(n_samples, len(df)), random_state=RANDOM_SEED)
    for idx, row in sample.iterrows():
        try:
            TicketRecord(**row.to_dict())
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Row {idx}: {exc}")
    if errors:
        logger.warning("Ticket validation issues:\n%s", "\n".join(errors))
    else:
        logger.info("Ticket sample validation passed (%d rows).", len(sample))


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TelcoSense data generator — churn & support tickets."
    )
    parser.add_argument(
        "--churn-only",
        action="store_true",
        help="Generate only the churn dataset.",
    )
    parser.add_argument(
        "--tickets-only",
        action="store_true",
        help="Generate only the ticket dataset.",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Force synthetic churn generation (skip download).",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=SYNTHETIC_ROWS,
        help=f"Number of rows for synthetic churn (default: {SYNTHETIC_ROWS}).",
    )
    parser.add_argument(
        "--ticket-rows",
        type=int,
        default=TICKET_ROWS,
        help=f"Number of ticket rows (default: {TICKET_ROWS}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help=f"Random seed (default: {RANDOM_SEED}).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    start = time.perf_counter()

    if not args.tickets_only:
        churn_path = generate_churn_dataset(
            force_synthetic=args.synthetic,
            synthetic_rows=args.rows,
            seed=args.seed,
        )
        churn_df = pd.read_csv(churn_path)
        validate_churn_sample(churn_df)

    if not args.churn_only:
        ticket_path = generate_ticket_dataset(
            n=args.ticket_rows,
            seed=args.seed,
        )
        ticket_df = pd.read_csv(ticket_path)
        validate_ticket_sample(ticket_df)

    elapsed = time.perf_counter() - start
    logger.info("Data generation complete in %.2fs.", elapsed)


if __name__ == "__main__":
    main(sys.argv[1:])
