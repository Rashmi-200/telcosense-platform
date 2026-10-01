"""
TelcoSense — Feature Store
===========================
Responsibility:
  - Load raw IBM Telco Churn CSV & Support Tickets CSV from data/raw/.
  - Encode categorical variables, impute missing values.
  - Compute derived features (tenure bands, charge_per_tenure, etc.).
  - Perform train/test split flagging.
  - Save processed Parquet feature datasets to data/processed/:
      - churn_processed.parquet
      - tickets_processed.parquet

This module is intentionally kept stateless — all transformations are
deterministic functions that can be called from training pipelines,
APIs, and the Airflow retraining DAG.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger("telcosense.feature_store")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RAW_DIR: Final[Path] = Path(__file__).resolve().parents[2] / "data" / "raw"
PROCESSED_DIR: Final[Path] = Path(__file__).resolve().parents[2] / "data" / "processed"

CHURN_CSV: Final[Path] = RAW_DIR / "ibm_churn.csv"
TICKETS_CSV: Final[Path] = RAW_DIR / "tickets.csv"

CHURN_PROCESSED_PARQUET: Final[Path] = PROCESSED_DIR / "churn_processed.parquet"
TICKETS_PROCESSED_PARQUET: Final[Path] = PROCESSED_DIR / "tickets_processed.parquet"
FEATURES_PARQUET: Final[Path] = PROCESSED_DIR / "features.parquet"  # Legacy/Alias

BINARY_YES_NO_COLS: Final[list[str]] = [
    "Partner", "Dependents", "PhoneService", "PaperlessBilling", "Churn",
]
MULTI_VALUE_COLS: Final[list[str]] = [
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaymentMethod",
]

TICKET_CATEGORIES: Final[list[str]] = ["Billing", "Network", "Hardware"]
TICKET_PRIORITIES: Final[list[str]] = ["Low", "High"]


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------


class ProcessedChurnFeatures(BaseModel):
    """Schema for a single row after churn feature engineering."""

    customerID: str
    gender_encoded: int = Field(ge=0, le=1)
    SeniorCitizen: int = Field(ge=0, le=1)
    tenure: int = Field(ge=0)
    MonthlyCharges: float = Field(ge=0.0)
    TotalCharges: float = Field(ge=0.0)
    charge_per_tenure: float
    tenure_band: int = Field(ge=0, le=4)
    Churn_label: int = Field(ge=0, le=1)
    is_train: int = Field(ge=0, le=1)

    @field_validator("charge_per_tenure")
    @classmethod
    def finite_charge(cls, v: float) -> float:
        if not np.isfinite(v):
            return 0.0
        return v


class ProcessedTicketFeatures(BaseModel):
    """Schema for a single row after ticket feature engineering."""

    ticket_id: str
    customer_id: str
    ticket_text: str
    category: str
    category_label: int = Field(ge=0, le=2)
    priority: str
    priority_label: int = Field(ge=0, le=1)
    escalation_risk: float = Field(ge=0.0, le=1.0)
    is_train: int = Field(ge=0, le=1)


# ---------------------------------------------------------------------------
# Transformation Helpers
# ---------------------------------------------------------------------------


def _encode_yes_no(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Encode Yes/No columns as 1/0 integers."""
    for col in columns:
        if col in df.columns:
            df[col] = df[col].map({"Yes": 1, "No": 0}).fillna(0).astype(int)
    return df


def _encode_gender(df: pd.DataFrame) -> pd.DataFrame:
    df["gender_encoded"] = (df["gender"] == "Female").astype(int)
    return df


def _encode_multi_value(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Label-encode multi-value categorical columns."""
    for col in columns:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].fillna("Unknown").astype(str))
    return df


def _impute_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """Impute TotalCharges = MonthlyCharges × tenure for new customers."""
    mask = df["TotalCharges"].isna() | (df["TotalCharges"] == 0)
    df.loc[mask, "TotalCharges"] = (
        df.loc[mask, "MonthlyCharges"] * df.loc[mask, "tenure"]
    )
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    return df


def _derive_churn_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features to the churn dataframe."""
    # Charge-per-month-of-tenure ratio (guard zero-tenure)
    df["charge_per_tenure"] = np.where(
        df["tenure"] > 0,
        df["TotalCharges"] / df["tenure"],
        df["MonthlyCharges"],
    )

    # Tenure band: 0-12m → 0, 13-24 → 1, 25-36 → 2, 37-48 → 3, 49+ → 4
    bins = [0, 12, 24, 36, 48, np.inf]
    labels = [0, 1, 2, 3, 4]
    df["tenure_band"] = pd.cut(
        df["tenure"], bins=bins, labels=labels, include_lowest=True
    ).astype(int)

    # Rename target for downstream clarity
    if "Churn" in df.columns and "Churn_label" not in df.columns:
        df["Churn_label"] = df["Churn"]

    return df


# ---------------------------------------------------------------------------
# Public Feature Store Pipeline APIs
# ---------------------------------------------------------------------------


def process_churn_features(
    churn_csv: Path = CHURN_CSV,
    output_parquet: Path = CHURN_PROCESSED_PARQUET,
    test_size: float = 0.2,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Process raw IBM Churn CSV into a Parquet feature dataset.
    Includes categorical encoding, charge_per_tenure, tenure_band,
    and a stratified train/test split flag (`is_train`).
    """
    logger.info("Processing Churn Data from %s …", churn_csv)
    df = pd.read_csv(churn_csv)
    logger.info("Loaded %d raw churn records.", len(df))

    df = _impute_total_charges(df)
    df = _encode_yes_no(df, BINARY_YES_NO_COLS)
    df = _encode_gender(df)
    df = _encode_multi_value(df, MULTI_VALUE_COLS)
    df = _derive_churn_features(df)

    # Drop original gender column if superseded by gender_encoded
    df = df.drop(columns=["gender"], errors="ignore")

    # Add train/test split column
    train_idx, test_idx = train_test_split(
        df.index,
        test_size=test_size,
        stratify=df["Churn_label"],
        random_state=seed,
    )
    df["is_train"] = 0
    df.loc[train_idx, "is_train"] = 1

    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_parquet, index=False)
    # Also save to legacy FEATURES_PARQUET path for backward compatibility
    df.to_parquet(FEATURES_PARQUET, index=False)

    logger.info(
        "Saved processed churn dataset → %s  [rows=%d, train=%d, test=%d]",
        output_parquet, len(df), (df["is_train"] == 1).sum(), (df["is_train"] == 0).sum(),
    )
    return df


def process_ticket_features(
    tickets_csv: Path = TICKETS_CSV,
    output_parquet: Path = TICKETS_PROCESSED_PARQUET,
    test_size: float = 0.2,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Process raw support tickets CSV into a Parquet feature dataset.
    Includes category and priority encodings, text normalization,
    and a stratified train/test split flag (`is_train`).
    """
    logger.info("Processing Ticket Data from %s …", tickets_csv)
    df = pd.read_csv(tickets_csv)
    logger.info("Loaded %d raw ticket records.", len(df))

    # Categorical encodings
    category_map = {cat: idx for idx, cat in enumerate(TICKET_CATEGORIES)}
    priority_map = {pri: idx for idx, pri in enumerate(TICKET_PRIORITIES)}

    df["category_label"] = df["category"].map(category_map).fillna(0).astype(int)
    df["priority_label"] = df["priority"].map(priority_map).fillna(0).astype(int)
    df["ticket_text"] = df["ticket_text"].fillna("").astype(str).str.strip()

    # Stratified train/test split
    train_idx, test_idx = train_test_split(
        df.index,
        test_size=test_size,
        stratify=df["category_label"],
        random_state=seed,
    )
    df["is_train"] = 0
    df.loc[train_idx, "is_train"] = 1

    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_parquet, index=False)

    logger.info(
        "Saved processed ticket dataset → %s  [rows=%d, train=%d, test=%d]",
        output_parquet, len(df), (df["is_train"] == 1).sum(), (df["is_train"] == 0).sum(),
    )
    return df


def build_feature_set(
    churn_csv: Path = CHURN_CSV,
    tickets_csv: Path = TICKETS_CSV,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Execute complete feature engineering pipeline for all datasets."""
    churn_df = process_churn_features(churn_csv=churn_csv)
    tickets_df = process_ticket_features(tickets_csv=tickets_csv)
    return churn_df, tickets_df


def load_feature_set(parquet_path: Path = CHURN_PROCESSED_PARQUET) -> pd.DataFrame:
    """Load a processed feature set from Parquet."""
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Feature dataset not found at {parquet_path}. "
            "Run build_feature_set() first."
        )
    df = pd.read_parquet(parquet_path)
    logger.info("Loaded feature dataset from %s  [rows=%d]", parquet_path, len(df))
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    build_feature_set()

