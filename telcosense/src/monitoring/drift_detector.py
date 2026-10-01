"""
TelcoSense — Data Drift Detector
==================================
Uses Evidently to compute feature-level data drift and model performance
drift, emitting structured JSON reports and Prometheus-style metrics.

Usage:
  python -m src.monitoring.drift_detector --current data/raw/ibm_churn.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Final

import pandas as pd
from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.report import Report
from pydantic import BaseModel, Field

logger = logging.getLogger("telcosense.monitoring.drift_detector")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROCESSED_DIR: Final[Path] = Path(__file__).resolve().parents[3] / "data" / "processed"
REFERENCE_PARQUET: Final[Path] = PROCESSED_DIR / "features.parquet"
REPORTS_DIR: Final[Path] = Path(__file__).resolve().parents[3] / "logs" / "drift_reports"
DRIFT_THRESHOLD: Final[float] = 0.1   # Wasserstein p-value threshold
TARGET_COL: Final[str] = "Churn_label"
NUMERIC_FEATURES: Final[list[str]] = [
    "tenure", "MonthlyCharges", "TotalCharges",
    "charge_per_tenure", "tenure_band",
]


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class DriftResult(BaseModel):
    """Summary of a drift detection run."""

    run_timestamp: str
    reference_rows: int
    current_rows: int
    dataset_drift_detected: bool
    drifted_features: list[str] = Field(default_factory=list)
    drift_share: float = Field(ge=0.0, le=1.0)
    report_path: str


class DriftConfig(BaseModel):
    """Configuration for the drift detector."""

    reference_path: Path = REFERENCE_PARQUET
    reports_dir: Path = REPORTS_DIR
    drift_threshold: float = Field(default=DRIFT_THRESHOLD, gt=0.0, lt=1.0)
    target_column: str = TARGET_COL


# ---------------------------------------------------------------------------
# Core detector
# ---------------------------------------------------------------------------


def detect_drift(
    current_df: pd.DataFrame,
    config: DriftConfig | None = None,
) -> DriftResult:
    """
    Compare *current_df* against the stored reference feature set.

    Generates an Evidently HTML + JSON report and returns a structured
    DriftResult summary.

    Parameters
    ----------
    current_df:
        Incoming batch of feature-engineered data to check.
    config:
        DriftConfig.  Defaults to REFERENCE_PARQUET as reference.

    Returns
    -------
    DriftResult
        Structured drift detection summary.
    """
    cfg = config or DriftConfig()
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)

    # ── Load reference ────────────────────────────────────────────────────────
    if not cfg.reference_path.exists():
        raise FileNotFoundError(
            f"Reference dataset not found at {cfg.reference_path}. "
            "Run feature_store.build_feature_set() first."
        )
    reference_df = pd.read_parquet(cfg.reference_path)
    logger.info(
        "Reference: %d rows | Current: %d rows.", len(reference_df), len(current_df)
    )

    # Align columns
    common_cols = [c for c in reference_df.columns if c in current_df.columns]
    reference_df = reference_df[common_cols]
    current_df = current_df[common_cols]

    # ── Column mapping ────────────────────────────────────────────────────────
    numerical_features = [
        c for c in NUMERIC_FEATURES if c in common_cols
    ]
    categorical_features = [
        c for c in common_cols
        if c not in numerical_features and c != cfg.target_column and c != "customerID"
    ]
    column_mapping = ColumnMapping(
        target=cfg.target_column if cfg.target_column in common_cols else None,
        numerical_features=numerical_features,
        categorical_features=categorical_features,
    )

    # ── Evidently report ──────────────────────────────────────────────────────
    report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
    report.run(
        reference_data=reference_df,
        current_data=current_df,
        column_mapping=column_mapping,
    )

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    html_path = cfg.reports_dir / f"drift_report_{timestamp}.html"
    json_path = cfg.reports_dir / f"drift_report_{timestamp}.json"

    report.save_html(str(html_path))
    report.save_json(str(json_path))
    logger.info("Reports saved → %s", html_path)

    # ── Parse drift summary ───────────────────────────────────────────────────
    report_dict: dict[str, Any] = json.loads(json_path.read_text())
    drifted_features: list[str] = []
    drift_share = 0.0

    try:
        metrics = report_dict.get("metrics", [])
        for metric in metrics:
            result = metric.get("result", {})
            if "drift_by_columns" in result:
                drift_by_cols: dict = result["drift_by_columns"]
                for feature, stats in drift_by_cols.items():
                    if stats.get("drift_detected", False):
                        drifted_features.append(feature)
            if "dataset_drift" in result:
                drift_share = float(result.get("share_of_drifted_columns", 0.0))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not parse Evidently JSON output: %s", exc)

    dataset_drift = len(drifted_features) > 0

    if dataset_drift:
        logger.warning(
            "DRIFT DETECTED — %d feature(s): %s",
            len(drifted_features),
            drifted_features,
        )
    else:
        logger.info("No significant drift detected.")

    return DriftResult(
        run_timestamp=timestamp,
        reference_rows=len(reference_df),
        current_rows=len(current_df),
        dataset_drift_detected=dataset_drift,
        drifted_features=drifted_features,
        drift_share=round(drift_share, 4),
        report_path=str(html_path),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TelcoSense drift detector.")
    parser.add_argument(
        "--current",
        type=Path,
        required=True,
        help="Path to current batch CSV or Parquet.",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=REFERENCE_PARQUET,
        help="Path to reference Parquet (default: processed/features.parquet).",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=REPORTS_DIR,
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    args = _parse_args(argv)

    ext = args.current.suffix.lower()
    if ext == ".parquet":
        current_df = pd.read_parquet(args.current)
    elif ext in {".csv", ".tsv"}:
        current_df = pd.read_csv(args.current)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    config = DriftConfig(reference_path=args.reference, reports_dir=args.reports_dir)
    result = detect_drift(current_df, config)
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
