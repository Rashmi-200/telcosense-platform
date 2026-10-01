"""
TelcoSense — Automated Retraining Airflow DAG
=============================================
Orchestrates automated continuous training (CT) and deployment based on
data and concept drift detection:

  1. check_data_drift:
     - Uses Evidently AI (`src/monitoring/drift_detector.py`) to check for drift
       between current incoming features and the reference baseline ($p < 0.05$).
  2. drift_branch:
     - Evaluates drift flag via BranchPythonOperator.
     - If drift detected: branches to feature engineering & retraining.
     - If no drift: skips to no_drift_detected sink.
  3. rebuild_feature_store:
     - Triggers `src/data/feature_store.py` to regenerate feature sets.
  4. train_challenger_model:
     - Retrains XGBoost pipeline via Optuna HPO (`src/models/train_churn.py`).
     - Logs candidate model as a new Challenger version in MLflow.
  5. evaluate_and_promote:
     - Compares Challenger ROC-AUC vs current Champion (Production) ROC-AUC.
     - If Challenger ROC-AUC > Champion ROC-AUC, transitions Challenger to Stage="Production".

Schedule: Daily / Drift-triggered
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path
PLATFORM_ROOT = Path(__file__).resolve().parents[1]
if str(PLATFORM_ROOT) not in sys.path:
    sys.path.insert(0, str(PLATFORM_ROOT))

from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator

logger = logging.getLogger("telcosense.airflow.retraining")

# ---------------------------------------------------------------------------
# Default DAG arguments
# ---------------------------------------------------------------------------
DEFAULT_ARGS: dict[str, Any] = {
    "owner": "telcosense-mlops",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


# ---------------------------------------------------------------------------
# Task Callables
# ---------------------------------------------------------------------------

def check_data_drift(**context: Any) -> bool:
    """
    Check for feature drift using Evidently AI.
    Pushes drift detection boolean flag to XCom.
    """
    import pandas as pd
    from src.monitoring.drift_detector import DriftConfig, detect_drift

    processed_dir = PLATFORM_ROOT / "data" / "processed"
    churn_parquet = processed_dir / "churn_processed.parquet"

    if not churn_parquet.exists():
        logger.warning("Processed data not found at %s. Marking drift=True to initialize.", churn_parquet)
        context["ti"].xcom_push(key="drift_detected", value=True)
        return True

    df = pd.read_parquet(churn_parquet)
    split_idx = int(len(df) * 0.7)
    reference_df = df.iloc[:split_idx]
    current_df = df.iloc[split_idx:]

    ref_path = processed_dir / "reference_baseline.parquet"
    reference_df.to_parquet(ref_path, index=False)

    config = DriftConfig(
        reference_path=ref_path,
        drift_threshold=0.05,  # p < 0.05 significance threshold
    )

    result = detect_drift(current_df=current_df, config=config)
    drift_detected = result.dataset_drift_detected or len(result.drifted_features) > 0

    logger.info(
        "Drift check complete: detected=%s, drifted_features=%s, drift_share=%.2f",
        drift_detected,
        result.drifted_features,
        result.drift_share,
    )

    context["ti"].xcom_push(key="drift_detected", value=drift_detected)
    context["ti"].xcom_push(key="drifted_features", value=result.drifted_features)
    return drift_detected


def decide_drift_branch(**context: Any) -> str:
    """Branch to retraining pipeline if drift was detected, otherwise skip."""
    ti = context["ti"]
    drift_detected = ti.xcom_pull(key="drift_detected", task_ids="check_data_drift")

    if drift_detected:
        logger.info("Drift detected (p < 0.05). Routing to feature store and model retraining.")
        return "rebuild_feature_store"
    else:
        logger.info("No significant drift detected. Skipping retraining.")
        return "no_drift_detected"


def rebuild_feature_store(**_: Any) -> None:
    """Execute feature store processing on raw data."""
    from src.data.feature_store import process_churn_features, process_ticket_features

    raw_dir = PLATFORM_ROOT / "data" / "raw"
    processed_dir = PLATFORM_ROOT / "data" / "processed"

    churn_raw = raw_dir / "ibm_churn.csv"
    churn_out = processed_dir / "churn_processed.parquet"

    if churn_raw.exists():
        logger.info("Processing churn features from %s -> %s", churn_raw, churn_out)
        process_churn_features(churn_csv=churn_raw, output_parquet=churn_out)

    tickets_raw = raw_dir / "tickets.csv"
    tickets_out = processed_dir / "tickets_processed.parquet"
    if tickets_raw.exists():
        logger.info("Processing ticket features from %s -> %s", tickets_raw, tickets_out)
        process_ticket_features(tickets_csv=tickets_raw, output_parquet=tickets_out)


def train_challenger_model(**context: Any) -> dict[str, Any]:
    """
    Train new Challenger XGBoost model using Optuna HPO and log to MLflow.
    Pushes challenger run_id and test ROC-AUC to XCom.
    """
    from src.models.train_churn import ChurnTrainConfig, train_churn_pipeline

    config = ChurnTrainConfig(
        n_trials=10,
        cv_folds=5,
        mlflow_tracking_uri=str(PLATFORM_ROOT / "mlruns"),
    )

    result = train_churn_pipeline(config)
    challenger_metrics = {
        "run_id": result.run_id,
        "model_version": result.model_version,
        "test_roc_auc": result.metrics.get("test_roc_auc", 0.0),
        "test_f1": result.metrics.get("test_f1", 0.0),
    }

    logger.info("Challenger model trained: %s", challenger_metrics)
    context["ti"].xcom_push(key="challenger_metrics", value=challenger_metrics)
    return challenger_metrics


def evaluate_and_promote(**context: Any) -> None:
    """
    Champion vs Challenger Gate:
    Compares Challenger ROC-AUC against current Champion in MLflow Stage='Production'.
    Promotes Challenger to 'Production' if Challenger score > Champion score.
    """
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow_dir = PLATFORM_ROOT / "mlruns"
    mlflow.set_tracking_uri(str(mlflow_dir))
    client = MlflowClient(tracking_uri=str(mlflow_dir))

    model_name = "telcosense-churn-model"
    ti = context["ti"]
    challenger = ti.xcom_pull(key="challenger_metrics", task_ids="train_challenger_model")

    if not challenger:
        logger.warning("No challenger metrics found in XCom.")
        return

    challenger_roc_auc = challenger.get("test_roc_auc", 0.0)
    challenger_version = challenger.get("model_version")

    champion_roc_auc = 0.0
    champion_version = None

    try:
        versions = client.search_model_versions(f"name='{model_name}'")
        for v in versions:
            if v.current_stage == "Production":
                champion_version = v.version
                run = client.get_run(v.run_id)
                champion_roc_auc = run.data.metrics.get("test_roc_auc", 0.0)
                break
    except Exception as exc:
        logger.info("Could not retrieve current Production model: %s", exc)

    logger.info(
        "Model Gate Comparison: Challenger v%s (ROC-AUC=%.4f) vs Champion v%s (ROC-AUC=%.4f)",
        challenger_version,
        challenger_roc_auc,
        champion_version,
        champion_roc_auc,
    )

    if challenger_roc_auc > champion_roc_auc or champion_version is None:
        logger.info(
            "Challenger outperformed Champion (%.4f > %.4f). Promoting v%s to 'Production'...",
            challenger_roc_auc,
            champion_roc_auc,
            challenger_version,
        )
        if challenger_version:
            if champion_version:
                client.transition_model_version_stage(
                    name=model_name,
                    version=champion_version,
                    stage="Archived",
                )
            client.transition_model_version_stage(
                name=model_name,
                version=challenger_version,
                stage="Production",
                archive_existing_versions=True,
            )
            logger.info("Successfully promoted v%s to Production in MLflow Model Registry.", challenger_version)
    else:
        logger.info(
            "Challenger ROC-AUC (%.4f) did not exceed Champion (%.4f). Retaining Champion v%s.",
            challenger_roc_auc,
            champion_roc_auc,
            champion_version,
        )


# ---------------------------------------------------------------------------
# DAG Definition
# ---------------------------------------------------------------------------

with DAG(
    dag_id="telcosense_retraining_dag",
    description="Automated drift-triggered CT/CD pipeline with Champion vs Challenger promotion",
    default_args=DEFAULT_ARGS,
    schedule_interval="@daily",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["telcosense", "mlops", "drift", "optuna", "mlflow"],
) as dag:

    check_drift_task = PythonOperator(
        task_id="check_data_drift",
        python_callable=check_data_drift,
        provide_context=True,
        doc_md="Evidently AI feature & dataset drift evaluation.",
    )

    branch_task = BranchPythonOperator(
        task_id="drift_branch",
        python_callable=decide_drift_branch,
        provide_context=True,
        doc_md="Conditional branch: retrain if drift detected, skip otherwise.",
    )

    rebuild_features_task = PythonOperator(
        task_id="rebuild_feature_store",
        python_callable=rebuild_feature_store,
        doc_md="Transform raw CSVs into engineered Parquet feature sets.",
    )

    train_challenger_task = PythonOperator(
        task_id="train_challenger_model",
        python_callable=train_challenger_model,
        provide_context=True,
        doc_md="Retrain XGBoost churn classifier using Optuna HPO.",
    )

    evaluate_promote_task = PythonOperator(
        task_id="evaluate_and_promote",
        python_callable=evaluate_and_promote,
        provide_context=True,
        doc_md="Champion vs Challenger ROC-AUC evaluation and automated promotion to Production.",
    )

    no_drift_sink = EmptyOperator(
        task_id="no_drift_detected",
        doc_md="No drift detected; retraining bypassed.",
    )

    # DAG Topology
    check_drift_task >> branch_task
    branch_task >> rebuild_features_task >> train_challenger_task >> evaluate_promote_task
    branch_task >> no_drift_sink
