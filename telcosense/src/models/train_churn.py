"""
TelcoSense — Churn Prediction Model Trainer
=============================================
Trains an XGBoost churn model with Optuna HPO and MLflow experiment tracking.

Usage:
  python -m src.models.train_churn
  python -m src.models.train_churn --trials 10 --cv 5
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Final

import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# Suppress Optuna logs below WARNING
optuna.logging.set_verbosity(optuna.logging.WARNING)

logger = logging.getLogger("telcosense.train_churn")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROCESSED_DIR: Final[Path] = Path(__file__).resolve().parents[2] / "data" / "processed"
CHURN_PROCESSED_PARQUET: Final[Path] = PROCESSED_DIR / "churn_processed.parquet"
TARGET_COL: Final[str] = "Churn_label"
DROP_COLS: Final[list[str]] = ["customerID", "Churn", "Churn_label", "is_train"]
MLFLOW_EXPERIMENT: Final[str] = "telcosense-churn-prediction"
MODEL_NAME: Final[str] = "telcosense-churn-model"
RANDOM_SEED: Final[int] = 42


# ---------------------------------------------------------------------------
# Pydantic config
# ---------------------------------------------------------------------------


class ChurnTrainConfig(BaseModel):
    """Hyperparameter search space bounds for Optuna."""

    n_trials: int = Field(default=10, ge=1)
    cv_folds: int = Field(default=5, ge=2)
    test_size: float = Field(default=0.2, gt=0.0, lt=1.0)
    early_stopping_rounds: int = Field(default=50, ge=1)
    random_seed: int = Field(default=RANDOM_SEED)
    mlflow_tracking_uri: str = "file:./mlruns"


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------


def _xgb_objective(
    trial: optuna.Trial,
    X: np.ndarray,
    y: np.ndarray,
    cv: StratifiedKFold,
) -> float:
    """Return mean cross-validated ROC-AUC for a given trial's parameters."""
    params: dict[str, Any] = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 1e-2, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 8),
        "gamma": trial.suggest_float("gamma", 0.0, 3.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
        "eval_metric": "logloss",
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
    }
    model = XGBClassifier(**params)
    scores: np.ndarray = cross_val_score(
        model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1
    )
    return float(scores.mean())


# ---------------------------------------------------------------------------
# Training pipeline
# ---------------------------------------------------------------------------


def train_churn_model(config: ChurnTrainConfig) -> dict[str, float]:
    """Full training run with Optuna HPO + MLflow tracking."""

    # ── Load features ────────────────────────────────────────────────────────
    logger.info("Loading processed churn features from %s …", CHURN_PROCESSED_PARQUET)
    if not CHURN_PROCESSED_PARQUET.exists():
        # Fallback if legacy path exists
        alt = PROCESSED_DIR / "features.parquet"
        if alt.exists():
            df = pd.read_parquet(alt)
        else:
            raise FileNotFoundError(f"Processed dataset not found at {CHURN_PROCESSED_PARQUET}")
    else:
        df = pd.read_parquet(CHURN_PROCESSED_PARQUET)

    feature_cols = [c for c in df.columns if c not in DROP_COLS]
    
    if "is_train" in df.columns:
        train_df = df[df["is_train"] == 1]
        test_df = df[df["is_train"] == 0]
        X_train = train_df[feature_cols].values.astype(np.float32)
        y_train = train_df[TARGET_COL].values.astype(int)
        X_test = test_df[feature_cols].values.astype(np.float32)
        y_test = test_df[TARGET_COL].values.astype(int)
    else:
        from sklearn.model_selection import train_test_split
        X = df[feature_cols].values.astype(np.float32)
        y = df[TARGET_COL].values.astype(int)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.test_size, stratify=y, random_state=config.random_seed
        )

    logger.info(
        "Train set: %s | Test set: %s | Churn rate: %.2f%%",
        X_train.shape, X_test.shape, y_train.mean() * 100
    )

    cv = StratifiedKFold(
        n_splits=config.cv_folds, shuffle=True, random_state=config.random_seed
    )

    # ── MLflow setup ─────────────────────────────────────────────────────────
    try:
        mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    except Exception:
        mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name="xgboost-optuna-10trials"):
        mlflow.log_params(config.model_dump())

        # ── Optuna HPO ───────────────────────────────────────────────────────
        logger.info("Starting Optuna HPO (%d trials) …", config.n_trials)
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=config.random_seed),
        )
        study.optimize(
            lambda trial: _xgb_objective(trial, X_train, y_train, cv),
            n_trials=config.n_trials,
            show_progress_bar=False,
        )

        best_params = study.best_params
        best_cv_auc = study.best_value
        logger.info("Best CV ROC-AUC: %.4f | Params: %s", best_cv_auc, best_params)
        mlflow.log_metric("cv_roc_auc", best_cv_auc)
        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})

        # ── Final model training ─────────────────────────────────────────────
        final_params = best_params.copy()
        final_params.update(
            {
                "eval_metric": "logloss",
                "random_state": config.random_seed,
                "n_jobs": -1,
            }
        )
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", XGBClassifier(**final_params)),
            ]
        )
        pipeline.fit(X_train, y_train)

        # ── Evaluation ───────────────────────────────────────────────────────
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        test_auc = float(roc_auc_score(y_test, y_prob))
        test_precision = float(precision_score(y_test, y_pred, average="binary", zero_division=0))
        test_recall = float(recall_score(y_test, y_pred, average="binary", zero_division=0))
        test_f1 = float(f1_score(y_test, y_pred, average="binary", zero_division=0))
        test_acc = float(accuracy_score(y_test, y_pred))

        metrics = {
            "ROC-AUC": round(test_auc, 4),
            "Precision": round(test_precision, 4),
            "Recall": round(test_recall, 4),
            "F1-Score": round(test_f1, 4),
            "Accuracy": round(test_acc, 4),
        }

        mlflow.log_metric("roc_auc", test_auc)
        mlflow.log_metric("precision", test_precision)
        mlflow.log_metric("recall", test_recall)
        mlflow.log_metric("f1_score", test_f1)
        mlflow.log_metric("accuracy", test_acc)

        # ── Log model artifact ───────────────────────────────────────────────
        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="churn_model",
            registered_model_name=MODEL_NAME,
        )
        logger.info("Churn model saved & registered to MLflow as '%s'.", MODEL_NAME)

        print("\n" + "=" * 50)
        print(" CHURN MODEL TRAINING METRICS (XGBoost + Optuna)")
        print("=" * 50)
        for name, val in metrics.items():
            print(f"  • {name:<12}: {val:.4f}")
        print("=" * 50 + "\n")

        return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TelcoSense churn model trainer.")
    parser.add_argument("--trials", type=int, default=10, help="Optuna trials (default: 10).")
    parser.add_argument("--cv", type=int, default=5, help="CV folds (default: 5).")
    parser.add_argument("--tracking-uri", default="file:./mlruns")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    args = _parse_args(argv)
    config = ChurnTrainConfig(
        n_trials=args.trials,
        cv_folds=args.cv,
        mlflow_tracking_uri=args.tracking_uri,
    )
    train_churn_model(config)


if __name__ == "__main__":
    main(sys.argv[1:])

