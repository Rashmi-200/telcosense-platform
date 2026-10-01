"""
TelcoSense — Ticket Category & Priority Classifier Trainer
============================================================
Trains a LightGBM multi-class ticket classifier (Billing / Network / Hardware)
and priority estimator using TF-IDF text features + Optuna HPO + MLflow tracking.

Usage:
  python -m src.models.train_classifier
  python -m src.models.train_classifier --trials 10 --cv 5
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
from lightgbm import LGBMClassifier
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger("telcosense.train_classifier")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROCESSED_DIR: Final[Path] = Path(__file__).resolve().parents[2] / "data" / "processed"
TICKETS_PROCESSED_PARQUET: Final[Path] = PROCESSED_DIR / "tickets_processed.parquet"
CATEGORIES: Final[list[str]] = ["Billing", "Network", "Hardware"]
MLFLOW_EXPERIMENT: Final[str] = "telcosense-ticket-classifier"
MODEL_NAME: Final[str] = "telcosense-ticket-classifier"
RANDOM_SEED: Final[int] = 42


# ---------------------------------------------------------------------------
# Pydantic config
# ---------------------------------------------------------------------------


class ClassifierTrainConfig(BaseModel):
    """Training configuration for the ticket classifier."""

    n_trials: int = Field(default=10, ge=1)
    cv_folds: int = Field(default=5, ge=2)
    test_size: float = Field(default=0.2, gt=0.0, lt=1.0)
    max_tfidf_features: int = Field(default=5000, ge=100)
    random_seed: int = Field(default=RANDOM_SEED)
    mlflow_tracking_uri: str = "file:./mlruns"


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------


def _lgbm_objective(
    trial: optuna.Trial,
    X: Any,
    y: np.ndarray,
    cv: StratifiedKFold,
) -> float:
    params: dict[str, Any] = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "num_leaves": trial.suggest_int("num_leaves", 15, 100),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 1e-2, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 30),
        "class_weight": "balanced",
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
        "verbose": -1,
    }
    model = LGBMClassifier(**params)
    scores: np.ndarray = cross_val_score(
        model, X, y, cv=cv, scoring="f1_macro", n_jobs=-1
    )
    return float(scores.mean())


# ---------------------------------------------------------------------------
# Training pipeline
# ---------------------------------------------------------------------------


def train_classifier(config: ClassifierTrainConfig) -> dict[str, float]:
    """Full ticket classifier training with TF-IDF + LightGBM + Optuna + MLflow."""

    # ── Load data ────────────────────────────────────────────────────────────
    logger.info("Loading processed ticket features from %s …", TICKETS_PROCESSED_PARQUET)
    if not TICKETS_PROCESSED_PARQUET.exists():
        raw_csv = PROCESSED_DIR.parent / "raw" / "tickets.csv"
        if raw_csv.exists():
            df = pd.read_csv(raw_csv)
            category_map = {cat: idx for idx, cat in enumerate(CATEGORIES)}
            df["category_label"] = df["category"].map(category_map).fillna(0).astype(int)
        else:
            raise FileNotFoundError(f"Ticket dataset not found at {TICKETS_PROCESSED_PARQUET}")
    else:
        df = pd.read_parquet(TICKETS_PROCESSED_PARQUET)

    logger.info("Loaded %d processed ticket records.", len(df))

    if "is_train" in df.columns:
        train_df = df[df["is_train"] == 1]
        test_df = df[df["is_train"] == 0]
        X_train_text = train_df["ticket_text"].tolist()
        y_train = train_df["category_label"].values
        X_test_text = test_df["ticket_text"].tolist()
        y_test = test_df["category_label"].values
    else:
        texts = df["ticket_text"].tolist()
        labels = df["category_label"].values
        X_train_text, X_test_text, y_train, y_test = train_test_split(
            texts, labels, test_size=config.test_size, stratify=labels,
            random_state=config.random_seed
        )

    # ── TF-IDF Feature Extraction ────────────────────────────────────────────
    tfidf = TfidfVectorizer(
        max_features=config.max_tfidf_features,
        ngram_range=(1, 2),
        sublinear_tf=True,
        strip_accents="unicode",
        analyzer="word",
        token_pattern=r"\w{2,}",
    )

    X_train_vec = tfidf.fit_transform(X_train_text)
    X_test_vec = tfidf.transform(X_test_text)

    cv = StratifiedKFold(
        n_splits=config.cv_folds, shuffle=True, random_state=config.random_seed
    )

    # ── MLflow ───────────────────────────────────────────────────────────────
    try:
        mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    except Exception:
        mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name="lgbm-tfidf-optuna-10trials"):
        mlflow.log_params(config.model_dump())

        # ── Optuna HPO ───────────────────────────────────────────────────────
        logger.info("Starting Optuna HPO (%d trials) …", config.n_trials)
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=config.random_seed),
        )
        study.optimize(
            lambda trial: _lgbm_objective(trial, X_train_vec, y_train, cv),
            n_trials=config.n_trials,
            show_progress_bar=False,
        )

        best_params = study.best_params
        best_f1 = study.best_value
        logger.info("Best CV F1-Macro: %.4f | Params: %s", best_f1, best_params)
        mlflow.log_metric("cv_f1_macro", best_f1)
        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})

        # ── Final pipeline training ──────────────────────────────────────────
        final_params = best_params.copy()
        final_params.update(
            {
                "class_weight": "balanced",
                "random_state": config.random_seed,
                "n_jobs": -1,
                "verbose": -1,
            }
        )
        pipeline = Pipeline(
            [
                ("tfidf", tfidf),
                ("clf", LGBMClassifier(**final_params)),
            ]
        )
        pipeline.fit(X_train_text, y_train)

        # ── Evaluation ───────────────────────────────────────────────────────
        y_pred = pipeline.predict(X_test_text)
        test_acc = float(accuracy_score(y_test, y_pred))
        test_f1_macro = float(f1_score(y_test, y_pred, average="macro"))
        test_f1_weighted = float(f1_score(y_test, y_pred, average="weighted"))
        test_precision_macro = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
        test_recall_macro = float(recall_score(y_test, y_pred, average="macro", zero_division=0))

        report = classification_report(y_test, y_pred, target_names=CATEGORIES)
        logger.info("Test Results:\n%s", report)

        metrics = {
            "Accuracy": round(test_acc, 4),
            "F1-Macro": round(test_f1_macro, 4),
            "F1-Weighted": round(test_f1_weighted, 4),
            "Precision-Macro": round(test_precision_macro, 4),
            "Recall-Macro": round(test_recall_macro, 4),
        }

        mlflow.log_metric("accuracy", test_acc)
        mlflow.log_metric("f1_macro", test_f1_macro)
        mlflow.log_metric("f1_weighted", test_f1_weighted)
        mlflow.log_metric("precision_macro", test_precision_macro)
        mlflow.log_metric("recall_macro", test_recall_macro)
        mlflow.log_text(report, "classification_report.txt")

        # ── Log model artifact ───────────────────────────────────────────────
        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="ticket_classifier",
            registered_model_name=MODEL_NAME,
        )
        logger.info("Ticket classifier saved & registered to MLflow as '%s'.", MODEL_NAME)

        print("\n" + "=" * 50)
        print(" TICKET CLASSIFIER TRAINING METRICS (LightGBM + TF-IDF)")
        print("=" * 50)
        for name, val in metrics.items():
            print(f"  • {name:<15}: {val:.4f}")
        print("=" * 50 + "\n")

        return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TelcoSense ticket classifier trainer.")
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
    config = ClassifierTrainConfig(
        n_trials=args.trials,
        cv_folds=args.cv,
        mlflow_tracking_uri=args.tracking_uri,
    )
    train_classifier(config)


if __name__ == "__main__":
    main(sys.argv[1:])

