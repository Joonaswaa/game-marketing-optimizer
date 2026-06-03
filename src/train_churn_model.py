"""Train and save the 14-day churn XGBoost classifier."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from churn_features import CHURN_FEATURE_COLUMNS, CHURN_TARGET
from preprocessing import build_churn_preprocessor
from utils import load_config, setup_logging

logger = logging.getLogger(__name__)


def load_acquisition_data(acquisition_path: str) -> pd.DataFrame:
    """
    Load the acquisition dataset from a project-relative path.

    Args:
        acquisition_path: Path to acquisition CSV relative to project root.

    Returns:
        Loaded acquisition DataFrame.
    """
    project_root = Path(__file__).parent.parent
    return pd.read_csv(project_root / acquisition_path)


def build_churn_pipeline(config: dict[str, Any]) -> Pipeline:
    """
    Build the churn classification pipeline with scaling and XGBoost.

    Args:
        config: Project configuration dictionary.

    Returns:
        Unfitted sklearn Pipeline ready for training.
    """
    churn_cfg = config.get("churn_training", {})
    seed = int(config["data"]["random_seed"])

    return Pipeline(
        [
            ("preprocessor", build_churn_preprocessor()),
            (
                "model",
                XGBClassifier(
                    n_estimators=int(churn_cfg.get("n_estimators", 200)),
                    max_depth=int(churn_cfg.get("max_depth", 5)),
                    learning_rate=float(churn_cfg.get("learning_rate", 0.05)),
                    random_state=seed,
                    eval_metric="logloss",
                ),
            ),
        ]
    )


def train_churn_model(df: pd.DataFrame, config: dict[str, Any]) -> tuple[Pipeline, dict[str, float]]:
    """
    Train the churn classifier and return evaluation metrics.

    Args:
        df: Acquisition dataset with churn features and target.
        config: Project configuration dictionary.

    Returns:
        Tuple of (fitted pipeline, metrics dict).
    """
    seed = int(config["data"]["random_seed"])
    test_size = float(config["training"]["test_size"])

    missing = [col for col in CHURN_FEATURE_COLUMNS + [CHURN_TARGET] if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing churn columns {missing}. Regenerate data with data_generation.py."
        )

    X = df[CHURN_FEATURE_COLUMNS]
    y = df[CHURN_TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    positive_count = max(int(y_train.sum()), 1)
    negative_count = int((y_train == 0).sum())
    scale_pos_weight = negative_count / positive_count

    pipeline = build_churn_pipeline(config)
    pipeline.set_params(model__scale_pos_weight=scale_pos_weight)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
    }

    logger.info("Churn model — Accuracy: %.4f", metrics["accuracy"])
    logger.info("Churn model — Precision: %.4f", metrics["precision"])
    logger.info("Churn model — Recall: %.4f", metrics["recall"])
    logger.info("Churn model — ROC-AUC: %.4f", metrics["roc_auc"])

    return pipeline, metrics


def save_model(pipeline: Pipeline, model_path: str) -> None:
    """
    Persist the trained churn pipeline to disk.

    Args:
        pipeline: Fitted churn pipeline.
        model_path: Output path relative to project root.
    """
    output_path = Path(__file__).parent.parent / model_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output_path)
    logger.info("Saved churn model to %s", output_path)


def save_metrics(metrics: dict[str, float], metrics_path: str) -> None:
    """
    Write churn evaluation metrics to JSON for the Streamlit UI.

    Args:
        metrics: Evaluation metric dictionary.
        metrics_path: Output path relative to project root.
    """
    output_path = Path(__file__).parent.parent / metrics_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    logger.info("Saved churn metrics to %s", output_path)


def main() -> None:
    """Load data, train churn model, and save model + metrics."""
    setup_logging()
    config = load_config()

    df = load_acquisition_data(str(config["data"]["acquisition_path"]))
    pipeline, metrics = train_churn_model(df, config)
    save_model(pipeline, str(config["models"]["churn_path"]))
    save_metrics(metrics, str(config["models"]["churn_metrics_path"]))


if __name__ == "__main__":
    main()
