"""Train and save the Day-7 retention XGBoost classifier."""

import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from preprocessing import FEATURE_COLUMNS, RETENTION_TARGET, build_preprocessor
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


def build_retention_pipeline(config: dict[str, Any]) -> Pipeline:
    """
    Build the retention classification pipeline with preprocessing and XGBoost.

    Args:
        config: Project configuration dictionary.

    Returns:
        Unfitted sklearn Pipeline ready for training.
    """
    training = config["training"]
    seed = int(config["data"]["random_seed"])

    return Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            (
                "model",
                XGBClassifier(
                    n_estimators=int(training["n_estimators"]),
                    max_depth=int(training["max_depth"]),
                    learning_rate=float(training["learning_rate"]),
                    random_state=seed,
                    eval_metric="logloss",
                ),
            ),
        ]
    )


def train_retention_model(df: pd.DataFrame, config: dict[str, Any]) -> Pipeline:
    """
    Train the retention classifier and log evaluation metrics.

    Args:
        df: Acquisition dataset containing features and retention target.
        config: Project configuration dictionary.

    Returns:
        Fitted retention pipeline.
    """
    seed = int(config["data"]["random_seed"])
    test_size = float(config["training"]["test_size"])

    X = df[FEATURE_COLUMNS]
    y = df[RETENTION_TARGET]

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

    pipeline = build_retention_pipeline(config)
    pipeline.set_params(model__scale_pos_weight=scale_pos_weight)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)

    logger.info("Retention model — ROC-AUC: %.4f", roc_auc)
    logger.info("Retention model — F1 score: %.4f", f1)

    if roc_auc < 0.75:
        logger.warning("ROC-AUC below target of 0.75")

    return pipeline


def save_model(pipeline: Pipeline, model_path: str) -> None:
    """
    Persist the trained retention pipeline to disk.

    Args:
        pipeline: Fitted retention pipeline.
        model_path: Output path relative to project root.
    """
    output_path = Path(__file__).parent.parent / model_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output_path)
    logger.info("Saved retention model to %s", output_path)


def main() -> None:
    """Load data, train retention model, and save to models/."""
    setup_logging()
    config = load_config()

    df = load_acquisition_data(str(config["data"]["acquisition_path"]))
    pipeline = train_retention_model(df, config)
    save_model(pipeline, str(config["models"]["retention_path"]))


if __name__ == "__main__":
    main()
