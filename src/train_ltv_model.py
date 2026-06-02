"""Train and save the 90-day LTV XGBoost regressor."""

import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from preprocessing import FEATURE_COLUMNS, LTV_TARGET, build_preprocessor
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


def build_ltv_pipeline(config: dict[str, Any]) -> Pipeline:
    """
    Build the LTV regression pipeline with preprocessing and XGBoost.

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
                XGBRegressor(
                    n_estimators=int(training["n_estimators"]),
                    max_depth=int(training["max_depth"]),
                    learning_rate=float(training["learning_rate"]),
                    random_state=seed,
                    eval_metric="rmse",
                ),
            ),
        ]
    )


def train_ltv_model(df: pd.DataFrame, config: dict[str, Any]) -> Pipeline:
    """
    Train the LTV regressor and log evaluation metrics.

    Args:
        df: Acquisition dataset containing features and LTV target.
        config: Project configuration dictionary.

    Returns:
        Fitted LTV pipeline.
    """
    seed = int(config["data"]["random_seed"])
    test_size = float(config["training"]["test_size"])

    X = df[FEATURE_COLUMNS]
    y = df[LTV_TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
    )

    pipeline = build_ltv_pipeline(config)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    mae = mean_absolute_error(y_test, y_pred)

    logger.info("LTV model — R²: %.4f", r2)
    logger.info("LTV model — RMSE: %.4f", rmse)
    logger.info("LTV model — MAE: %.4f", mae)

    if r2 < 0.70:
        logger.warning("R² below target of 0.70")

    negative_predictions = int((y_pred < 0).sum())
    if negative_predictions > 0:
        logger.warning("Model produced %d negative LTV predictions", negative_predictions)

    return pipeline


def save_model(pipeline: Pipeline, model_path: str) -> None:
    """
    Persist the trained LTV pipeline to disk.

    Args:
        pipeline: Fitted LTV pipeline.
        model_path: Output path relative to project root.
    """
    output_path = Path(__file__).parent.parent / model_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output_path)
    logger.info("Saved LTV model to %s", output_path)


def main() -> None:
    """Load data, train LTV model, and save to models/."""
    setup_logging()
    config = load_config()

    df = load_acquisition_data(str(config["data"]["acquisition_path"]))
    pipeline = train_ltv_model(df, config)
    save_model(pipeline, str(config["models"]["ltv_xgboost_path"]))


if __name__ == "__main__":
    main()
