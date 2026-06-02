"""Train and save BG/NBD + Gamma-Gamma LTV models using transaction history."""

import logging
from pathlib import Path

import pandas as pd
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import summary_data_from_transaction_data

from utils import load_config, save_bgnbd_models, setup_logging

logger = logging.getLogger(__name__)


def load_transaction_data(transaction_path: str) -> pd.DataFrame:
    """
    Load the transaction dataset from a project-relative path.

    Args:
        transaction_path: Path to transaction CSV relative to project root.

    Returns:
        Loaded transaction DataFrame.
    """
    project_root = Path(__file__).parent.parent
    df = pd.read_csv(project_root / transaction_path)
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    return df


def build_rfm_summary(
    transactions_df: pd.DataFrame,
    observation_period_end: str,
) -> pd.DataFrame:
    """
    Build an RFM summary table from raw transaction records.

    Args:
        transactions_df: Transaction history with user_id, date, and amount.
        observation_period_end: End of observation window (YYYY-MM-DD).

    Returns:
        RFM summary suitable for lifetimes model fitting.
    """
    return summary_data_from_transaction_data(
        transactions_df,
        customer_id_col="user_id",
        datetime_col="transaction_date",
        monetary_value_col="transaction_amount",
        observation_period_end=observation_period_end,
    )


def train_bgnbd_models(
    summary: pd.DataFrame,
) -> tuple[BetaGeoFitter, GammaGammaFitter]:
    """
    Fit BG/NBD and Gamma-Gamma models on RFM summary data.

    Gamma-Gamma is fit only on repeat buyers (frequency > 0).

    Args:
        summary: RFM summary from transaction history.

    Returns:
        Tuple of fitted (BetaGeoFitter, GammaGammaFitter) models.
    """
    bgf = BetaGeoFitter(penalizer_coef=0.01)
    bgf.fit(summary["frequency"], summary["recency"], summary["T"])
    logger.info("BG/NBD model fitted on %d customers", len(summary))

    repeat_buyers = summary[summary["frequency"] > 0]
    if repeat_buyers.empty:
        raise ValueError("No repeat buyers found — cannot fit Gamma-Gamma model")

    ggf = GammaGammaFitter(penalizer_coef=0.01)
    ggf.fit(repeat_buyers["frequency"], repeat_buyers["monetary_value"])
    logger.info("Gamma-Gamma model fitted on %d repeat buyers", len(repeat_buyers))

    return bgf, ggf


def validate_ltv_predictions(
    bgf: BetaGeoFitter,
    ggf: GammaGammaFitter,
    summary: pd.DataFrame,
) -> None:
    """
    Log sample BG/NBD LTV predictions to confirm positive outputs.

    Args:
        bgf: Fitted BetaGeoFitter model.
        ggf: Fitted GammaGammaFitter model.
        summary: RFM summary used for prediction.
    """
    repeat_buyers = summary[summary["frequency"] > 0]
    ltv = ggf.customer_lifetime_value(
        bgf,
        repeat_buyers["frequency"],
        repeat_buyers["recency"],
        repeat_buyers["T"],
        repeat_buyers["monetary_value"],
        time=3,
        freq="D",
        discount_rate=0.01,
    )

    logger.info(
        "BG/NBD LTV predictions — count: %d, mean: %.2f, min: %.2f",
        len(ltv),
        ltv.mean(),
        ltv.min(),
    )

    if (ltv < 0).any():
        logger.warning("BG/NBD produced negative LTV predictions")


def save_models(
    bgf: BetaGeoFitter,
    ggf: GammaGammaFitter,
    model_path: str,
) -> None:
    """
    Persist fitted BG/NBD and Gamma-Gamma models to disk.

    Args:
        bgf: Fitted BetaGeoFitter model.
        ggf: Fitted GammaGammaFitter model.
        model_path: Output path relative to project root.
    """
    save_bgnbd_models(bgf, ggf, model_path)
    logger.info("Saved BG/NBD models to %s", Path(__file__).parent.parent / model_path)


def main() -> None:
    """Load transactions, fit BG/NBD models, validate, and save."""
    setup_logging()
    config = load_config()

    transactions_df = load_transaction_data(str(config["data"]["transaction_path"]))
    observation_period_end = str(config["data"]["observation_period_end"])

    summary = build_rfm_summary(transactions_df, observation_period_end)
    bgf, ggf = train_bgnbd_models(summary)
    validate_ltv_predictions(bgf, ggf, summary)
    save_models(bgf, ggf, str(config["models"]["bgnbd_path"]))


if __name__ == "__main__":
    main()
