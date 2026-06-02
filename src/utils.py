"""Shared utilities for config loading and logging."""

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from lifetimes import BetaGeoFitter, GammaGammaFitter


def setup_logging(level: str = "INFO") -> None:
    """Configure logging for all modules."""
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s — %(name)s — %(levelname)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_config() -> dict[str, Any]:
    """Load project config from config/config.yaml."""
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_bgnbd_models(
    bgf: BetaGeoFitter,
    ggf: GammaGammaFitter,
    model_path: str,
) -> None:
    """
    Persist BG/NBD models using parameter bundles (lifetimes fitters are not picklable).

    Args:
        bgf: Fitted BetaGeoFitter model.
        ggf: Fitted GammaGammaFitter model.
        model_path: Output path relative to project root.
    """
    import joblib

    output_path = Path(__file__).parent.parent / model_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    bundle = {
        "bgf_params": bgf.params_.to_dict(),
        "bgf_scale": bgf._scale,
        "bgf_penalizer": bgf.penalizer_coef,
        "ggf_params": ggf.params_.to_dict(),
        "ggf_penalizer": ggf.penalizer_coef,
    }
    joblib.dump(bundle, output_path)


def load_bgnbd_models(model_path: str) -> tuple[BetaGeoFitter, GammaGammaFitter]:
    """
    Restore BG/NBD models from a saved parameter bundle.

    Args:
        model_path: Path to saved model bundle relative to project root.

    Returns:
        Tuple of restored (BetaGeoFitter, GammaGammaFitter) models.
    """
    import joblib

    bundle = joblib.load(Path(__file__).parent.parent / model_path)

    bgf = BetaGeoFitter(penalizer_coef=bundle["bgf_penalizer"])
    bgf.params_ = pd.Series(bundle["bgf_params"])
    bgf._scale = bundle["bgf_scale"]
    bgf.predict = bgf.conditional_expected_number_of_purchases_up_to_time

    ggf = GammaGammaFitter(penalizer_coef=bundle["ggf_penalizer"])
    ggf.params_ = pd.Series(bundle["ggf_params"])

    return bgf, ggf
