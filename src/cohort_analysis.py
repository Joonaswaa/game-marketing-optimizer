"""Retention cohort metrics and matrices for mobile game analytics."""

from __future__ import annotations

import numpy as np
import pandas as pd

RETENTION_DAY_COLUMNS = ["D1", "D7", "D30"]


def ensure_cohort_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Guarantee install_date and last_login_date exist (supports legacy CSV schemas).

    Args:
        df: Raw acquisition DataFrame.

    Returns:
        DataFrame with install_date and last_login_date columns.
    """
    cohort_df = df.copy()
    install_col = "install_date" if "install_date" in cohort_df.columns else "acquisition_date"
    if install_col not in cohort_df.columns:
        raise ValueError("Dataset needs acquisition_date or install_date for cohort analysis.")

    install_dt = pd.to_datetime(cohort_df[install_col])
    cohort_df["install_date"] = install_dt

    if "last_login_date" not in cohort_df.columns:
        if {"days_active", "days_since_last_login"}.issubset(cohort_df.columns):
            active_days = (
                cohort_df["days_active"] - cohort_df["days_since_last_login"]
            ).clip(lower=0)
        elif "retained_day7" in cohort_df.columns:
            active_days = np.where(cohort_df["retained_day7"] == 1, 14, 2)
        else:
            active_days = 3
        cohort_df["last_login_date"] = install_dt + pd.to_timedelta(active_days, unit="D")
    else:
        cohort_df["last_login_date"] = pd.to_datetime(cohort_df["last_login_date"])

    return cohort_df


def prepare_cohort_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add install week and day-level retention flags from install and last-login dates.

    D1: player was active at least once after install day (span >= 1 day).
    D7: player was active 7+ days after install.
    D30: player was active 30+ days after install.

    Args:
        df: Acquisition dataset with install_date and last_login_date columns.

    Returns:
        Copy of df with install_week, retained_d1, retained_d7, retained_d30.
    """
    cohort_df = ensure_cohort_columns(df)

    days_since_install = (
        cohort_df["last_login_date"] - cohort_df["install_date"]
    ).dt.days.clip(lower=0)

    cohort_df["install_week"] = (
        cohort_df["install_date"].dt.to_period("W").astype(str)
    )
    cohort_df["retained_d1"] = (days_since_install >= 1).astype(int)
    cohort_df["retained_d7"] = (days_since_install >= 7).astype(int)
    cohort_df["retained_d30"] = (days_since_install >= 30).astype(int)
    return cohort_df


def build_retention_matrix(cohort_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a cohort retention matrix (% retained) by install week.

    Args:
        cohort_df: Output of prepare_cohort_frame.

    Returns:
        DataFrame indexed by install_week with D1, D7, D30 columns (0–100).
    """
    grouped = (
        cohort_df.groupby("install_week")[["retained_d1", "retained_d7", "retained_d30"]]
        .mean()
        .mul(100)
        .round(1)
    )
    grouped.columns = RETENTION_DAY_COLUMNS
    return grouped.sort_index()


def average_retention_metrics(retention_matrix: pd.DataFrame) -> dict[str, float]:
    """
    Compute mean D1/D7/D30 retention across all install cohorts.

    Args:
        retention_matrix: Cohort matrix from build_retention_matrix.

    Returns:
        Dict with keys d1, d7, d30 (percentages).
    """
    if retention_matrix.empty:
        return {"d1": 0.0, "d7": 0.0, "d30": 0.0}
    return {
        "d1": float(retention_matrix["D1"].mean()),
        "d7": float(retention_matrix["D7"].mean()),
        "d30": float(retention_matrix["D30"].mean()),
    }


def retention_curve_averages(retention_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Build a single-row retention curve averaged across install weeks.

    Args:
        retention_matrix: Cohort matrix from build_retention_matrix.

    Returns:
        DataFrame with columns day_label and retention_pct.
    """
    averages = average_retention_metrics(retention_matrix)
    return pd.DataFrame(
        {
            "day_label": RETENTION_DAY_COLUMNS,
            "retention_pct": [
                averages["d1"],
                averages["d7"],
                averages["d30"],
            ],
        }
    )
