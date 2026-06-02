"""Generate synthetic Clash Royale-style acquisition and transaction datasets."""

import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker

from utils import load_config, setup_logging

logger = logging.getLogger(__name__)

# Channel mix, CPI, retention baseline, and LTV tier aligned with product_requirements.
CHANNEL_QUALITY: dict[str, dict[str, float]] = {
    "TikTok": {
        "channel_prob": 0.25,
        "cpi_base": 1.80,
        "base_retention": 0.33,
        "retention_boost": -0.05,
        "avg_ltv_multiplier": 1.0,
    },
    "Facebook": {
        "channel_prob": 0.20,
        "cpi_base": 3.20,
        "base_retention": 0.40,
        "retention_boost": 0.05,
        "avg_ltv_multiplier": 1.5,
    },
    "Google": {
        "channel_prob": 0.30,
        "cpi_base": 3.80,
        "base_retention": 0.42,
        "retention_boost": 0.10,
        "avg_ltv_multiplier": 2.0,
    },
    "Instagram": {
        "channel_prob": 0.15,
        "cpi_base": 2.90,
        "base_retention": 0.37,
        "retention_boost": 0.02,
        "avg_ltv_multiplier": 1.2,
    },
    "YouTube": {
        "channel_prob": 0.10,
        "cpi_base": 4.50,
        "base_retention": 0.48,
        "retention_boost": 0.15,
        "avg_ltv_multiplier": 2.5,
    },
}

DEVICE_TYPES = ["iOS", "Android"]
AGE_GROUPS = ["18-24", "25-34", "35-44", "45+"]
TRANSACTION_TYPES = ["gem_pack", "pass_royale", "chest_offer", "evo_shards"]

COUNTRY_ENGAGEMENT: dict[str, float] = {
    "US": 1.10,
    "UK": 1.05,
    "DE": 1.00,
    "JP": 1.15,
    "BR": 0.90,
    "FR": 0.95,
    "KR": 1.20,
}


def generate_acquisition_data(
    n_users: int,
    seed: int,
    channels: list[str],
    countries: list[str],
    observation_period_end: str,
) -> pd.DataFrame:
    """
    Generate a synthetic Clash Royale-style user acquisition dataset.

    Week-1 battles, arena progression, and clan activity drive retention and
    90-day IAP (gem spend). Channel mix reflects typical mobile UA spend.

    Args:
        n_users: Number of users to generate.
        seed: Random seed for reproducibility.
        channels: List of acquisition channel names.
        countries: List of country codes.
        observation_period_end: Last date in the observation window (YYYY-MM-DD).

    Returns:
        DataFrame with 15 columns including behavioral features and targets.
    """
    rng = np.random.default_rng(seed)
    fake = Faker()
    Faker.seed(seed)

    end_date = datetime.strptime(observation_period_end, "%Y-%m-%d")
    start_date = end_date - timedelta(days=365)

    channel_probs = np.array(
        [CHANNEL_QUALITY[ch]["channel_prob"] for ch in channels], dtype=float
    )
    channel_probs = channel_probs / channel_probs.sum()
    channel_idx = rng.choice(len(channels), size=n_users, p=channel_probs)
    country_idx = rng.integers(0, len(countries), size=n_users)
    acquisition_channels = np.array(channels)[channel_idx]
    country_codes = np.array(countries)[country_idx]

    user_ids = [fake.uuid4() for _ in range(n_users)]
    day_offsets = rng.integers(0, 366, size=n_users)
    acquisition_dates = [
        (start_date + timedelta(days=int(offset))).strftime("%Y-%m-%d")
        for offset in day_offsets
    ]

    device_types = rng.choice(DEVICE_TYPES, size=n_users)
    age_groups = rng.choice(
        AGE_GROUPS, size=n_users, p=[0.30, 0.35, 0.25, 0.10]
    )

    cost_per_install = np.array(
        [
            CHANNEL_QUALITY[ch]["cpi_base"] * rng.uniform(0.85, 1.15)
            for ch in acquisition_channels
        ]
    )

    channel_session_boost = np.array(
        [CHANNEL_QUALITY[ch]["avg_ltv_multiplier"] * 0.15 for ch in acquisition_channels]
    )
    country_engagement = np.array(
        [COUNTRY_ENGAGEMENT[c] for c in country_codes]
    )
    device_boost = np.where(device_types == "iOS", 1.08, 1.0)

    battle_mean = np.clip(
        10 + channel_session_boost * 12 + (country_engagement - 1) * 6, 4, 45
    )
    battles_week1 = np.clip(
        rng.poisson(battle_mean) * device_boost, 1, 50
    ).astype(int)

    playtime_week1 = np.clip(
        battles_week1 * rng.uniform(0.35, 0.75, size=n_users) + rng.normal(0, 1.5, n_users),
        0.5,
        45.0,
    )
    arena_level = np.clip(
        1 + (battles_week1 * rng.uniform(0.12, 0.35, size=n_users)).astype(int)
        + rng.integers(0, 3, size=n_users),
        1,
        15,
    )
    clan_donations_week1 = np.clip(
        (battles_week1 * rng.uniform(0.4, 1.8, size=n_users)).astype(int),
        0,
        200,
    )

    retention_boost = np.array(
        [CHANNEL_QUALITY[ch]["retention_boost"] for ch in acquisition_channels]
    )
    base_retention = np.array(
        [CHANNEL_QUALITY[ch]["base_retention"] for ch in acquisition_channels]
    )
    # Week-1 battles and arena progression (Clash Royale engagement signals).
    engagement_index = (
        (battles_week1 / 50.0) * 0.35
        + (playtime_week1 / 45.0) * 0.20
        + (arena_level / 15.0) * 0.25
        + (clan_donations_week1 / 200.0) * 0.10
        + retention_boost
        + (country_engagement - 1.0) * 0.08
        + (base_retention - 0.40) * 0.35
    )
    retention_prob = 1.0 / (1.0 + np.exp(-12.0 * (engagement_index - 0.42)))
    retention_prob = np.clip(
        retention_prob + rng.normal(0, 0.008, n_users),
        0.01,
        0.99,
    )
    retained_day7 = (rng.random(n_users) < retention_prob).astype(int)

    ltv_multiplier = np.array(
        [CHANNEL_QUALITY[ch]["avg_ltv_multiplier"] for ch in acquisition_channels]
    )
    channel_ltv_effect = ltv_multiplier * 12.0
    ltv_day90 = np.clip(
        battles_week1 * 3.2
        + playtime_week1 * 2.2
        + arena_level * 8.5
        + clan_donations_week1 * 0.45
        + cost_per_install * 2.0
        + channel_ltv_effect
        + retained_day7 * 35.0
        + rng.normal(0, 5.0, n_users),
        0.5,
        500.0,
    )

    ltv_rank = pd.Series(ltv_day90).rank(pct=True).to_numpy()
    purchaser_prob = np.clip(0.05 + ltv_rank * 0.35 + retained_day7 * 0.08, 0, 0.85)
    is_purchaser = (rng.random(n_users) < purchaser_prob).astype(int)

    roas_90d = np.where(cost_per_install > 0, ltv_day90 / cost_per_install, 0.0)

    df = pd.DataFrame(
        {
            "user_id": user_ids,
            "acquisition_date": acquisition_dates,
            "acquisition_channel": acquisition_channels,
            "country": country_codes,
            "device_type": device_types,
            "age_group": age_groups,
            "cost_per_install": np.round(cost_per_install, 2),
            "battles_week1": battles_week1,
            "playtime_week1": np.round(playtime_week1, 2),
            "arena_level": arena_level,
            "clan_donations_week1": clan_donations_week1,
            "retained_day7": retained_day7,
            "ltv_day90": np.round(ltv_day90, 2),
            "is_purchaser": is_purchaser,
            "roas_90d": np.round(roas_90d, 2),
        }
    )

    logger.info(
        "Generated acquisition data: %d users, retention rate %.1f%%, purchaser rate %.1f%%",
        len(df),
        df["retained_day7"].mean() * 100,
        df["is_purchaser"].mean() * 100,
    )
    return df


def generate_transaction_data(
    acquisition_df: pd.DataFrame,
    seed: int,
    observation_period_end: str,
) -> pd.DataFrame:
    """
    Generate purchase history for purchasing users.

    Roughly 20% of users make at least one purchase. A subset are repeat buyers
    to support BG/NBD + Gamma-Gamma modeling in a later phase.

    Args:
        acquisition_df: Acquisition dataset containing purchaser flags and LTV.
        seed: Random seed for reproducibility.
        observation_period_end: Last date in the observation window (YYYY-MM-DD).

    Returns:
        DataFrame with one row per transaction.
    """
    rng = np.random.default_rng(seed + 1)
    end_date = datetime.strptime(observation_period_end, "%Y-%m-%d")

    purchasers = acquisition_df[acquisition_df["is_purchaser"] == 1].copy()
    if purchasers.empty:
        raise ValueError("No purchasers found in acquisition data.")

    records: list[dict[str, str | float]] = []

    for _, user in purchasers.iterrows():
        user_ltv = float(user["ltv_day90"])
        acquisition_date = datetime.strptime(str(user["acquisition_date"]), "%Y-%m-%d")
        days_available = max((end_date - acquisition_date).days, 7)

        if user_ltv > 40 and rng.random() < 0.45:
            n_transactions = int(rng.integers(2, 6))
        else:
            n_transactions = 1

        remaining_ltv = user_ltv
        for txn_index in range(n_transactions):
            if txn_index == n_transactions - 1:
                amount = max(round(remaining_ltv, 2), 0.99)
            else:
                share = float(rng.uniform(0.15, 0.55))
                amount = max(round(user_ltv * share, 2), 0.99)
                remaining_ltv -= amount

            min_offset = min(3 + txn_index * 7, days_available - 1)
            max_offset = max(min_offset + 1, days_available)
            offset = int(rng.integers(min_offset, max_offset))
            transaction_date = (acquisition_date + timedelta(days=offset)).strftime(
                "%Y-%m-%d"
            )
            transaction_type = str(
                rng.choice(
                    TRANSACTION_TYPES,
                    p=[0.50, 0.25, 0.15, 0.10],
                )
            )

            records.append(
                {
                    "user_id": user["user_id"],
                    "transaction_date": transaction_date,
                    "transaction_amount": amount,
                    "transaction_type": transaction_type,
                }
            )

    transactions_df = pd.DataFrame(records)
    repeat_buyers = transactions_df.groupby("user_id").size()
    logger.info(
        "Generated transaction data: %d transactions across %d purchasers (%d repeat buyers)",
        len(transactions_df),
        transactions_df["user_id"].nunique(),
        int((repeat_buyers > 1).sum()),
    )
    return transactions_df


def validate_correlations(acquisition_df: pd.DataFrame) -> None:
    """
    Log sanity checks for key business correlations in the acquisition dataset.

    Args:
        acquisition_df: Generated acquisition DataFrame.
    """
    battle_retention_corr = acquisition_df["battles_week1"].corr(
        acquisition_df["retained_day7"]
    )
    battle_ltv_corr = acquisition_df["battles_week1"].corr(acquisition_df["ltv_day90"])
    channel_ltv = acquisition_df.groupby("acquisition_channel")["ltv_day90"].mean()

    logger.info("Correlation battles_week1 vs retained_day7: %.3f", battle_retention_corr)
    logger.info("Correlation battles_week1 vs ltv_day90: %.3f", battle_ltv_corr)
    logger.info("Average LTV by channel:\n%s", channel_ltv.sort_values(ascending=False))


def save_datasets(
    acquisition_df: pd.DataFrame,
    transactions_df: pd.DataFrame,
    acquisition_path: str,
    transaction_path: str,
) -> None:
    """
    Write acquisition and transaction datasets to CSV.

    Args:
        acquisition_df: Acquisition dataset.
        transactions_df: Transaction dataset.
        acquisition_path: Output path for acquisition CSV.
        transaction_path: Output path for transaction CSV.
    """
    project_root = Path(__file__).parent.parent
    acquisition_file = project_root / acquisition_path
    transaction_file = project_root / transaction_path

    acquisition_file.parent.mkdir(parents=True, exist_ok=True)
    transaction_file.parent.mkdir(parents=True, exist_ok=True)

    acquisition_df.to_csv(acquisition_file, index=False)
    transactions_df.to_csv(transaction_file, index=False)

    logger.info("Saved acquisition data to %s", acquisition_file)
    logger.info("Saved transaction data to %s", transaction_file)


def main() -> None:
    """Load config, generate datasets, validate correlations, and save CSVs."""
    setup_logging()
    config = load_config()

    n_users = int(config["data"]["n_users"])
    seed = int(config["data"]["random_seed"])
    observation_period_end = str(config["data"]["observation_period_end"])
    channels = list(config["channels"])
    countries = list(config["countries"])

    acquisition_df = generate_acquisition_data(
        n_users=n_users,
        seed=seed,
        channels=channels,
        countries=countries,
        observation_period_end=observation_period_end,
    )
    transactions_df = generate_transaction_data(
        acquisition_df=acquisition_df,
        seed=seed,
        observation_period_end=observation_period_end,
    )

    validate_correlations(acquisition_df)
    save_datasets(
        acquisition_df=acquisition_df,
        transactions_df=transactions_df,
        acquisition_path=str(config["data"]["acquisition_path"]),
        transaction_path=str(config["data"]["transaction_path"]),
    )


if __name__ == "__main__":
    main()
