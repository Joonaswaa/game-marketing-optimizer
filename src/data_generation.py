"""Generate synthetic Clash Royale player acquisition and IAP transaction data."""

import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker

from features import trophies_to_arena
from utils import load_config, setup_logging

logger = logging.getLogger(__name__)

CHANNEL_QUALITY: dict[str, dict[str, float]] = {
    "TikTok": {
        "channel_prob": 0.25,
        "cpi_base": 1.80,
        "base_retention": 0.33,
        "retention_boost": -0.05,
        "avg_ltv_multiplier": 1.0,
        "skill_bias": -0.08,
    },
    "Facebook": {
        "channel_prob": 0.20,
        "cpi_base": 3.20,
        "base_retention": 0.40,
        "retention_boost": 0.05,
        "avg_ltv_multiplier": 1.5,
        "skill_bias": 0.02,
    },
    "Google": {
        "channel_prob": 0.30,
        "cpi_base": 3.80,
        "base_retention": 0.42,
        "retention_boost": 0.10,
        "avg_ltv_multiplier": 2.0,
        "skill_bias": 0.05,
    },
    "Instagram": {
        "channel_prob": 0.15,
        "cpi_base": 2.90,
        "base_retention": 0.37,
        "retention_boost": 0.02,
        "avg_ltv_multiplier": 1.2,
        "skill_bias": 0.0,
    },
    "YouTube": {
        "channel_prob": 0.10,
        "cpi_base": 4.50,
        "base_retention": 0.48,
        "retention_boost": 0.15,
        "avg_ltv_multiplier": 2.5,
        "skill_bias": 0.10,
    },
}

DEVICE_TYPES = ["iOS", "Android"]
TRANSACTION_TYPES = ["gem_pack", "pass_royale", "chest_offer", "evo_shards", "gold_offer"]

COUNTRY_ENGAGEMENT: dict[str, float] = {
    "US": 1.10,
    "UK": 1.05,
    "DE": 1.00,
    "JP": 1.15,
    "BR": 0.90,
    "FR": 0.95,
    "KR": 1.20,
}

TROPHIES_PER_WIN = 30
TROPHIES_PER_LOSS = 27


def generate_acquisition_data(
    n_users: int,
    seed: int,
    channels: list[str],
    countries: list[str],
    observation_period_end: str,
) -> pd.DataFrame:
    """
    Generate synthetic Clash Royale player profiles with game-accurate progression logic.

    New players start at low king level and ~0 trophies. Week-1 battles produce wins/losses,
    trophy movement, arena placement, card upgrades, clan join, and optional Pass Royale.
    Retention and 90-day IAP follow from skill loop outcomes (win rate, progression, spend hooks).

    Args:
        n_users: Number of players to generate.
        seed: Random seed for reproducibility.
        channels: UA channel names.
        countries: Country codes.
        observation_period_end: Last date in the observation window (YYYY-MM-DD).

    Returns:
        DataFrame with CR player features and retention/IAP targets.
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
    cost_per_install = np.array(
        [
            CHANNEL_QUALITY[ch]["cpi_base"] * rng.uniform(0.85, 1.15)
            for ch in acquisition_channels
        ]
    )

    country_engagement = np.array([COUNTRY_ENGAGEMENT[c] for c in country_codes])
    skill_bias = np.array([CHANNEL_QUALITY[ch]["skill_bias"] for ch in acquisition_channels])
    device_boost = np.where(device_types == "iOS", 1.06, 1.0)

    battle_mean = np.clip(
        8 + country_engagement * 4 + (skill_bias + 0.1) * 20,
        3,
        55,
    )
    battles_week1 = np.clip(
        rng.poisson(battle_mean) * device_boost, 1, 60
    ).astype(int)

    player_skill = (
        rng.normal(0.0, 0.35, size=n_users)
        + skill_bias
        + (country_engagement - 1.0) * 0.15
    )
    win_rate_week1 = np.clip(
        0.38 + player_skill * 0.12 + rng.normal(0, 0.04, n_users),
        0.18,
        0.82,
    )
    wins_week1 = np.clip(
        rng.binomial(battles_week1, win_rate_week1),
        0,
        battles_week1,
    ).astype(int)
    losses_week1 = battles_week1 - wins_week1

    trophies_end_week1 = np.clip(
        wins_week1 * TROPHIES_PER_WIN
        - losses_week1 * TROPHIES_PER_LOSS
        + rng.integers(-20, 40, size=n_users),
        0,
        4500,
    ).astype(int)
    arena_id = np.array([trophies_to_arena(t) for t in trophies_end_week1])

    king_level = np.clip(
        1
        + (battles_week1 // 12)
        + rng.integers(0, 2, size=n_users),
        1,
        14,
    ).astype(int)

    cards_upgraded_week1 = np.clip(
        (wins_week1 * rng.uniform(0.4, 1.1, size=n_users)).astype(int)
        + (king_level * rng.integers(0, 3, size=n_users)),
        0,
        80,
    )

    clan_join_prob = np.clip(
        0.12 + (battles_week1 / 60) * 0.45 + win_rate_week1 * 0.25,
        0.05,
        0.85,
    )
    clan_member = (rng.random(n_users) < clan_join_prob).astype(int)

    pass_prob = np.clip(
        0.04
        + (trophies_end_week1 / 4500) * 0.10
        + win_rate_week1 * 0.06
        + cards_upgraded_week1 / 80 * 0.08,
        0.01,
        0.35,
    )
    pass_royale_active = (rng.random(n_users) < pass_prob).astype(int)

    retention_boost = np.array(
        [CHANNEL_QUALITY[ch]["retention_boost"] for ch in acquisition_channels]
    )
    base_retention = np.array(
        [CHANNEL_QUALITY[ch]["base_retention"] for ch in acquisition_channels]
    )
    engagement_index = (
        (battles_week1 / 60.0) * 0.22
        + win_rate_week1 * 0.28
        + (trophies_end_week1 / 4500.0) * 0.18
        + (arena_id / 15.0) * 0.10
        + clan_member * 0.12
        + pass_royale_active * 0.05
        + retention_boost
        + (base_retention - 0.40) * 0.30
    )
    retention_prob = 1.0 / (1.0 + np.exp(-10.0 * (engagement_index - 0.38)))
    retention_prob = np.clip(
        retention_prob + rng.normal(0, 0.008, n_users),
        0.01,
        0.99,
    )
    retained_day7 = (rng.random(n_users) < retention_prob).astype(int)

    ltv_multiplier = np.array(
        [CHANNEL_QUALITY[ch]["avg_ltv_multiplier"] for ch in acquisition_channels]
    )
    trophy_wall_spend = np.where(
        (trophies_end_week1 >= 500) & (trophies_end_week1 <= 1200),
        12.0,
        0.0,
    )
    ltv_day90 = np.clip(
        pass_royale_active * 62.0
        + cards_upgraded_week1 * 2.4
        + trophies_end_week1 * 0.055
        + king_level * 5.5
        + trophy_wall_spend
        + retained_day7 * 32.0
        + clan_member * 8.0
        + ltv_multiplier * 12.0
        + rng.exponential(scale=4.0, size=n_users)
        + rng.normal(0, 3.0, n_users),
        0.0,
        500.0,
    )

    ltv_rank = pd.Series(ltv_day90).rank(pct=True).to_numpy()
    purchaser_prob = np.clip(
        0.04
        + pass_royale_active * 0.35
        + ltv_rank * 0.30
        + retained_day7 * 0.08
        + (cards_upgraded_week1 / 80) * 0.10,
        0,
        0.88,
    )
    is_purchaser = (rng.random(n_users) < purchaser_prob).astype(int)

    roas_90d = np.where(cost_per_install > 0, ltv_day90 / cost_per_install, 0.0)

    days_active = np.clip(
        rng.integers(2, 45, size=n_users)
        + retained_day7 * rng.integers(10, 55, size=n_users)
        + (battles_week1 // 4),
        1,
        120,
    ).astype(int)

    avg_sessions_per_day = np.clip(
        battles_week1 / 7.0 * rng.uniform(0.75, 1.25, size=n_users)
        + retained_day7 * 0.35
        + rng.normal(0, 0.25, size=n_users),
        0.2,
        12.0,
    )

    avg_battle_duration = np.clip(
        rng.normal(3.2, 0.9, size=n_users) + battles_week1 * 0.015,
        1.0,
        8.0,
    )

    matches_played = np.clip(
        battles_week1 + rng.integers(0, 180, size=n_users) * retained_day7,
        1,
        500,
    ).astype(int)

    total_spend = np.round(ltv_day90 * rng.uniform(0.80, 1.20, size=n_users), 2)

    # Login gap from engagement (used for churn label only — not a model feature).
    login_gap = (
        (1 - retained_day7) * rng.integers(6, 20, size=n_users)
        + (1 - win_rate_week1) * 5
        + np.maximum(0, 12 - battles_week1 // 5)
        + (1 - clan_member) * 3
        - pass_royale_active * 4
        - np.minimum(avg_sessions_per_day, 6) * 1.5
        + rng.exponential(2.0, size=n_users)
    )
    days_since_last_login = np.clip(login_gap.astype(int), 0, 90)

    # Spec: churned when idle 14+ days; label is not leaked into XGBoost features.
    churned = (days_since_last_login >= 14).astype(int)

    last_login_dates: list[str] = []
    for index in range(n_users):
        acq_dt = datetime.strptime(acquisition_dates[index], "%Y-%m-%d")
        max_offset = max((end_date - acq_dt).days, 0)
        if churned[index] == 1:
            offset = max(days_active[index] - days_since_last_login[index], 0)
        else:
            offset = min(days_active[index] - 1, max_offset)
        offset = int(np.clip(offset, 0, max_offset))
        last_login_dates.append((acq_dt + timedelta(days=offset)).strftime("%Y-%m-%d"))

    df = pd.DataFrame(
        {
            "user_id": user_ids,
            "acquisition_date": acquisition_dates,
            "install_date": acquisition_dates,
            "last_login_date": last_login_dates,
            "acquisition_channel": acquisition_channels,
            "country": country_codes,
            "device_type": device_types,
            "king_level": king_level,
            "trophies_end_week1": trophies_end_week1,
            "arena_id": arena_id,
            "battles_week1": battles_week1,
            "win_rate_week1": np.round(win_rate_week1, 3),
            "cards_upgraded_week1": cards_upgraded_week1,
            "clan_member": clan_member,
            "pass_royale_active": pass_royale_active,
            "cost_per_install": np.round(cost_per_install, 2),
            "retained_day7": retained_day7,
            "ltv_day90": np.round(ltv_day90, 2),
            "is_purchaser": is_purchaser,
            "roas_90d": np.round(roas_90d, 2),
            "trophies": trophies_end_week1,
            "win_rate": np.round(win_rate_week1, 3),
            "matches_played": matches_played,
            "pass_royale": pass_royale_active,
            "days_active": days_active,
            "days_since_last_login": days_since_last_login,
            "avg_sessions_per_day": np.round(avg_sessions_per_day, 2),
            "avg_battle_duration": np.round(avg_battle_duration, 2),
            "total_spend": total_spend,
            "churned": churned,
        }
    )

    logger.info(
        "Generated CR player data: %d users, retention %.1f%%, churn %.1f%%, "
        "pass rate %.1f%%, clan rate %.1f%%, purchaser rate %.1f%%",
        len(df),
        df["retained_day7"].mean() * 100,
        df["churned"].mean() * 100,
        df["pass_royale_active"].mean() * 100,
        df["clan_member"].mean() * 100,
        df["is_purchaser"].mean() * 100,
    )
    return df


def generate_transaction_data(
    acquisition_df: pd.DataFrame,
    seed: int,
    observation_period_end: str,
) -> pd.DataFrame:
    """
    Generate gem/IAP purchase history aligned with Clash Royale spend patterns.

    Pass Royale buyers get recurring pass transactions; gem packs dominate one-off spend.

    Args:
        acquisition_df: Acquisition dataset with purchaser flags and IAP totals.
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
        has_pass = int(user["pass_royale_active"]) == 1
        acquisition_date = datetime.strptime(str(user["acquisition_date"]), "%Y-%m-%d")
        days_available = max((end_date - acquisition_date).days, 7)

        if has_pass and user_ltv > 25:
            n_transactions = int(rng.integers(2, 5))
        elif user_ltv > 45 and rng.random() < 0.40:
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

            min_offset = min(2 + txn_index * 10, days_available - 1)
            max_offset = max(min_offset + 1, days_available)
            offset = int(rng.integers(min_offset, max_offset))
            transaction_date = (acquisition_date + timedelta(days=offset)).strftime(
                "%Y-%m-%d"
            )

            if has_pass and txn_index == 0:
                transaction_type = "pass_royale"
            else:
                transaction_type = str(
                    rng.choice(
                        TRANSACTION_TYPES,
                        p=[0.45, 0.10, 0.20, 0.15, 0.10],
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
    Log sanity checks for Clash Royale progression and monetization correlations.

    Args:
        acquisition_df: Generated acquisition DataFrame.
    """
    win_retention = acquisition_df["win_rate_week1"].corr(acquisition_df["retained_day7"])
    trophy_ltv = acquisition_df["trophies_end_week1"].corr(acquisition_df["ltv_day90"])
    pass_ltv = acquisition_df["pass_royale_active"].corr(acquisition_df["ltv_day90"])
    channel_ltv = acquisition_df.groupby("acquisition_channel")["ltv_day90"].mean()

    logger.info("Correlation win_rate_week1 vs retained_day7: %.3f", win_retention)
    logger.info("Correlation trophies_end_week1 vs ltv_day90: %.3f", trophy_ltv)
    logger.info("Correlation pass_royale_active vs ltv_day90: %.3f", pass_ltv)
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
