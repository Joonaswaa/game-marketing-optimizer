"""Feature preprocessing pipeline for XGBoost models."""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

CATEGORICAL_FEATURES: list[str] = [
    "acquisition_channel",
    "country",
    "device_type",
    "clan_member",
    "pass_royale_active",
]

NUMERIC_FEATURES: list[str] = [
    "cost_per_install",
    "king_level",
    "trophies_end_week1",
    "arena_id",
    "battles_week1",
    "win_rate_week1",
    "cards_upgraded_week1",
]

FEATURE_COLUMNS: list[str] = CATEGORICAL_FEATURES + NUMERIC_FEATURES

RETENTION_TARGET = "retained_day7"
LTV_TARGET = "ltv_day90"

# Trophy gates for arena assignment (simplified ladder inspired by Clash Royale progression).
ARENA_TROPHY_GATES: list[int] = [
    0,
    300,
    600,
    900,
    1200,
    1500,
    1800,
    2100,
    2400,
    2700,
    3000,
    3300,
    3600,
    3900,
    4200,
]


def trophies_to_arena(trophies: int | float) -> int:
    """
    Map trophy count to arena id (1–15) using simplified trophy gates.

    Args:
        trophies: Player trophy count at end of week 1.

    Returns:
        Arena id between 1 and 15.
    """
    trophy_count = max(int(trophies), 0)
    arena_id = 1
    for index, gate in enumerate(ARENA_TROPHY_GATES):
        if trophy_count >= gate:
            arena_id = index + 1
    return min(arena_id, len(ARENA_TROPHY_GATES))


def build_preprocessor() -> ColumnTransformer:
    """
    Build a ColumnTransformer for categorical and numeric model features.

    Categorical columns are ordinally encoded with unknown-category handling.
    Numeric columns are standardized for XGBoost stability.

    Returns:
        Fitted-ready sklearn ColumnTransformer.
    """
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                CATEGORICAL_FEATURES,
            ),
            ("num", StandardScaler(), NUMERIC_FEATURES),
        ]
    )
