"""Churn model feature names and target (no sklearn dependency)."""

CHURN_TARGET = "churned"

# Excludes days_since_last_login — that field defines the churn label (>=14 days idle).
CHURN_FEATURE_COLUMNS: list[str] = [
    "king_level",
    "trophies",
    "win_rate",
    "matches_played",
    "days_active",
    "pass_royale",
    "avg_sessions_per_day",
    "avg_battle_duration",
    "total_spend",
]
