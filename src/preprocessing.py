"""Feature preprocessing pipeline for XGBoost models."""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from churn_features import CHURN_FEATURE_COLUMNS
from features import CATEGORICAL_FEATURES, NUMERIC_FEATURES

FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES
RETENTION_TARGET = "retained_day7"
LTV_TARGET = "ltv_day90"


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


def build_churn_preprocessor() -> ColumnTransformer:
    """
    Build a scaler-only preprocessor for churn model numeric features.

    Returns:
        Fitted-ready ColumnTransformer for CHURN_FEATURE_COLUMNS.
    """
    return ColumnTransformer(
        transformers=[("num", StandardScaler(), CHURN_FEATURE_COLUMNS)],
    )
