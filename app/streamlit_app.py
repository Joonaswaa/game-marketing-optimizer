"""Game Marketing Campaign Optimizer — Streamlit dashboard."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st
from lifetimes.utils import summary_data_from_transaction_data

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.budget_optimizer import optimize_budget
from src.preprocessing import FEATURE_COLUMNS
from src.utils import load_bgnbd_models, load_config

logger = logging.getLogger(__name__)

AGE_GROUPS = ["18-24", "25-34", "35-44", "45+"]
DEVICE_TYPES = ["iOS", "Android"]


def _resolve_path(relative_path: str) -> Path:
    """Resolve a config path relative to the project root."""
    return PROJECT_ROOT / relative_path


@st.cache_data
def load_acquisition_data() -> pd.DataFrame:
    """Load acquisition CSV with caching."""
    config = load_config()
    return pd.read_csv(_resolve_path(config["data"]["acquisition_path"]))


@st.cache_data
def load_transaction_data() -> pd.DataFrame:
    """Load transaction CSV with caching."""
    config = load_config()
    df = pd.read_csv(_resolve_path(config["data"]["transaction_path"]))
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    return df


@st.cache_data
def load_channel_rfm_profiles() -> pd.DataFrame:
    """
    Build channel-level median RFM profiles for BG/NBD predictions.

    Returns:
        DataFrame indexed by acquisition_channel with median RFM metrics.
    """
    config = load_config()
    transactions = load_transaction_data()
    acquisition = load_acquisition_data()

    summary = summary_data_from_transaction_data(
        transactions,
        customer_id_col="user_id",
        datetime_col="transaction_date",
        monetary_value_col="transaction_amount",
        observation_period_end=config["data"]["observation_period_end"],
    )
    summary = summary.reset_index().rename(columns={"user_id": "user_id"})

    channel_map = acquisition[["user_id", "acquisition_channel"]]
    summary = summary.merge(channel_map, on="user_id", how="left")
    summary = summary.dropna(subset=["acquisition_channel"])

    repeat_buyers = summary[summary["frequency"] > 0]
    profiles = (
        repeat_buyers.groupby("acquisition_channel")[
            ["frequency", "recency", "T", "monetary_value"]
        ]
        .median()
        .reset_index()
    )
    return profiles


@st.cache_resource
def load_retention_model():
    """Load retention classifier pipeline."""
    config = load_config()
    path = _resolve_path(config["models"]["retention_path"])
    if not path.exists():
        return None
    return joblib.load(path)


@st.cache_resource
def load_ltv_model():
    """Load LTV regressor pipeline."""
    config = load_config()
    path = _resolve_path(config["models"]["ltv_xgboost_path"])
    if not path.exists():
        return None
    return joblib.load(path)


@st.cache_resource
def load_bgnbd_bundle():
    """Load BG/NBD and Gamma-Gamma models."""
    config = load_config()
    path = _resolve_path(config["models"]["bgnbd_path"])
    if not path.exists():
        return None
    return load_bgnbd_models(config["models"]["bgnbd_path"])


def page_overview() -> None:
    """Campaign overview with KPIs and channel performance charts."""
    st.title("Campaign Overview")
    st.caption("High-level performance across all acquisition channels.")

    df = load_acquisition_data()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Users", f"{len(df):,}")
    col2.metric("Avg LTV", f"${df['ltv_day90'].mean():.2f}")
    col3.metric("Day-7 Retention Rate", f"{df['retained_day7'].mean():.1%}")
    col4.metric("Total Ad Spend", f"${df['cost_per_install'].sum():,.0f}")

    cpa_df = (
        df.groupby("acquisition_channel", as_index=False)["cost_per_install"]
        .mean()
        .sort_values("cost_per_install", ascending=False)
    )
    cpa_fig = px.bar(
        cpa_df,
        x="acquisition_channel",
        y="cost_per_install",
        title="Average CPA by Channel",
        labels={"acquisition_channel": "Channel", "cost_per_install": "CPA ($)"},
        template="plotly_white",
    )
    st.plotly_chart(cpa_fig, use_container_width=True)

    roas_df = df.copy()
    roas_df["acquisition_date"] = pd.to_datetime(roas_df["acquisition_date"])
    roas_df["month"] = roas_df["acquisition_date"].dt.to_period("M").astype(str)
    roas_trend = (
        roas_df.groupby(["month", "acquisition_channel"], as_index=False)
        .agg(revenue=("ltv_day90", "sum"), spend=("cost_per_install", "sum"))
    )
    roas_trend["roas"] = roas_trend["revenue"] / roas_trend["spend"]

    roas_fig = px.line(
        roas_trend,
        x="month",
        y="roas",
        color="acquisition_channel",
        title="ROAS Trend Over Time by Channel",
        labels={"month": "Month", "roas": "ROAS", "acquisition_channel": "Channel"},
        template="plotly_white",
        markers=True,
    )
    st.plotly_chart(roas_fig, use_container_width=True)


def _predict_bgnbd_ltv(channel: str) -> float | None:
    """
    Predict BG/NBD LTV using median RFM profile for a channel.

    Args:
        channel: Acquisition channel name.

    Returns:
        Predicted 90-day LTV or None if models or profile are unavailable.
    """
    bundle = load_bgnbd_bundle()
    if bundle is None:
        return None

    bgf, ggf = bundle
    profiles = load_channel_rfm_profiles()
    channel_profile = profiles[profiles["acquisition_channel"] == channel]
    if channel_profile.empty:
        return None

    row = channel_profile.iloc[0]
    ltv = ggf.customer_lifetime_value(
        bgf,
        frequency=pd.Series([row["frequency"]], index=[0]),
        recency=pd.Series([row["recency"]], index=[0]),
        T=pd.Series([row["T"]], index=[0]),
        monetary_value=pd.Series([row["monetary_value"]], index=[0]),
        time=3,
        freq="D",
        discount_rate=0.01,
    )
    return float(ltv.iloc[0])


def page_predictions() -> None:
    """Player retention and LTV prediction form."""
    st.title("Player Predictions")
    st.caption("Predict Day-7 retention and 90-day LTV for a new player profile.")

    config = load_config()
    channels = config["channels"]
    countries = config["countries"]

    retention_model = load_retention_model()
    ltv_model = load_ltv_model()

    if retention_model is None or ltv_model is None:
        st.error(
            "Models not found. Run:\n"
            "`py src/train_retention_model.py`\n"
            "`py src/train_ltv_model.py`"
        )
        return

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            channel = st.selectbox("Acquisition Channel", channels)
            country = st.selectbox("Country", countries)
            device = st.selectbox("Device Type", DEVICE_TYPES)
            age_group = st.selectbox("Age Group", AGE_GROUPS)
        with col2:
            sessions = st.number_input("Sessions (Week 1)", min_value=1, max_value=30, value=10)
            playtime = st.number_input(
                "Playtime (Week 1, hours)", min_value=0.5, max_value=45.0, value=8.0, step=0.5
            )
            levels = st.number_input("Levels Completed", min_value=0, max_value=120, value=15)
            social = st.number_input(
                "Social Interactions", min_value=0, max_value=80, value=10
            )
            cpi = st.number_input(
                "Cost Per Install ($)", min_value=0.5, max_value=20.0, value=3.0, step=0.1
            )

        has_purchase_history = st.checkbox(
            "Include purchase history (enables BG/NBD comparison)",
            value=False,
        )
        submitted = st.form_submit_button("Predict")

    if submitted:
        input_df = pd.DataFrame(
            [
                {
                    "acquisition_channel": channel,
                    "country": country,
                    "device_type": device,
                    "age_group": age_group,
                    "cost_per_install": cpi,
                    "sessions_week1": sessions,
                    "playtime_week1": playtime,
                    "levels_completed": levels,
                    "social_interactions": social,
                }
            ]
        )

        retention_prob = float(retention_model.predict_proba(input_df[FEATURE_COLUMNS])[0, 1])
        ltv_estimate = float(ltv_model.predict(input_df[FEATURE_COLUMNS])[0])

        st.subheader("XGBoost Predictions")
        m1, m2 = st.columns(2)
        m1.metric("Day-7 Retention Probability", f"{retention_prob:.1%}")
        m2.metric("90-Day LTV Estimate", f"${ltv_estimate:,.2f}")

        st.subheader("BG/NBD LTV Comparison")
        if has_purchase_history:
            bgnbd_ltv = _predict_bgnbd_ltv(channel)
            if bgnbd_ltv is None:
                st.warning(
                    "BG/NBD model or channel RFM profile unavailable. "
                    "Run `py src/train_bgnbd_model.py` first."
                )
            else:
                diff = ltv_estimate - bgnbd_ltv
                st.metric(
                    "BG/NBD 90-Day LTV (channel median purchaser)",
                    f"${bgnbd_ltv:,.2f}",
                    delta=f"{diff:+,.2f} vs XGBoost",
                )
                st.info(
                    "BG/NBD uses median repeat-buyer RFM stats for the selected channel. "
                    "XGBoost uses the full player profile above."
                )
        else:
            st.info(
                "Enable **Include purchase history** to compare against the BG/NBD model. "
                "BG/NBD applies to users with transaction history."
            )


def page_optimizer() -> None:
    """Budget allocation optimizer across ad channels."""
    st.title("Budget Optimizer")
    st.caption("Allocate spend across channels to maximize expected return.")

    config = load_config()
    channels = config["channels"]
    df = load_acquisition_data()

    predicted_roas = (
        df.groupby("acquisition_channel")["roas_90d"].mean().to_dict()
    )

    st.subheader("Inputs")
    total_budget = st.slider(
        "Total Budget ($)",
        min_value=1_000,
        max_value=1_000_000,
        value=100_000,
        step=1_000,
    )

    st.write("Minimum spend per channel")
    min_cols = st.columns(len(channels))
    min_spends: dict[str, float] = {}
    for col, channel in zip(min_cols, channels):
        with col:
            min_spends[channel] = float(
                st.number_input(
                    channel,
                    min_value=0.0,
                    max_value=float(total_budget),
                    value=1_000.0,
                    step=500.0,
                    key=f"min_{channel}",
                )
            )

    st.subheader("Predicted ROAS by Channel")
    roas_display = pd.DataFrame(
        {"Channel": channels, "Predicted ROAS": [predicted_roas[ch] for ch in channels]}
    )
    st.dataframe(roas_display, use_container_width=True, hide_index=True)

    if st.button("Optimize"):
        try:
            result = optimize_budget(total_budget, min_spends, predicted_roas)
        except ValueError as exc:
            st.error(str(exc))
            return

        st.success(f"Total Expected Return: ${result['total_expected_return']:,.2f}")

        allocation_df = pd.DataFrame(
            {
                "Channel": channels,
                "Allocation ($)": [result["allocations"][ch] for ch in channels],
                "Share (%)": [result["percentages"][ch] for ch in channels],
                "Expected Return ($)": [result["expected_returns"][ch] for ch in channels],
            }
        )
        st.dataframe(allocation_df, use_container_width=True, hide_index=True)

        pie_fig = px.pie(
            allocation_df,
            names="Channel",
            values="Allocation ($)",
            title="Budget Allocation by Channel",
            template="plotly_white",
        )
        st.plotly_chart(pie_fig, use_container_width=True)

        bar_fig = px.bar(
            allocation_df,
            x="Channel",
            y="Expected Return ($)",
            title="Expected Return by Channel",
            template="plotly_white",
        )
        st.plotly_chart(bar_fig, use_container_width=True)


def main() -> None:
    """Run the multipage Streamlit application."""
    st.set_page_config(
        page_title="Game Marketing Campaign Optimizer",
        page_icon="📊",
        layout="wide",
    )

    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Campaign Overview", "Player Predictions", "Budget Optimizer"],
        label_visibility="collapsed",
    )

    if page == "Campaign Overview":
        page_overview()
    elif page == "Player Predictions":
        page_predictions()
    else:
        page_optimizer()


if __name__ == "__main__":
    main()
