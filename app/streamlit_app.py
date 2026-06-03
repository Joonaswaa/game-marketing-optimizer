"""Game Marketing Campaign Optimizer — Streamlit dashboard."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import logging
from typing import Any

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_option_menu import option_menu

from src.cohort_analysis import (
    average_retention_metrics,
    build_retention_matrix,
    prepare_cohort_frame,
    retention_curve_averages,
)
from src.churn_features import CHURN_FEATURE_COLUMNS
from src.features import FEATURE_COLUMNS, trophies_to_arena
from src.utils import load_bgnbd_models, load_config

logger = logging.getLogger(__name__)

DEVICE_TYPES = ["iOS", "Android"]

CHANNEL_COLORS: dict[str, str] = {
    "TikTok": "#6366F1",
    "YouTube": "#EF4444",
    "Instagram": "#EC4899",
    "Facebook": "#3B82F6",
    "Google": "#10B981",
}

APP_TITLE = "UA Optimizer"
APP_TAGLINE = (
    "Synthetic Clash Royale player data · retention & IAP ML · UA budget optimization"
)

NAV_OPTIONS: list[str] = [
    "Campaign Overview",
    "Player Predictions",
    "Churn Prediction",
    "Cohort Analysis",
    "Budget Optimizer",
    "Metodologia & Info",
]
NAV_ICONS: list[str] = [
    "bar-chart",
    "person-badge",
    "person-dash",
    "grid-3x3-gap",
    "wallet2",
    "info-circle",
]

PLOTLY_LAYOUT: dict[str, Any] = {
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "font": {"family": "Inter, sans-serif", "color": "#334155", "size": 13},
    "margin": {"l": 24, "r": 24, "t": 48, "b": 24},
    "hovermode": "x unified",
    "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
}


def format_usd_k(value: float) -> str:
    """
    Format a dollar amount rounded to the nearest thousand.

    Example: 20525.99 -> "$21000"

    Args:
        value: Dollar amount.

    Returns:
        Formatted string; values under $500 keep two decimal places.
    """
    amount = float(value)
    if abs(amount) < 500:
        return f"${amount:,.2f}"
    rounded = int(round(amount / 1000) * 1000)
    return f"${rounded:,}"


def format_count_k(value: float) -> str:
    """
    Format a large count rounded to the nearest thousand.

    Args:
        value: Numeric count (e.g. total players).

    Returns:
        Formatted string such as "100,000" for 100_000.
    """
    rounded = int(round(float(value) / 1000) * 1000)
    return f"{rounded:,}"


def _resolve_path(relative_path: str) -> Path:
    """Resolve a config path relative to the project root."""
    return PROJECT_ROOT / relative_path


def inject_custom_css() -> None:
    """Inject global styles for a modern, responsive dashboard look."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        section[data-testid="stSidebar"] {
            display: none;
        }
        [data-testid="collapsedControl"] {
            display: none;
        }

        .main .block-container {
            padding-top: 1rem;
            padding-bottom: 2.5rem;
            max-width: 1280px;
            background: #f1f5f9;
        }

        /* Chrome: header + tabs in one card; page body in another */
        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: #ffffff;
            border-color: #e2e8f0 !important;
            border-radius: 14px !important;
            box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06);
            padding: 0.35rem 0.5rem 0.5rem;
            margin-bottom: 1rem;
        }

        .app-shell-header h1 {
            font-size: 1.65rem;
            font-weight: 800;
            color: #0f172a;
            margin: 0 0 0.2rem 0;
            letter-spacing: -0.03em;
        }

        .app-shell-header .app-tagline {
            color: #64748b;
            font-size: 0.9rem;
            margin: 0 0 0.75rem 0;
            line-height: 1.45;
        }

        .app-shell-nav-divider {
            height: 1px;
            background: #e2e8f0;
            margin: 0 0 0.5rem 0;
        }

        .page-header {
            margin: 0 0 1rem 0;
            padding-bottom: 0.75rem;
            border-bottom: 1px solid #f1f5f9;
        }

        .page-header h2 {
            font-size: 1.35rem;
            font-weight: 700;
            color: #1e3a8a;
            margin: 0 0 0.25rem 0;
        }

        .page-header p {
            margin: 0;
            color: #64748b;
            font-size: 0.92rem;
        }

        .hero-banner {
            background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 55%, #d97706 100%);
            border-radius: 12px;
            padding: 0.9rem 1.25rem;
            margin-bottom: 1rem;
            color: #ffffff;
            box-shadow: 0 8px 24px -8px rgba(30, 58, 138, 0.35);
        }

        .hero-banner h1 {
            font-size: 1.25rem;
            font-weight: 700;
            margin: 0 0 0.2rem 0;
            letter-spacing: -0.02em;
        }

        .hero-banner p {
            margin: 0;
            opacity: 0.92;
            font-size: 0.88rem;
            font-weight: 400;
        }

        .hero-badge {
            display: none;
        }

        .info-section-title {
            font-size: 1rem;
            font-weight: 700;
            color: #1e3a8a;
            margin: 0 0 0.5rem 0;
        }

        .pipeline-step {
            display: flex;
            gap: 0.75rem;
            align-items: flex-start;
            margin-bottom: 0.65rem;
        }

        .pipeline-num {
            flex-shrink: 0;
            width: 1.6rem;
            height: 1.6rem;
            border-radius: 999px;
            background: #eff6ff;
            color: #2563eb;
            font-size: 0.8rem;
            font-weight: 700;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .kpi-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 1rem;
            margin-bottom: 1.5rem;
        }

        @media (max-width: 900px) {
            .kpi-grid { grid-template-columns: repeat(2, 1fr); }
        }

        @media (max-width: 520px) {
            .kpi-grid { grid-template-columns: 1fr; }
        }

        .kpi-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 16px;
            padding: 1.15rem 1.25rem;
            box-shadow: 0 4px 6px -1px rgba(15, 23, 42, 0.06);
            transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
        }

        .kpi-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 12px 24px -8px rgba(217, 119, 6, 0.3);
            border-color: #fcd34d;
        }

        .kpi-label {
            display: block;
            font-size: 0.78rem;
            font-weight: 600;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.4rem;
        }

        .kpi-value {
            display: block;
            font-size: 1.65rem;
            font-weight: 800;
            color: #0f172a;
            letter-spacing: -0.02em;
            line-height: 1.1;
        }

        .section-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 18px;
            padding: 0.25rem 0.5rem 0.5rem;
            margin-bottom: 1rem;
            box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
        }

        .result-card {
            background: linear-gradient(135deg, #eff6ff 0%, #fef3c7 100%);
            border: 1px solid #fcd34d;
            border-radius: 16px;
            padding: 1.25rem 1.5rem;
            margin: 1rem 0;
        }

        .result-card .kpi-value { color: #b45309; }

        div.stButton > button[kind="primary"],
        div.stFormSubmitButton > button {
            background: linear-gradient(135deg, #2563eb, #d97706) !important;
            border: none !important;
            border-radius: 12px !important;
            padding: 0.6rem 1.5rem !important;
            font-weight: 600 !important;
            box-shadow: 0 4px 14px rgba(37, 99, 235, 0.4) !important;
            transition: transform 0.15s ease, box-shadow 0.15s ease !important;
        }

        div.stButton > button[kind="primary"]:hover,
        div.stFormSubmitButton > button:hover {
            transform: translateY(-1px) !important;
            box-shadow: 0 8px 20px rgba(217, 119, 6, 0.45) !important;
        }

        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 14px;
            padding: 0.85rem 1rem;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_static_header() -> None:
    """Render app title chrome once; stays mounted across tab switches."""
    with st.container(border=True):
        st.markdown(
            f"""
            <div class="app-shell-header">
                <h1>{APP_TITLE}</h1>
                <p class="app-tagline">{APP_TAGLINE}</p>
            </div>
            <div class="app-shell-nav-divider"></div>
            """,
            unsafe_allow_html=True,
        )


@st.fragment
def render_app_router() -> None:
    """
    Tab navigation and page body — fragment rerun keeps tab switches fast.

    Only this block re-executes when the user changes tabs, not CSS or the header.
    """
    with st.container(border=True):
        selected = render_top_navigation()

    with st.container(border=True):
        _dispatch_page(selected)


def _dispatch_page(selected: str) -> None:
    """Route to the active dashboard page."""
    if selected == "Campaign Overview":
        page_overview()
    elif selected == "Player Predictions":
        page_predictions()
    elif selected == "Churn Prediction":
        page_churn_prediction()
    elif selected == "Cohort Analysis":
        page_cohort_analysis()
    elif selected == "Budget Optimizer":
        page_optimizer()
    else:
        page_methodology()


def render_top_navigation() -> str:
    """
    Render horizontal navbar and return the selected page label.

    Returns:
        Selected navigation option string.
    """
    return option_menu(
        menu_title=None,
        options=NAV_OPTIONS,
        icons=NAV_ICONS,
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
        styles={
            "container": {
                "padding": "0.25rem 0.5rem!important",
                "background-color": "#f8fafc",
                "border-radius": "10px",
                "margin": "0!important",
            },
            "icon": {"font-size": "15px", "color": "#475569"},
            "nav-link": {
                "font-size": "14px",
                "text-align": "center",
                "margin": "0 2px",
                "padding": "8px 12px",
                "border-radius": "8px",
                "--hover-color": "#e2e8f0",
            },
            "nav-link-selected": {
                "background-color": "#2563eb",
                "color": "#ffffff",
                "font-weight": "600",
            },
        },
    )


def render_page_header(title: str, subtitle: str) -> None:
    """Render a compact page title below the global navbar."""
    st.markdown(
        f"""
        <div class="page-header">
            <h2>{title}</h2>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_methodology_expander(title: str, body_md: str) -> None:
    """
    Show collapsible methodology copy on an analysis page.

    Args:
        title: Expander label.
        body_md: Markdown content explaining logic and models.
    """
    with st.expander(title, expanded=False):
        st.markdown(body_md)


def render_kpi_cards(items: list[tuple[str, str]]) -> None:
    """Render a responsive grid of KPI cards."""
    cards_html = "".join(
        f'<div class="kpi-card"><span class="kpi-label">{label}</span>'
        f'<span class="kpi-value">{value}</span></div>'
        for label, value in items
    )
    st.markdown(f'<div class="kpi-grid">{cards_html}</div>', unsafe_allow_html=True)


def style_plotly_fig(fig: go.Figure) -> go.Figure:
    """Apply consistent modern styling to Plotly figures."""
    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_xaxes(showgrid=True, gridcolor="rgba(148,163,184,0.2)", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(148,163,184,0.2)", zeroline=False)
    return fig


def channel_color_map(channels: list[str]) -> dict[str, str]:
    """Return a color map for the given channels."""
    return {ch: CHANNEL_COLORS.get(ch, "#64748B") for ch in channels}


def churn_risk_label(probability: float) -> str:
    """
    Map churn probability to Low / Medium / High risk band.

    Args:
        probability: Predicted churn probability in [0, 1].

    Returns:
        Risk level label.
    """
    if probability < 0.30:
        return "Low"
    if probability < 0.60:
        return "Medium"
    return "High"


@st.cache_data(ttl=300)
def _acquisition_data_mtime() -> float:
    """File modification time used to invalidate Streamlit data cache."""
    config = load_config()
    return _resolve_path(config["data"]["acquisition_path"]).stat().st_mtime


@st.cache_data(ttl=300)
def _transaction_data_mtime() -> float:
    """Transaction CSV mtime for cache invalidation."""
    config = load_config()
    return _resolve_path(config["data"]["transaction_path"]).stat().st_mtime


def _churn_model_mtime() -> float:
    """Model file mtime for cache invalidation after retraining."""
    config = load_config()
    path = _resolve_path(config["models"]["churn_path"])
    return path.stat().st_mtime if path.exists() else 0.0


@st.cache_data(show_spinner=False)
def load_acquisition_data(data_mtime: float) -> pd.DataFrame:
    """Load acquisition CSV with caching (refreshes when the file changes)."""
    config = load_config()
    return pd.read_csv(_resolve_path(config["data"]["acquisition_path"]))


@st.cache_data(show_spinner=False)
def load_transaction_data(txn_mtime: float) -> pd.DataFrame:
    """Load transaction CSV with caching."""
    config = load_config()
    df = pd.read_csv(_resolve_path(config["data"]["transaction_path"]))
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    return df


@st.cache_data(show_spinner=False)
def compute_overview_aggregates(data_mtime: float) -> dict[str, Any]:
    """
    Precompute Campaign Overview KPIs and chart source tables.

    Args:
        data_mtime: Acquisition CSV modification time for cache keying.

    Returns:
        Dict with kpi_cards, cpa_df, and roas_trend DataFrames.
    """
    df = load_acquisition_data(data_mtime)
    roas_df = df.copy()
    roas_df["acquisition_date"] = pd.to_datetime(roas_df["acquisition_date"])
    roas_df["month"] = roas_df["acquisition_date"].dt.to_period("M").astype(str)
    roas_trend = (
        roas_df.groupby(["month", "acquisition_channel"], as_index=False)
        .agg(revenue=("ltv_day90", "sum"), spend=("cost_per_install", "sum"))
    )
    roas_trend["roas"] = roas_trend["revenue"] / roas_trend["spend"]

    return {
        "kpi_cards": [
            ("Players Acquired", format_count_k(len(df))),
            ("Avg 90d IAP", format_usd_k(df["ltv_day90"].mean())),
            ("Day-7 Retention", f"{df['retained_day7'].mean():.1%}"),
            ("Total UA Spend", format_usd_k(df["cost_per_install"].sum())),
        ],
        "cpa_df": (
            df.groupby("acquisition_channel", as_index=False)["cost_per_install"]
            .mean()
            .sort_values("cost_per_install", ascending=True)
        ),
        "roas_trend": roas_trend,
    }


@st.cache_data(show_spinner=False)
def compute_cohort_outputs(data_mtime: float) -> tuple[pd.DataFrame, dict[str, float], pd.DataFrame]:
    """
    Precompute cohort retention matrix and summary metrics.

    Args:
        data_mtime: Acquisition CSV modification time for cache keying.

    Returns:
        Tuple of (retention_matrix, average_metrics, retention_curve_df).
    """
    df = load_acquisition_data(data_mtime)
    cohort_df = prepare_cohort_frame(df)
    retention_matrix = build_retention_matrix(cohort_df)
    averages = average_retention_metrics(retention_matrix)
    curve_df = retention_curve_averages(retention_matrix)
    return retention_matrix, averages, curve_df


@st.cache_data(show_spinner=False)
def compute_channel_roas_stats(data_mtime: float) -> tuple[dict[str, float], dict[str, float]]:
    """
    Precompute per-channel ROAS mean and std for the budget optimizer.

    Args:
        data_mtime: Acquisition CSV modification time for cache keying.

    Returns:
        Tuple of (mean ROAS by channel, std ROAS by channel).
    """
    df = load_acquisition_data(data_mtime)
    predicted_roas = df.groupby("acquisition_channel")["roas_90d"].mean().to_dict()
    roas_std = df.groupby("acquisition_channel")["roas_90d"].std().fillna(0).to_dict()
    return predicted_roas, roas_std


@st.cache_data(show_spinner=False)
def load_channel_rfm_profiles(data_mtime: float, txn_mtime: float) -> pd.DataFrame:
    """
    Build channel-level median RFM profiles for BG/NBD predictions.

    Returns:
        DataFrame indexed by acquisition_channel with median RFM metrics.
    """
    config = load_config()
    transactions = load_transaction_data(txn_mtime)
    acquisition = load_acquisition_data(data_mtime)

    try:
        from lifetimes.utils import summary_data_from_transaction_data
    except ImportError as exc:
        raise ImportError(
            "lifetimes is not installed. BG/NBD profiles are unavailable on this runtime."
        ) from exc

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
    try:
        return load_bgnbd_models(config["models"]["bgnbd_path"])
    except ImportError:
        logger.warning("lifetimes not installed; BG/NBD predictions unavailable")
        return None


@st.cache_resource
def load_churn_model(model_mtime: float) -> Any:
    """Load churn classifier pipeline (refreshes when the model file changes)."""
    config = load_config()
    path = _resolve_path(config["models"]["churn_path"])
    if not path.exists():
        return None
    return joblib.load(path)


@st.cache_data
def load_churn_metrics() -> dict[str, float] | None:
    """Load persisted churn model evaluation metrics."""
    config = load_config()
    path = _resolve_path(config["models"]["churn_metrics_path"])
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def page_overview() -> None:
    """Campaign overview with KPIs and channel performance charts."""
    render_page_header(
        "Campaign Overview",
        "Channel-level UA performance — installs, 90-day IAP, and ROAS (synthetic data).",
    )
    render_methodology_expander(
        "How are these metrics calculated?",
        """
        **Data:** 100k synthetic Clash Royale–style player profiles with correlated week-1
        progression (battles → win rate → trophies → IAP).

        **KPIs:** Totals and means from `acquisition_data.csv`. ROAS = `ltv_day90 / cost_per_install`
        aggregated by channel and month.

        **Charts:** Plotly bar (CPA by channel) and line (ROAS trend). All figures use
        `use_container_width=True` for responsive layout.
        """,
    )

    data_mtime = _acquisition_data_mtime()
    overview = compute_overview_aggregates(data_mtime)

    render_kpi_cards(overview["kpi_cards"])

    chart_col1, chart_col2 = st.columns(2, gap="large")

    cpa_df = overview["cpa_df"]
    cpa_fig = px.bar(
        cpa_df,
        x="cost_per_install",
        y="acquisition_channel",
        orientation="h",
        title="Cost Per Install by Channel",
        labels={"acquisition_channel": "Channel", "cost_per_install": "CPA ($)"},
        color="acquisition_channel",
        color_discrete_map=channel_color_map(cpa_df["acquisition_channel"].tolist()),
    )
    cpa_fig.update_traces(marker_line_width=0, opacity=0.92)
    style_plotly_fig(cpa_fig)

    roas_trend = overview["roas_trend"]
    roas_fig = px.line(
        roas_trend,
        x="month",
        y="roas",
        color="acquisition_channel",
        title="ROAS Trend Over Time",
        labels={"month": "Month", "roas": "ROAS", "acquisition_channel": "Channel"},
        color_discrete_map=channel_color_map(
            roas_trend["acquisition_channel"].unique().tolist()
        ),
        markers=True,
    )
    roas_fig.update_traces(line=dict(width=2.5), marker=dict(size=6))
    style_plotly_fig(roas_fig)

    with chart_col1:
        st.plotly_chart(cpa_fig, use_container_width=True)
    with chart_col2:
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
    try:
        data_mtime = _acquisition_data_mtime()
        txn_mtime = _transaction_data_mtime()
        profiles = load_channel_rfm_profiles(data_mtime, txn_mtime)
    except ImportError:
        return None
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
    """Player retention and IAP prediction form using Clash Royale week-1 profile."""
    render_page_header(
        "Player Predictions",
        "Week-1 profile → Day-7 retention probability and 90-day IAP estimate.",
    )
    render_methodology_expander(
        "How do retention and LTV predictions work?",
        """
        **Day-7 retention:** XGBoost classifier on UA channel, geo, device, king level, trophies,
        battles, win rate, card upgrades, clan, and Pass Royale. Output is
        `predict_proba` for class 1 (retained).

        **90-day IAP (LTV):** XGBoost regressor on the same feature set, trained on synthetic
        `ltv_day90` with progression-driven correlations.

        **BG/NBD (optional):** When enabled, compares XGBoost LTV to a **lifetimes**
        BG/NBD + Gamma-Gamma estimate using median RFM per UA channel for gem purchasers.
        """,
    )

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

    with st.container(border=True):
        st.markdown("##### Week-1 player profile")
        with st.form("prediction_form", border=False):
            col1, col2 = st.columns(2, gap="large")
            with col1:
                st.markdown("**Install & account**")
                channel = st.selectbox("UA channel", channels)
                country = st.selectbox("Country", countries)
                device = st.selectbox("Platform", DEVICE_TYPES)
                king_level = st.slider("King level", min_value=1, max_value=14, value=3)
                cpi = st.slider(
                    "Cost per install ($)", min_value=0.5, max_value=20.0, value=3.0, step=0.1
                )
            with col2:
                st.markdown("**Ladder & economy (week 1)**")
                battles = st.slider("Ladder battles played", min_value=1, max_value=60, value=15)
                win_rate = st.slider(
                    "Win rate", min_value=0.18, max_value=0.82, value=0.52, step=0.01
                )
                trophies = st.slider("Trophies (end of week 1)", min_value=0, max_value=4500, value=400)
                arena = trophies_to_arena(trophies)
                st.caption(f"Arena **{arena}** (derived from trophies)")
                cards_upgraded = st.slider("Cards upgraded", min_value=0, max_value=80, value=8)

            col3, col4 = st.columns(2)
            with col3:
                clan_member = st.checkbox("Joined a clan", value=False)
            with col4:
                pass_royale = st.checkbox("Pass Royale active", value=False)

            has_purchase_history = st.checkbox(
                "Compare BG/NBD (requires prior gem purchases in cohort)",
                value=False,
            )
            submitted = st.form_submit_button("Predict", type="primary", use_container_width=True)

    if submitted:
        input_df = pd.DataFrame(
            [
                {
                    "acquisition_channel": channel,
                    "country": country,
                    "device_type": device,
                    "king_level": king_level,
                    "trophies_end_week1": trophies,
                    "arena_id": arena,
                    "battles_week1": battles,
                    "win_rate_week1": win_rate,
                    "cards_upgraded_week1": cards_upgraded,
                    "clan_member": int(clan_member),
                    "pass_royale_active": int(pass_royale),
                    "cost_per_install": cpi,
                }
            ]
        )

        retention_prob = float(retention_model.predict_proba(input_df[FEATURE_COLUMNS])[0, 1])
        ltv_estimate = float(ltv_model.predict(input_df[FEATURE_COLUMNS])[0])

        st.markdown("##### Model output")
        m1, m2, m3 = st.columns(3, gap="medium")
        with m1:
            st.metric("Day-7 retention", f"{retention_prob:.1%}")
        with m2:
            st.metric("90-day IAP (USD)", format_usd_k(ltv_estimate))
        with m3:
            st.metric("Arena / trophies", f"Arena {arena} · {trophies:,} 🏆")

        st.markdown(
            f"""
            <div class="result-card">
                <span class="kpi-label">Profile summary</span>
                <span class="kpi-value">
                    {battles} battles @ {win_rate:.0%} WR · King {king_level} ·
                    {'clan' if clan_member else 'solo'} ·
                    {retention_prob:.0%} retain · {format_usd_k(ltv_estimate)} IAP
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("##### BG/NBD IAP comparison")
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
                    "BG/NBD 90-day IAP",
                    format_usd_k(bgnbd_ltv),
                    delta=f"{format_usd_k(diff)} vs XGBoost",
                )
                st.info(
                    "BG/NBD uses median gem-purchaser RFM for this UA channel. "
                    "XGBoost uses the full CR player profile above."
                )
        else:
            st.info("Check **Compare BG/NBD** to show cohort-based IAP from transaction history.")


def page_churn_prediction() -> None:
    """Predict 14-day churn risk from player engagement and spend signals."""
    render_page_header(
        "Churn Prediction",
        "14-day churn risk from engagement and spend (not from days since last login).",
    )
    render_methodology_expander(
        "How is churn defined and modeled?",
        """
        **Definition:** A player **churns** if they have not logged in for **14+ days**
        (`days_since_last_login >= 14` in synthetic data).

        **Model:** XGBoost classifier (`n_estimators=200`, `max_depth=5`, `learning_rate=0.05`)
        on king level, trophies, win rate, matches played, days active, Pass Royale,
        sessions/day, battle duration, and total spend.

        **Note:** `days_since_last_login` is **not** a model input (it would leak the label).
        Risk bands: **Low** under 30%, **Medium** 30–60%, **High** 60%+.
        """,
    )

    churn_model = load_churn_model(_churn_model_mtime())
    metrics = load_churn_metrics()

    if churn_model is None:
        st.error(
            "Churn model not found. Run:\n"
            "`py src/data_generation.py`\n"
            "`py src/train_churn_model.py`"
        )
        return

    if metrics:
        st.markdown("##### Model performance (hold-out test)")
        m1, m2, m3, m4 = st.columns(4, gap="medium")
        with m1:
            st.metric("Accuracy", f"{metrics['accuracy']:.1%}")
        with m2:
            st.metric("Precision", f"{metrics['precision']:.1%}")
        with m3:
            st.metric("Recall", f"{metrics['recall']:.1%}")
        with m4:
            st.metric("ROC-AUC", f"{metrics['roc_auc']:.2f}")

    with st.container(border=True):
        st.markdown("##### Player engagement inputs")
        with st.form("churn_form", border=False):
            col1, col2 = st.columns(2, gap="large")
            with col1:
                king_level = st.number_input(
                    "King level", min_value=1, max_value=14, value=5, step=1
                )
                trophies = st.number_input(
                    "Trophies", min_value=0, max_value=4500, value=600, step=50
                )
                win_rate = st.slider(
                    "Win rate", min_value=0.18, max_value=0.82, value=0.50, step=0.01
                )
                matches_played = st.number_input(
                    "Matches played", min_value=1, max_value=500, value=40, step=1
                )
                days_active = st.number_input(
                    "Days active", min_value=1, max_value=120, value=14, step=1
                )
            with col2:
                pass_royale = st.selectbox("Pass Royale", [0, 1], format_func=lambda x: "Yes" if x else "No")
                avg_sessions_per_day = st.slider(
                    "Avg sessions per day",
                    min_value=0.2,
                    max_value=12.0,
                    value=2.5,
                    step=0.1,
                )
                avg_battle_duration = st.slider(
                    "Avg battle duration (min)",
                    min_value=1.0,
                    max_value=8.0,
                    value=3.5,
                    step=0.1,
                )
                total_spend = st.number_input(
                    "Total spend ($)",
                    min_value=0.0,
                    max_value=500.0,
                    value=25.0,
                    step=1.0,
                )

            submitted = st.form_submit_button(
                "Predict Churn Risk", type="primary", use_container_width=True
            )

    if submitted:
        input_df = pd.DataFrame(
            [
                {
                    "king_level": king_level,
                    "trophies": trophies,
                    "win_rate": win_rate,
                    "matches_played": matches_played,
                    "days_active": days_active,
                    "pass_royale": pass_royale,
                    "avg_sessions_per_day": avg_sessions_per_day,
                    "avg_battle_duration": avg_battle_duration,
                    "total_spend": total_spend,
                }
            ]
        )
        churn_prob = float(churn_model.predict_proba(input_df[CHURN_FEATURE_COLUMNS])[0, 1])
        risk = churn_risk_label(churn_prob)

        st.markdown("##### Prediction")
        c1, c2 = st.columns(2, gap="medium")
        with c1:
            st.metric("Churn probability", f"{churn_prob:.0%}")
        with c2:
            st.metric("Risk level", risk)

        xgb_model = churn_model.named_steps["model"]
        importance = pd.DataFrame(
            {
                "feature": CHURN_FEATURE_COLUMNS,
                "importance": xgb_model.feature_importances_,
            }
        ).sort_values("importance", ascending=True)

        imp_fig = px.bar(
            importance,
            x="importance",
            y="feature",
            orientation="h",
            title="Top Churn Drivers",
            labels={"importance": "Importance", "feature": "Feature"},
            template="plotly_white",
        )
        style_plotly_fig(imp_fig)
        st.plotly_chart(imp_fig, use_container_width=True)


def page_cohort_analysis() -> None:
    """Retention cohort heatmap and summary metrics by install week."""
    render_page_header(
        "Cohort Analysis",
        "Install-week retention (D1, D7, D30) from synthetic login activity.",
    )
    render_methodology_expander(
        "How are cohort retention rates built?",
        """
        **Cohorts:** `install_week` from `install_date` (weekly period).

        **Retention flags** (days between install and last login):
        - **D1:** active at least once after install day (span ≥ 1)
        - **D7:** span ≥ 7 days
        - **D30:** span ≥ 30 days

        The heatmap shows each cohort’s % retained; the curve averages D1/D7/D30 across weeks.
        """,
    )

    data_mtime = _acquisition_data_mtime()
    retention_matrix, averages, curve_df = compute_cohort_outputs(data_mtime)

    st.markdown("##### Cohort summary")
    s1, s2, s3 = st.columns(3, gap="medium")
    with s1:
        st.metric("Average D1 Retention", f"{averages['d1']:.1f}%")
    with s2:
        st.metric("Average D7 Retention", f"{averages['d7']:.1f}%")
    with s3:
        st.metric("Average D30 Retention", f"{averages['d30']:.1f}%")

    heatmap_fig = px.imshow(
        retention_matrix,
        text_auto=".1f",
        aspect="auto",
        color_continuous_scale="Blues",
        labels=dict(x="Retention day", y="Install week", color="Retention %"),
        title="Retention Cohort Heatmap",
    )
    style_plotly_fig(heatmap_fig)
    st.plotly_chart(heatmap_fig, use_container_width=True)

    curve_fig = px.line(
        curve_df,
        x="day_label",
        y="retention_pct",
        title="Retention Curve",
        labels={"day_label": "Day", "retention_pct": "Retention %"},
        markers=True,
        template="plotly_white",
    )
    curve_fig.update_traces(line=dict(width=2.5), marker=dict(size=8))
    style_plotly_fig(curve_fig)
    st.plotly_chart(curve_fig, use_container_width=True)

    st.markdown("##### Retention matrix (%)")
    display_matrix = retention_matrix.reset_index().rename(columns={"install_week": "Install Week"})
    st.dataframe(display_matrix, use_container_width=True, hide_index=True)


@st.fragment
def _optimizer_panel(
    channels: list[str],
    predicted_roas: dict[str, float],
    roas_std: dict[str, float],
    saturation_scales: dict[str, float],
    default_max_share: float,
    mc_simulations: int,
    roas_cv: float,
    random_seed: int,
) -> None:
    """Reactive budget optimizer panel — reruns on widget changes."""
    st.markdown("##### Budget & constraints")
    st.caption(
        "Uses **diminishing returns** per channel (log response curve) and a max share cap — "
        "not naive linear ROAS, so one channel cannot absorb the entire budget."
    )

    total_budget = st.slider(
        "Total budget ($)",
        min_value=1_000,
        max_value=1_000_000,
        value=100_000,
        step=1_000,
    )
    max_channel_share = st.slider(
        "Max share per channel (%)",
        min_value=20,
        max_value=60,
        value=int(default_max_share * 100),
        step=5,
        help="Growth teams rarely put more than 40–50% of UA budget in one channel.",
    ) / 100.0

    st.caption("Minimum spend per channel")
    min_cols = st.columns(len(channels), gap="small")
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

    roas_df = pd.DataFrame(
        {"Channel": channels, "Predicted ROAS": [predicted_roas[ch] for ch in channels]}
    )
    st.dataframe(
        roas_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Predicted ROAS": st.column_config.NumberColumn(format="%.2f"),
        },
    )

    try:
        from src.budget_optimizer import optimize_budget_with_uncertainty
    except ImportError as exc:
        st.error(
            "Budget optimizer failed to load (SciPy may be missing on the server). "
            f"Details: {exc}"
        )
        return

    try:
        result = optimize_budget_with_uncertainty(
            total_budget,
            min_spends,
            predicted_roas,
            saturation_scales=saturation_scales,
            roas_std=roas_std,
            max_channel_share=max_channel_share,
            n_simulations=mc_simulations,
            seed=random_seed,
            default_roas_cv=roas_cv,
        )
    except ValueError as exc:
        st.error(str(exc))
        return

    uncertainty = result["uncertainty"]
    total_band = uncertainty["total"]

    st.markdown(
        f"""
        <div class="result-card">
            <span class="kpi-label">Total expected return (P50)</span>
            <span class="kpi-value">{format_usd_k(total_band['p50'])}</span>
            <span class="kpi-label" style="margin-top:0.5rem;display:block;">
                90% band: {format_usd_k(total_band['p10'])} – {format_usd_k(total_band['p90'])}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    allocation_df = pd.DataFrame(
        {
            "Channel": channels,
            "Allocation": [format_usd_k(result["allocations"][ch]) for ch in channels],
            "Share (%)": [round(result["percentages"][ch], 1) for ch in channels],
            "Return P50": [
                format_usd_k(uncertainty["channel_bands"][ch]["p50"]) for ch in channels
            ],
            "Return P10–P90": [
                f"{format_usd_k(uncertainty['channel_bands'][ch]['p10'])} – "
                f"{format_usd_k(uncertainty['channel_bands'][ch]['p90'])}"
                for ch in channels
            ],
            "Marginal ROAS": [round(result["marginal_roas"][ch], 2) for ch in channels],
        }
    )
    st.dataframe(allocation_df, use_container_width=True, hide_index=True)

    st.markdown("##### Return uncertainty (Monte Carlo)")
    mc_df = pd.DataFrame({"Total Return ($)": uncertainty["simulated_totals"]})
    hist_fig = px.histogram(
        mc_df,
        x="Total Return ($)",
        nbins=40,
        title=f"Simulated total return ({mc_simulations:,} runs, ROAS noise)",
        template="plotly_white",
    )
    hist_fig.add_vline(
        x=total_band["p10"],
        line_dash="dash",
        line_color="#64748b",
        annotation_text="P10",
    )
    hist_fig.add_vline(
        x=total_band["p50"],
        line_dash="solid",
        line_color="#2563eb",
        annotation_text="P50",
    )
    hist_fig.add_vline(
        x=total_band["p90"],
        line_dash="dash",
        line_color="#64748b",
        annotation_text="P90",
    )
    style_plotly_fig(hist_fig)
    st.plotly_chart(hist_fig, use_container_width=True)

    chart_values = pd.DataFrame(
        {
            "Channel": channels,
            "Allocation ($)": [result["allocations"][ch] for ch in channels],
            "Expected Return ($)": [result["expected_returns"][ch] for ch in channels],
        }
    )

    chart_col1, chart_col2 = st.columns(2, gap="large")

    pie_fig = px.pie(
        chart_values,
        names="Channel",
        values="Allocation ($)",
        title="Budget allocation",
        color="Channel",
        color_discrete_map=channel_color_map(channels),
        hole=0.45,
    )
    pie_fig.update_traces(textposition="inside", textinfo="percent+label")
    style_plotly_fig(pie_fig)

    p50_returns = [uncertainty["channel_bands"][ch]["p50"] for ch in channels]
    error_plus = [
        uncertainty["channel_bands"][ch]["p90"] - uncertainty["channel_bands"][ch]["p50"]
        for ch in channels
    ]
    error_minus = [
        uncertainty["channel_bands"][ch]["p50"] - uncertainty["channel_bands"][ch]["p10"]
        for ch in channels
    ]
    bar_fig = go.Figure(
        data=[
            go.Bar(
                x=channels,
                y=p50_returns,
                marker_color=[CHANNEL_COLORS.get(ch, "#64748B") for ch in channels],
                error_y={
                    "type": "data",
                    "symmetric": False,
                    "array": error_plus,
                    "arrayminus": error_minus,
                },
            )
        ]
    )
    bar_fig.update_layout(
        title="Expected return by channel (P50, P10–P90 band)",
        xaxis_title="Channel",
        yaxis_title="Return ($)",
    )
    style_plotly_fig(bar_fig)

    with chart_col1:
        st.plotly_chart(pie_fig, use_container_width=True)
    with chart_col2:
        st.plotly_chart(bar_fig, use_container_width=True)


def page_optimizer() -> None:
    """Budget allocation optimizer across ad channels."""
    render_page_header(
        "Budget Optimizer",
        "Allocate UA budget across channels with diminishing returns and uncertainty bands.",
    )
    render_methodology_expander(
        "How does budget optimization work?",
        """
        **Objective:** Maximize expected IAP return subject to total budget, per-channel
        minimums, and a **max share per channel** cap (default 40%).

        **Response curve:** `return = base_roas × saturation × log(1 + spend/saturation)` so
        marginal ROAS falls as spend rises (audience saturation).

        **Uncertainty:** Monte Carlo simulates ROAS noise (historical std or 15% CV) for
        P10/P50/P90 total and per-channel returns.

        **Solver:** SciPy SLSQP (nonlinear). TikTok has lower saturation scale in config.
        """,
    )

    config = load_config()
    channels = config["channels"]
    data_mtime = _acquisition_data_mtime()
    predicted_roas, roas_std = compute_channel_roas_stats(data_mtime)
    opt_cfg = config.get("optimizer", {})
    default_sat = float(opt_cfg.get("default_saturation", 25000))
    saturation_scales = {
        ch: float(opt_cfg.get("channel_saturation", {}).get(ch, default_sat))
        for ch in channels
    }
    max_share = float(opt_cfg.get("max_channel_share", 0.40))
    mc_sims = int(opt_cfg.get("monte_carlo_simulations", 2000))
    roas_cv = float(opt_cfg.get("roas_uncertainty_cv", 0.15))
    seed = int(config["data"]["random_seed"])

    with st.container(border=True):
        _optimizer_panel(
            channels,
            predicted_roas,
            roas_std,
            saturation_scales,
            max_share,
            mc_sims,
            roas_cv,
            seed,
        )


def page_methodology() -> None:
    """Project documentation, model stack, and synthetic data notes."""
    render_page_header(
        "Metodologia & Info",
        "Architecture, models, and how synthetic Clash Royale UA data is built.",
    )

    with st.container(border=True):
        st.markdown('<p class="info-section-title">Product scope</p>', unsafe_allow_html=True)
        st.markdown(
            """
            **UA Optimizer** is a portfolio-grade analytics app for mobile game UA teams.
            It replaces a scattered CSV → Jupyter → Tableau workflow with one product:
            campaign KPIs, ML predictions, churn risk, cohort retention, and budget allocation.
            """
        )
        st.caption("Fan project · Not affiliated with Supercell")

    with st.container(border=True):
        st.markdown('<p class="info-section-title">Data pipeline</p>', unsafe_allow_html=True)
        pipeline_steps = [
            ("`src/data_generation.py`", "100k synthetic players with CR-style progression"),
            ("`src/train_*_model.py`", "Retention, LTV, churn, and BG/NBD training"),
            ("`app/streamlit_app.py`", "Interactive dashboard (this UI)"),
            ("`config/config.yaml`", "Paths, channels, and optimizer saturation scales"),
        ]
        for index, (path, description) in enumerate(pipeline_steps, start=1):
            st.markdown(
                f"""
                <div class="pipeline-step">
                    <span class="pipeline-num">{index}</span>
                    <span><strong>{path}</strong> — {description}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    col1, col2 = st.columns(2, gap="medium")
    with col1:
        with st.container(border=True):
            st.markdown('<p class="info-section-title">ML models</p>', unsafe_allow_html=True)
            st.dataframe(
                pd.DataFrame(
                    {
                        "Model": ["Retention", "LTV", "Churn", "BG/NBD"],
                        "Algorithm": [
                            "XGBoost classifier",
                            "XGBoost regressor",
                            "XGBoost classifier",
                            "lifetimes",
                        ],
                        "Target": [
                            "Day-7 retention",
                            "90-day IAP",
                            "14-day idle churn",
                            "Gem-buyer LTV",
                        ],
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )
    with col2:
        with st.container(border=True):
            st.markdown('<p class="info-section-title">Optimization</p>', unsafe_allow_html=True)
            st.markdown(
                """
                **Budget allocation** uses log diminishing returns per channel and
                Monte Carlo ROAS noise (2,000 simulations by default).

                **SciPy SLSQP** enforces total budget, per-channel minimums, and a max
                channel share cap (typically 40%).
                """
            )

    with st.container(border=True):
        st.markdown('<p class="info-section-title">Tech stack</p>', unsafe_allow_html=True)
        stack_cols = st.columns(4, gap="small")
        for col, label in zip(
            stack_cols,
            ["Streamlit", "Pandas · NumPy", "XGBoost · sklearn", "SciPy · Plotly"],
        ):
            with col:
                st.markdown(
                    f'<div style="text-align:center;padding:0.5rem;background:#f8fafc;'
                    f'border-radius:8px;font-size:0.85rem;font-weight:600;color:#475569;">'
                    f"{label}</div>",
                    unsafe_allow_html=True,
                )
        st.caption("BG/NBD training uses `lifetimes` from requirements-dev.txt")


def main() -> None:
    """Run the multipage Streamlit application."""
    try:
        _run_app()
    except Exception as exc:
        st.error("The app failed to start. Details below:")
        st.exception(exc)
        logger.exception("Streamlit app startup failed")


def _run_app() -> None:
    """Render the multipage dashboard."""
    config = load_config()
    game_title = config.get("game", {}).get("title", APP_TITLE)

    st.set_page_config(
        page_title=game_title,
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    if not st.session_state.get("_css_injected"):
        inject_custom_css()
        st.session_state["_css_injected"] = True

    if not st.session_state.get("_data_warmed"):
        warm_mtime = _acquisition_data_mtime()
        load_acquisition_data(warm_mtime)
        st.session_state["_data_warmed"] = True

    render_static_header()
    render_app_router()


main()
