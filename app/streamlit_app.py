"""Game Marketing Campaign Optimizer — Streamlit dashboard."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from lifetimes.utils import summary_data_from_transaction_data

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.budget_optimizer import optimize_budget
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

        .main .block-container {
            padding-top: 1.5rem;
            padding-bottom: 3rem;
            max-width: 1180px;
        }

        .hero-banner {
            background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 40%, #d97706 100%);
            border-radius: 20px;
            padding: 1.75rem 2rem;
            margin-bottom: 1.75rem;
            color: #ffffff;
            box-shadow: 0 20px 40px -12px rgba(30, 58, 138, 0.45);
        }

        .hero-banner h1 {
            font-size: 1.85rem;
            font-weight: 800;
            margin: 0 0 0.35rem 0;
            letter-spacing: -0.02em;
        }

        .hero-banner p {
            margin: 0;
            opacity: 0.92;
            font-size: 1rem;
            font-weight: 400;
        }

        .hero-badge {
            display: inline-block;
            background: rgba(255,255,255,0.18);
            border: 1px solid rgba(255,255,255,0.25);
            border-radius: 999px;
            padding: 0.25rem 0.75rem;
            font-size: 0.75rem;
            font-weight: 600;
            margin-bottom: 0.75rem;
            letter-spacing: 0.04em;
            text-transform: uppercase;
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

        div[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
        }

        div[data-testid="stSidebar"] .stRadio label {
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 12px;
            padding: 0.65rem 0.85rem;
            margin-bottom: 0.45rem;
            transition: all 0.18s ease;
            color: #e2e8f0 !important;
            font-weight: 500;
        }

        div[data-testid="stSidebar"] .stRadio label:hover {
            background: rgba(99, 102, 241, 0.25);
            border-color: rgba(129, 140, 248, 0.5);
        }

        div[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label[data-baseweb="radio"] {
            border: none;
        }

        div[data-testid="stSidebar"] h1, div[data-testid="stSidebar"] h2,
        div[data-testid="stSidebar"] h3, div[data-testid="stSidebar"] p,
        div[data-testid="stSidebar"] span, div[data-testid="stSidebar"] label {
            color: #f1f5f9 !important;
        }

        .sidebar-brand {
            font-size: 1.1rem;
            font-weight: 800;
            color: #ffffff !important;
            margin-bottom: 0.25rem;
            letter-spacing: -0.02em;
        }

        .sidebar-tagline {
            font-size: 0.8rem;
            color: #94a3b8 !important;
            margin-bottom: 1.5rem;
            line-height: 1.4;
        }

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


def render_hero(title: str, subtitle: str, badge: str = "Live dashboard") -> None:
    """Render a gradient page header."""
    st.markdown(
        f"""
        <div class="hero-banner">
            <div class="hero-badge">{badge}</div>
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


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
    render_hero(
        "Campaign Overview",
        "Clash Royale UA performance — installs, 90-day IAP, and channel ROAS (synthetic data).",
        badge="Analytics",
    )

    df = load_acquisition_data()

    render_kpi_cards(
        [
            ("Players Acquired", format_count_k(len(df))),
            ("Avg 90d IAP", format_usd_k(df["ltv_day90"].mean())),
            ("Day-7 Retention", f"{df['retained_day7'].mean():.1%}"),
            ("Total UA Spend", format_usd_k(df["cost_per_install"].sum())),
        ]
    )

    chart_col1, chart_col2 = st.columns(2, gap="large")

    cpa_df = (
        df.groupby("acquisition_channel", as_index=False)["cost_per_install"]
        .mean()
        .sort_values("cost_per_install", ascending=True)
    )
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
    """Player retention and IAP prediction form using Clash Royale week-1 profile."""
    render_hero(
        "Player Predictions",
        "Model uses CR progression signals: trophies, king level, win rate, Pass Royale, and clan status.",
        badge="ML Models",
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


@st.fragment
def _optimizer_panel(
    channels: list[str],
    predicted_roas: dict[str, float],
    saturation_scales: dict[str, float],
    default_max_share: float,
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
        result = optimize_budget(
            total_budget,
            min_spends,
            predicted_roas,
            saturation_scales=saturation_scales,
            max_channel_share=max_channel_share,
        )
    except ValueError as exc:
        st.error(str(exc))
        return

    st.markdown(
        f"""
        <div class="result-card">
            <span class="kpi-label">Total expected return</span>
            <span class="kpi-value">{format_usd_k(result['total_expected_return'])}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    allocation_df = pd.DataFrame(
        {
            "Channel": channels,
            "Allocation": [format_usd_k(result["allocations"][ch]) for ch in channels],
            "Share (%)": [round(result["percentages"][ch], 1) for ch in channels],
            "Expected Return": [format_usd_k(result["expected_returns"][ch]) for ch in channels],
            "Marginal ROAS": [round(result["marginal_roas"][ch], 2) for ch in channels],
        }
    )
    st.dataframe(allocation_df, use_container_width=True, hide_index=True)

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

    bar_fig = px.bar(
        chart_values,
        x="Channel",
        y="Expected Return ($)",
        title="Expected return by channel",
        color="Channel",
        color_discrete_map=channel_color_map(channels),
    )
    bar_fig.update_traces(marker_line_width=0, opacity=0.92)
    style_plotly_fig(bar_fig)

    with chart_col1:
        st.plotly_chart(pie_fig, use_container_width=True)
    with chart_col2:
        st.plotly_chart(bar_fig, use_container_width=True)


def page_optimizer() -> None:
    """Budget allocation optimizer across ad channels."""
    render_hero(
        "Budget Optimizer",
        "Maximize expected IAP return across UA channels for Clash Royale campaigns.",
        badge="Optimization",
    )

    config = load_config()
    channels = config["channels"]
    df = load_acquisition_data()

    predicted_roas = df.groupby("acquisition_channel")["roas_90d"].mean().to_dict()
    opt_cfg = config.get("optimizer", {})
    default_sat = float(opt_cfg.get("default_saturation", 25000))
    saturation_scales = {
        ch: float(opt_cfg.get("channel_saturation", {}).get(ch, default_sat))
        for ch in channels
    }
    max_share = float(opt_cfg.get("max_channel_share", 0.40))

    with st.container(border=True):
        _optimizer_panel(channels, predicted_roas, saturation_scales, max_share)


def main() -> None:
    """Run the multipage Streamlit application."""
    config = load_config()
    game_title = config.get("game", {}).get("title", "Clash Royale UA Optimizer")

    st.set_page_config(
        page_title=game_title,
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    inject_custom_css()

    with st.sidebar:
        st.markdown(f'<p class="sidebar-brand">{game_title}</p>', unsafe_allow_html=True)
        st.markdown(
            '<p class="sidebar-tagline">Synthetic CR player data · retention & IAP ML · UA budget</p>',
            unsafe_allow_html=True,
        )
        page = st.radio(
            "Navigation",
            ["Campaign Overview", "Player Predictions", "Budget Optimizer"],
            label_visibility="collapsed",
        )
        st.divider()
        st.caption("Fan project · Not affiliated with Supercell")

    if page == "Campaign Overview":
        page_overview()
    elif page == "Player Predictions":
        page_predictions()
    else:
        page_optimizer()


if __name__ == "__main__":
    main()
