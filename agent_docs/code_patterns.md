# Code Patterns — Game Marketing Campaign Optimizer

## File Responsibility Rules

Each file has ONE job. Never mix concerns:

| File | Responsibility | Must NOT contain |
|------|---------------|-----------------|
| `data_generation.py` | Generate synthetic CSVs | Model training, Streamlit |
| `preprocessing.py` | ColumnTransformer pipeline | Model fitting, data gen |
| `train_retention_model.py` | Train + save XGBoost classifier | Streamlit, data gen |
| `train_ltv_model.py` | Train + save XGBoost regressor | Streamlit, data gen |
| `train_bgnbd_model.py` | Fit + save BG/NBD models | Streamlit, data gen |
| `budget_optimizer.py` | SciPy linprog wrapper | Streamlit, model training |
| `utils.py` | Config loader, logging setup, shared helpers | Business logic |
| `streamlit_app.py` | UI only — load models, display results | Training, data gen |

---

## Type Hints — Required on All Functions

```python
# ✅ CORRECT — every function parameter and return typed
def generate_acquisition_data(n_users: int, seed: int) -> pd.DataFrame:
    ...

def train_retention_model(df: pd.DataFrame, config: dict) -> Pipeline:
    ...

def optimize_budget(
    total_budget: float,
    min_spends: dict[str, float],
    predicted_roas: dict[str, float]
) -> dict[str, float | dict]:
    ...

# ❌ WRONG — no type hints
def train_model(df, config):
    ...
```

---

## Docstrings — Required on All Public Functions and Classes

```python
def optimize_budget(total_budget: float, min_spends: dict, predicted_roas: dict) -> dict:
    """
    Allocate ad budget across channels to maximize expected ROAS.

    Uses scipy.optimize.linprog with method='highs' (NOT simplex).
    Minimizes negative ROAS (equivalent to maximizing ROAS).

    Args:
        total_budget: Total budget to allocate in dollars
        min_spends: Dict of channel -> minimum required spend
        predicted_roas: Dict of channel -> predicted ROAS multiplier from ML models

    Returns:
        Dict containing:
            - allocations: channel -> dollar amount
            - percentages: channel -> % of total budget
            - expected_returns: channel -> expected dollar return
            - total_expected_return: float

    Raises:
        ValueError: If SciPy optimization fails (infeasible constraints)
    """
```

---

## Logging Setup — utils.py

```python
# src/utils.py
import logging
import yaml
from pathlib import Path

def setup_logging(level: str = "INFO") -> None:
    """Configure logging for all modules."""
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s — %(name)s — %(levelname)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

def load_config() -> dict:
    """Load project config from config/config.yaml."""
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)
```

---

## Data Generation Pattern

Correlations must be INTENTIONAL — not random:

```python
import numpy as np
import pandas as pd
from faker import Faker

fake = Faker()
rng = np.random.default_rng(seed=42)   # Use default_rng, not np.random.seed()

# ✅ Correlated generation example
# Step 1: Generate base signals
sessions = rng.integers(1, 31, size=n_users)               # 1–30 sessions

# Step 2: Derive retention from sessions (strong positive correlation)
retention_prob = 0.2 + (sessions / 30) * 0.6              # 0.2 to 0.8 range
retained = rng.random(n_users) < retention_prob            # boolean

# Step 3: Derive LTV from retention (retained users spend more)
base_ltv = rng.exponential(scale=10, size=n_users)
ltv = np.where(retained, base_ltv * 3.5, base_ltv * 0.5)  # retained = 3.5x LTV

# ❌ WRONG — purely random, no business meaning
retained = rng.choice([True, False], size=n_users)
ltv = rng.uniform(0, 100, size=n_users)
```

---

## Streamlit Page Structure Pattern

Every page follows this structure:

```python
# Pattern for each page in streamlit_app.py

import streamlit as st
import plotly.express as px
import pandas as pd
import joblib
from src.utils import load_config

# ── 1. Cached loaders (ALWAYS at module level or top of page function) ──

@st.cache_data
def load_acquisition_data() -> pd.DataFrame:
    config = load_config()
    return pd.read_csv(config["data"]["acquisition_path"])

@st.cache_resource
def load_retention_model():
    config = load_config()
    return joblib.load(config["models"]["retention_path"])

# ── 2. Page function ──

def page_overview():
    st.title("📊 Campaign Overview")

    df = load_acquisition_data()

    # ── 3. KPI cards ──
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Users", f"{len(df):,}")
    col2.metric("Avg LTV", f"${df['ltv_day90'].mean():.2f}")
    col3.metric("Day-7 Retention", f"{df['retained_day7'].mean():.1%}")
    col4.metric("Total Ad Spend", f"${df['cost_per_install'].sum():,.0f}")

    # ── 4. Plotly charts (always use_container_width=True) ──
    cpa = df.groupby("acquisition_channel")["cost_per_install"].mean().reset_index()
    fig = px.bar(cpa, x="acquisition_channel", y="cost_per_install",
                 title="Average CPA by Channel", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)
```

---

## Naming Conventions

```python
# Files: snake_case
data_generation.py
train_retention_model.py

# Functions: snake_case, verb-first
def load_data() -> pd.DataFrame: ...
def train_model() -> Pipeline: ...
def optimize_budget() -> dict: ...

# Constants: UPPER_SNAKE_CASE
CHANNELS = ["TikTok", "YouTube", "Instagram", "Facebook", "Google"]
COUNTRIES = ["US", "UK", "DE", "JP", "BR", "FR", "KR"]
SEED = 42

# Variables: snake_case, descriptive
acquisition_df = load_acquisition_data()
retention_model = load_retention_model()
predicted_roas = compute_channel_roas(acquisition_df, ltv_model)
```

---

## Error Handling Pattern

```python
# ✅ Always handle optimizer failure
from scipy.optimize import linprog

result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
if not result.success:
    raise ValueError(f"Budget optimization failed: {result.message}")

# ✅ Handle missing model file gracefully in Streamlit
@st.cache_resource
def load_retention_model():
    try:
        return joblib.load("models/retention_model.joblib")
    except FileNotFoundError:
        st.error("Model not found. Run: python src/train_retention_model.py")
        st.stop()

# ✅ Validate BG/NBD only runs on users with purchases
repeat_buyers = summary[summary["frequency"] > 0]
if len(repeat_buyers) == 0:
    raise ValueError("No repeat buyers found — cannot fit Gamma-Gamma model")
```

---

## .gitignore Contents

```gitignore
# Models — train locally, don't commit large binary files
models/*.joblib

# Data — regenerate with data_generation.py
data/*.csv

# Python
__pycache__/
*.pyc
.venv/
*.egg-info/

# Notebooks checkpoints
.ipynb_checkpoints/

# OS
.DS_Store
```
