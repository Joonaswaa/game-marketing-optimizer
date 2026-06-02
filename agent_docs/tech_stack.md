# Tech Stack & Tools — Game Marketing Campaign Optimizer

## Full Library List

| Library | Version | Purpose | Install |
|---------|---------|---------|---------|
| Python | 3.12+ | Language | System |
| Streamlit | 1.55.0 | Web app framework | `pip install streamlit==1.55.0` |
| Pandas | >=2.0.0 | DataFrames | `pip install pandas` |
| NumPy | >=1.26.0 | Numerical ops, correlation generation | `pip install numpy` |
| Faker | >=24.0.0 | Realistic user IDs, dates | `pip install faker` |
| Scikit-learn | >=1.4.0 | Preprocessing pipeline, metrics | `pip install scikit-learn` |
| XGBoost | >=2.0.0 | Retention classifier + LTV regressor | `pip install xgboost` |
| lifetimes | >=0.11.3 | BG/NBD + Gamma-Gamma LTV | `pip install lifetimes` |
| SciPy | >=1.12.0 | Budget optimization (linprog) | `pip install scipy` |
| Plotly | >=5.20.0 | All charts and visualizations | `pip install plotly` |
| joblib | >=1.3.0 | Save/load ML models | `pip install joblib` |
| PyYAML | >=6.0.0 | Load config/config.yaml | `pip install pyyaml` |

---

## ⚠️ Critical Technical Flags — Read Before Coding

### Flag 1: SciPy Optimizer Method
```python
# ✅ CORRECT — always use this
result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

# ❌ WRONG — deprecated, unreliable in SciPy >= 1.9
result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='simplex')
```

### Flag 2: Streamlit Caching — Mandatory on Every Loader
```python
# ✅ For DataFrames and computed values
@st.cache_data
def load_acquisition_data() -> pd.DataFrame:
    return pd.read_csv("data/acquisition_data.csv")

# ✅ For ML models (loaded once, shared across all sessions)
@st.cache_resource
def load_retention_model():
    return joblib.load("models/retention_model.joblib")

# ❌ Never load a model or CSV without caching
# Without @st.cache_data, Streamlit re-runs every function on every click
```

### Flag 3: Visualization — Plotly Only
```python
# ✅ CORRECT
import plotly.express as px
fig = px.bar(df, x="channel", y="cpa", title="CPA by Channel")
st.plotly_chart(fig, use_container_width=True)

# ❌ NEVER use this
import matplotlib.pyplot as plt
plt.bar(...)
st.pyplot(fig)
```

### Flag 4: Logging — No print() Statements
```python
# ✅ CORRECT
import logging
logger = logging.getLogger(__name__)
logger.info("Training retention model...")
logger.warning("Low sample count for channel: %s", channel)

# ❌ NEVER use print in production code
print("Training model...")
```

---

## Config Loading Pattern

Always load paths and hyperparameters from `config/config.yaml`:

```python
# src/utils.py
import yaml
from pathlib import Path

def load_config() -> dict:
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)

# Usage in any module
config = load_config()
SEED = config["data"]["random_seed"]          # 42
N_USERS = config["data"]["n_users"]           # 100000
ACQUISITION_PATH = config["data"]["acquisition_path"]
```

---

## ML Pipeline Pattern

Use `sklearn.pipeline.Pipeline` + `ColumnTransformer` for all models:

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from xgboost import XGBClassifier
import joblib

CATEGORICAL_FEATURES = ["acquisition_channel", "country", "device_type", "age_group"]
NUMERIC_FEATURES = ["cost_per_install", "sessions_week1", "playtime_week1",
                    "levels_completed", "social_interactions"]

preprocessor = ColumnTransformer([
    ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
     CATEGORICAL_FEATURES),
    ("num", StandardScaler(), NUMERIC_FEATURES)
])

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=SEED,
        eval_metric="logloss"
    ))
])

pipeline.fit(X_train, y_train)
joblib.dump(pipeline, "models/retention_model.joblib")
```

---

## BG/NBD Pattern (lifetimes library)

```python
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import summary_data_from_transaction_data

# Step 1: Create RFM summary from transaction data
summary = summary_data_from_transaction_data(
    transactions_df,
    customer_id_col="user_id",
    datetime_col="transaction_date",
    monetary_value_col="transaction_amount",
    observation_period_end="2024-12-31"
)

# Step 2: Fit BG/NBD (predicts future purchase count)
bgf = BetaGeoFitter(penalizer_coef=0.01)
bgf.fit(summary["frequency"], summary["recency"], summary["T"])

# Step 3: Fit Gamma-Gamma (predicts revenue per purchase)
# IMPORTANT: Only fit on customers with frequency > 0
repeat_buyers = summary[summary["frequency"] > 0]
ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(repeat_buyers["frequency"], repeat_buyers["monetary_value"])

# Step 4: Predict 90-day LTV
ltv = ggf.customer_lifetime_value(
    bgf,
    summary["frequency"],
    summary["recency"],
    summary["T"],
    summary["monetary_value"],
    time=3,          # 3 months ≈ 90 days
    freq="D",
    discount_rate=0.01
)

# Save both models together
joblib.dump((bgf, ggf), "models/bgnbd_model.joblib")
```

---

## Budget Optimizer Pattern (SciPy)

```python
import numpy as np
from scipy.optimize import linprog

def optimize_budget(total_budget: float,
                    min_spends: dict[str, float],
                    predicted_roas: dict[str, float]) -> dict:
    channels = list(predicted_roas.keys())
    n = len(channels)

    # SciPy minimizes — negate ROAS to maximize
    c = [-predicted_roas[ch] for ch in channels]

    # Constraint: total spend <= budget
    A_ub = [np.ones(n)]
    b_ub = [total_budget]

    # Per-channel bounds: [min_spend, total_budget]
    bounds = [(min_spends.get(ch, 0), total_budget) for ch in channels]

    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

    if not result.success:
        raise ValueError(f"Optimization failed: {result.message}")

    allocations = dict(zip(channels, result.x))
    return {
        "allocations": allocations,
        "percentages": {ch: v / total_budget * 100 for ch, v in allocations.items()},
        "expected_returns": {ch: predicted_roas[ch] * allocations[ch] for ch in channels},
        "total_expected_return": sum(predicted_roas[ch] * allocations[ch] for ch in channels)
    }
```

---

## Project Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Generate datasets (run once)
python src/data_generation.py

# Train all models (run once after data generation)
python src/train_retention_model.py
python src/train_ltv_model.py
python src/train_bgnbd_model.py

# Run the Streamlit app
streamlit run app/streamlit_app.py

# Deploy (after pushing to GitHub)
# Go to share.streamlit.io → Connect repo → Select app/streamlit_app.py
```
