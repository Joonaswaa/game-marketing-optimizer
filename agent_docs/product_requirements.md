# Product Requirements — Game Marketing Campaign Optimizer

## What We're Building
A free, open-source ML-powered Streamlit web app that gives mobile game marketing analysts:
1. Campaign performance visibility across 5 ad channels
2. Player Day-7 retention probability + 90-day LTV predictions (XGBoost + BG/NBD)
3. Data-driven ad budget allocation using linear programming

**Target User:** Marketing analyst at a mobile game company — intermediate tech level, not a data scientist  
**Timeline:** 7 days  
**Budget:** $0 — entirely free/open-source  

---

## Primary User Story
"As a marketing analyst, I want to see all my campaign performance data in one dashboard so that I can quickly identify which channels are delivering the best ROI without exporting multiple CSVs."

---

## Must-Have Features (P0 — All required for MVP)

### 1. Synthetic Data Pipeline
- Generate acquisition dataset: 100,000 users with campaign + behavioral features
- Generate transaction dataset: purchase history for ~20% of users
- Realistic correlations required (higher sessions → retention → LTV, YouTube > TikTok quality)
- Reproducible via random seed from config

### 2. Campaign Overview Dashboard (Page 1)
- 4 KPI cards: Total Users, Avg LTV, Day-7 Retention Rate, Total Ad Spend
- Bar chart: Average CPA by channel (Plotly)
- Line chart: ROAS trend over time (Plotly)
- All data loaded with `@st.cache_data`

### 3. Retention Classifier (XGBoost)
- Binary classification: will user be retained on Day 7?
- Features: channel, country, device, sessions_week1, playtime_week1, levels_completed
- Target ROC-AUC > 0.75
- Saved with joblib

### 4. LTV Regressor (XGBoost)
- Regression: predict 90-day player LTV in dollars
- Same features as retention model
- Target R² > 0.70
- Saved with joblib

### 5. LTV Model — BG/NBD + Gamma-Gamma (lifetimes)
- Fits on transaction history (repeat buyers only)
- Predicts expected purchases + revenue for next 90 days
- Shown alongside XGBoost prediction on Page 2 for comparison
- Only applies to users with purchase history

### 6. Player Predictions Interface (Page 2)
- Input form: channel, country, device, sessions, playtime, levels
- Output: Day-7 retention probability (XGBoost)
- Output: 90-day LTV estimate (XGBoost)
- Output: BG/NBD LTV comparison (for purchasers)
- Response within 1 second

### 7. Budget Optimizer (Page 3)
- Slider: total budget ($1,000 – $1,000,000)
- Per-channel minimum spend inputs
- SciPy linprog with `method='highs'`
- Output: % allocation per channel
- Output: expected return per channel
- Plotly pie chart (allocation) + bar chart (expected returns)

---

## NOT in MVP (Do Not Build These)
- Live API connections to Facebook/Google Ads
- User authentication or login
- Mobile app version
- Markov Chain attribution modeling
- Downloadable CSV export
- Dark mode UI toggle
- GDPR/compliance features

---

## Acceptance Criteria Summary

| Feature | Pass Condition |
|---------|---------------|
| Data generation | Both CSVs created, correlations verified visually |
| Retention model | ROC-AUC > 0.75 on test set |
| LTV model | R² > 0.70 on test set |
| BG/NBD model | Fits without error, produces per-user LTV values |
| Page 1 | Loads in < 3s, all 4 KPIs visible, 2 interactive charts |
| Page 2 | Form submits, all 3 model outputs displayed |
| Page 3 | Optimizer runs, pie + bar charts display, total return shown |
| Deployment | App live at public Streamlit Community Cloud URL |

---

## Channel & Country Reference

```python
CHANNELS = ["TikTok", "YouTube", "Instagram", "Facebook", "Google"]

COUNTRIES = ["US", "UK", "DE", "JP", "BR", "FR", "KR"]

# Channel quality for data generation (YouTube = highest LTV users)
CHANNEL_QUALITY = {
    "YouTube":   {"avg_ltv_multiplier": 2.5, "retention_boost": 0.15},
    "Google":    {"avg_ltv_multiplier": 2.0, "retention_boost": 0.10},
    "Facebook":  {"avg_ltv_multiplier": 1.5, "retention_boost": 0.05},
    "Instagram": {"avg_ltv_multiplier": 1.2, "retention_boost": 0.02},
    "TikTok":    {"avg_ltv_multiplier": 1.0, "retention_boost": -0.05},
}
```

---

## config/config.yaml Reference

```yaml
data:
  acquisition_path: "data/acquisition_data.csv"
  transaction_path: "data/transaction_data.csv"
  n_users: 100000
  random_seed: 42

models:
  retention_path: "models/retention_model.joblib"
  ltv_xgboost_path: "models/ltv_xgboost_model.joblib"
  bgnbd_path: "models/bgnbd_model.joblib"

training:
  test_size: 0.2
  n_estimators: 200
  max_depth: 6
  learning_rate: 0.1

channels: [TikTok, YouTube, Instagram, Facebook, Google]
countries: [US, UK, DE, JP, BR, FR, KR]
```
