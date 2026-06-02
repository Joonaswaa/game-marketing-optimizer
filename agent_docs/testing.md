# Testing Strategy — Game Marketing Campaign Optimizer

## Philosophy
This is a 1-week portfolio build. We use **manual verification checkpoints** rather than full unit test coverage. The goal is: nothing ships broken, and every day ends with a working state.

---

## Verification Loop (After Every File)

### After `data_generation.py`
```bash
python src/data_generation.py
```
**Check manually:**
- [ ] `data/acquisition_data.csv` exists with ~100,000 rows
- [ ] `data/transaction_data.csv` exists
- [ ] No NaN values in key columns:
  ```python
  import pandas as pd
  df = pd.read_csv("data/acquisition_data.csv")
  print(df.shape)              # Should be (100000, 15)
  print(df.isnull().sum())     # Should be all 0
  print(df["retained_day7"].value_counts(normalize=True))  # ~40–60% retention
  print(df.groupby("acquisition_channel")["ltv_day90"].mean())  # YouTube > TikTok
  ```
- [ ] Correlations look realistic (YouTube has highest avg LTV, TikTok lowest)

---

### After `train_retention_model.py`
```bash
python src/train_retention_model.py
```
**Check manually:**
- [ ] `models/retention_model.joblib` file exists
- [ ] Console shows ROC-AUC > 0.75
- [ ] Console shows F1 score printed
- Quick sanity check:
  ```python
  import joblib, pandas as pd
  model = joblib.load("models/retention_model.joblib")
  df = pd.read_csv("data/acquisition_data.csv").head(5)
  features = ["acquisition_channel","country","device_type","age_group",
              "cost_per_install","sessions_week1","playtime_week1",
              "levels_completed","social_interactions"]
  print(model.predict_proba(df[features]))  # Should return probabilities, not crash
  ```

---

### After `train_ltv_model.py`
```bash
python src/train_ltv_model.py
```
**Check manually:**
- [ ] `models/ltv_xgboost_model.joblib` file exists
- [ ] Console shows R² > 0.70
- [ ] Console shows RMSE and MAE
- [ ] Predictions are positive dollar amounts (not negative)

---

### After `train_bgnbd_model.py`
```bash
python src/train_bgnbd_model.py
```
**Check manually:**
- [ ] `models/bgnbd_model.joblib` file exists
- [ ] No division-by-zero or convergence errors
- [ ] BG/NBD is only fit on users with `frequency > 0`
- [ ] LTV predictions are positive values

---

### After each Streamlit page (Pages 1, 2, 3)
```bash
streamlit run app/streamlit_app.py
```
**Check manually in browser:**

**Page 1:**
- [ ] All 4 KPI cards display numbers (not errors)
- [ ] CPA bar chart renders and has hover tooltips
- [ ] ROAS line chart renders with multiple channel lines
- [ ] Page loads in < 5 seconds on first load, < 1 second on revisit (caching works)

**Page 2:**
- [ ] Form shows all input fields
- [ ] Clicking "Predict" returns retention probability (a % between 0–100)
- [ ] Clicking "Predict" returns LTV estimate (a positive dollar amount)
- [ ] BG/NBD section shows a table or note about purchase history requirement
- [ ] No red error box appears

**Page 3:**
- [ ] Budget slider moves smoothly
- [ ] Min spend inputs accept numbers
- [ ] "Optimize" button returns pie chart
- [ ] "Optimize" button returns bar chart
- [ ] Total expected return is a positive number
- [ ] Channel percentages sum to ~100%

---

### After `budget_optimizer.py`
Quick unit check:
```python
from src.budget_optimizer import optimize_budget

result = optimize_budget(
    total_budget=100_000,
    min_spends={"TikTok": 1000, "YouTube": 1000, "Instagram": 1000,
                "Facebook": 1000, "Google": 1000},
    predicted_roas={"TikTok": 1.8, "YouTube": 3.2, "Instagram": 2.1,
                    "Facebook": 2.4, "Google": 2.8}
)
print(result["percentages"])         # Should sum to ~100%
print(result["total_expected_return"])  # Should be > total_budget (positive ROI)
assert sum(result["percentages"].values()) > 99  # Sanity check
```

---

## Pre-Commit Checklist (Before Each Git Commit)

Run through this before every `git commit`:

- [ ] `streamlit run app/streamlit_app.py` starts without import errors
- [ ] No `matplotlib` imports anywhere: `grep -r "matplotlib" src/ app/`
- [ ] No `method='simplex'` anywhere: `grep -r "simplex" src/`
- [ ] No bare `print()` in production files: `grep -r "^print(" src/ app/`
- [ ] All model files exist in `models/` (if running locally)
- [ ] `requirements.txt` is up to date with any new packages

---

## End-of-Day Verification (Daily)

Before ending each day's work, confirm:

**Day 1:** `python src/data_generation.py` runs cleanly, both CSVs exist, correlations verified  
**Day 2:** Retention model saved, ROC-AUC > 0.75 printed in console  
**Day 3:** LTV model saved (R² > 0.70), BG/NBD model saved without errors  
**Day 4:** Page 1 loads in browser, KPI cards + 2 charts visible  
**Day 5:** Page 2 form returns predictions without errors  
**Day 6:** Page 3 optimizer returns allocation charts  
**Day 7:** App loads from Streamlit Community Cloud public URL  

---

## Common Errors & Fixes

| Error | Likely Cause | Fix |
|-------|-------------|-----|
| `FileNotFoundError: models/retention_model.joblib` | Model not trained yet | Run `python src/train_retention_model.py` first |
| `KeyError: 'acquisition_channel'` | Column name mismatch | Check exact column names in CSV with `df.columns` |
| `linprog: The problem is infeasible` | Min spends exceed total budget | Check that sum of min_spends < total_budget |
| `lifetimes ConvergenceError` | Too few data points | Ensure repeat_buyers has > 50 rows before fitting |
| `StreamlitAPIException: cache` | Wrong cache decorator | Use `@st.cache_data` for data, `@st.cache_resource` for models |
| App reloads on every click | Missing `@st.cache_data` | Add decorator to all data-loading functions |
