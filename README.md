# Game Marketing Campaign Optimizer

An ML-powered **Streamlit dashboard** for mobile game marketing analysts. It replaces the typical workflow of exporting CSVs into Jupyter and Tableau by putting **campaign analytics**, **player retention/LTV predictions**, and **ad budget optimization** in one free web app.

Built as a portfolio project demonstrating end-to-end data science: synthetic data → trained models → interactive product.

---

## Business value

| Problem | What this app does |
|--------|---------------------|
| Channel performance is scattered across exports | **Campaign Overview** — KPIs, CPA by channel, ROAS trends |
| Hard to score new players before spend ramps | **Player Predictions** — Day-7 retention probability and 90-day LTV (XGBoost + BG/NBD comparison) |
| Budget splits are gut-feel | **Budget Optimizer** — SciPy linear programming allocates spend to maximize expected return |

**Validated model quality (test set):** retention ROC-AUC **0.80**, LTV R² **0.87**.

---

## Features

1. **Campaign Overview** — total users, average LTV, Day-7 retention rate, total ad spend; interactive Plotly bar and line charts.
2. **Player Predictions** — form inputs (channel, geo, device, week-1 behavior); XGBoost retention + LTV; optional BG/NBD LTV for purchasers.
3. **Budget Optimizer** — budget slider, per-channel minimums, optimal allocation pie chart and expected-return bar chart.

Data is **synthetic but realistic** (100k users, correlated sessions → retention → LTV, channel quality tiers). No live ad APIs or authentication — focused on ML and UX for analysts.

---

## Tech stack

| Layer | Tools |
|-------|--------|
| App | [Streamlit](https://streamlit.io) 1.55 |
| ML | XGBoost (retention + LTV), lifetimes BG/NBD + Gamma-Gamma |
| Optimization | SciPy `linprog` (`method='highs'`) |
| Data | Pandas, NumPy, Faker |
| Viz | Plotly |
| Config | PyYAML (`config/config.yaml`) |

Python **3.11+** recommended (3.12 supported).

---

## Quick start (local)

### 1. Clone and install

```powershell
cd game-marketing-optimizer
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
```

### 2. Generate data and train models (required once)

Either run the all-in-one bootstrap:

```powershell
py scripts/bootstrap_artifacts.py
```

Or step by step:

```powershell
py src/data_generation.py
py src/train_retention_model.py
py src/train_ltv_model.py
py src/train_bgnbd_model.py
```

This creates `data/*.csv` and `models/*.joblib` (both are **gitignored** by design).

### 3. Run the app

```powershell
py -m streamlit run app/streamlit_app.py
```

Open the URL shown in the terminal (usually http://localhost:8501).

---

## Project layout

```
app/streamlit_app.py      # Main entry point (use this for Streamlit Cloud)
config/config.yaml        # Paths, seeds, hyperparameters
src/                      # Data generation, training, budget optimizer
data/                     # Generated CSVs (not in git)
models/                   # Trained .joblib files (not in git)
scripts/bootstrap_artifacts.py
```

---

## Deploy to Streamlit Community Cloud

Streamlit Cloud deploys from a **public GitHub repo**. This project does **not** commit `data/*.csv` or `models/*.joblib` (large binaries, reproducible locally). You must supply artifacts on the server using one of the options below.

### Prerequisites (your machine)

1. **Create a GitHub repository** (public) if you do not have one yet.
2. **Initialize git** in this folder (if not already):

   ```powershell
   git init
   git add .
   git commit -m "Initial commit: Game Marketing Campaign Optimizer"
   ```

3. **Do not** commit secrets or local virtualenvs (`.venv/` is gitignored).

### Option A — Recommended for demo deploy: include artifacts once

After running `py scripts/bootstrap_artifacts.py` locally, force-add artifacts **only for deployment**:

```powershell
git add -f data/acquisition_data.csv data/transaction_data.csv
git add -f models/retention_model.joblib models/ltv_xgboost_model.joblib models/bgnbd_model.joblib
git commit -m "Add generated data and models for Streamlit Cloud"
```

> Use a dedicated deploy branch or repo if you prefer to keep `main` free of binaries.

### Option B — Regenerate on every cold start (advanced)

Host bootstrap logic or download artifacts from cloud storage at app startup. Not included in the MVP; see Streamlit docs on [secrets](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management) and external files.

### Push to GitHub

```powershell
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/game-marketing-optimizer.git
git push -u origin main
```

### Connect Streamlit Community Cloud

1. Go to **[share.streamlit.io](https://share.streamlit.io)** and sign in with GitHub.
2. Click **Create app** → select your repository and branch (`main`).
3. Set **Main file path** to: `app/streamlit_app.py`
4. Leave **App URL** as default or customize.
5. Click **Deploy**. Streamlit installs `requirements.txt` from the repo root automatically.
6. First load may take 1–2 minutes (cold start + cached data/models).

No `packages.txt` is required (no system-level apt dependencies). Optional UI settings live in `.streamlit/config.toml`.

### After deploy

- Each `git push` to the connected branch redeploys the app.
- Confirm all three sidebar pages load and **Player Predictions** returns scores (needs models on the server).

---

## Development notes

- Paths and seeds: `config/config.yaml`
- Master plan / phases: `AGENTS.md`
- Anti-patterns: no Matplotlib, no `method='simplex'`, use `logging` not `print()`

---

## License

MIT (or your choice) — portfolio / educational use.
