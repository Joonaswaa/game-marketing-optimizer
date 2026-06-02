# Project Brief — Persistent Rules & Conventions

**Last Updated:** June 2, 2026  
**Update this file** whenever a new convention is agreed, a decision is changed, or a new tool is added.

---

## Product Vision (One Line)
A free Streamlit ML dashboard that replaces the CSV-export-Jupyter-Tableau workflow for mobile game marketing analysts — campaign data, retention/LTV predictions, and budget optimization in one place.

---

## Hard Rules (Never Break These)

| Rule | Why |
|------|-----|
| `method='highs'` in all SciPy calls | `method='simplex'` deprecated in SciPy >= 1.9 |
| `@st.cache_data` on all CSV loaders | App reloads dataset on every click without it |
| `@st.cache_resource` on all model loaders | Models are large — reload is slow and unnecessary |
| Plotly only — never Matplotlib | Analysts need interactive tooltips |
| `logging` module only — never `print()` | print() is for scripts, logging is for production |
| Load paths from `config/config.yaml` | Hardcoded paths break on other machines |
| Type hints on every function | Required for maintainability |
| Docstrings on every public function | Required for portfolio quality |
| `models/*.joblib` in `.gitignore` | Binary files shouldn't be in git |
| `data/*.csv` in `.gitignore` | Regenerate with `data_generation.py` |

---

## Architecture Decisions (Agreed)

| Decision | Choice | Reason | Date |
|----------|--------|--------|------|
| App framework | Streamlit | Fastest ML web app in Python | June 2026 |
| ML — classification | XGBoost | Best accuracy/speed on tabular data | June 2026 |
| ML — regression | XGBoost Regressor | Same library, consistent | June 2026 |
| LTV probabilistic | BG/NBD + Gamma-Gamma (lifetimes) | Gold standard for transactional LTV | June 2026 |
| Optimization | SciPy linprog | Free, simple, sufficient for 5 channels | June 2026 |
| Visualization | Plotly | Interactive, Streamlit-native | June 2026 |
| Deployment | Streamlit Community Cloud | Free, GitHub-connected, instant | June 2026 |
| Config | PyYAML | Avoid hardcoded paths/seeds | June 2026 |

---

## Scope Boundaries

**In scope (build this):**
- Synthetic data pipeline
- XGBoost retention classifier
- XGBoost LTV regressor
- BG/NBD + Gamma-Gamma LTV model
- 3-page Streamlit app
- SciPy budget optimizer
- Streamlit Community Cloud deployment

**Out of scope (do not build):**
- Live Facebook/Google Ads API connections
- User authentication
- Mobile app
- Markov Chain attribution
- CSV export button
- Dark mode
- Database (SQLite, Postgres, etc.)
- Docker/containerization

---

## Key Commands

```bash
# Setup
pip install -r requirements.txt

# Generate data (once)
python src/data_generation.py

# Train models (once, after data generation)
python src/train_retention_model.py
python src/train_ltv_model.py
python src/train_bgnbd_model.py

# Run app locally
streamlit run app/streamlit_app.py

# Check for anti-patterns before committing
grep -r "matplotlib" src/ app/          # Should return nothing
grep -r "simplex" src/                  # Should return nothing
grep -r "^    print(" src/ app/         # Should return nothing

# Git workflow
git add .
git commit -m "Day X: [brief description]"
git push origin main
# Streamlit Community Cloud auto-deploys on push
```

---

## Folder Structure (Do Not Change)

```
game-marketing-optimizer/
├── AGENTS.md                        ← AI master plan (this session)
├── CLAUDE.md                        ← Claude Code config
├── .cursorrules                     ← Cursor config
├── .gitignore
├── requirements.txt
├── README.md
│
├── config/
│   └── config.yaml                  ← All paths, seeds, hyperparams
│
├── data/                            ← Generated CSVs (gitignored)
│   ├── acquisition_data.csv
│   └── transaction_data.csv
│
├── models/                          ← Saved models (gitignored)
│   ├── retention_model.joblib
│   ├── ltv_xgboost_model.joblib
│   └── bgnbd_model.joblib
│
├── src/                             ← Python modules
│   ├── data_generation.py
│   ├── preprocessing.py
│   ├── train_retention_model.py
│   ├── train_ltv_model.py
│   ├── train_bgnbd_model.py
│   ├── budget_optimizer.py
│   └── utils.py
│
├── app/
│   └── streamlit_app.py             ← Main entry point
│
├── notebooks/
│   └── exploration.ipynb
│
└── agent_docs/                      ← AI context docs
    ├── tech_stack.md
    ├── code_patterns.md
    ├── product_requirements.md
    ├── testing.md
    └── project_brief.md             ← This file
```

---

## Update Cadence

Update this file when:
- A new library is added to `requirements.txt`
- An architectural decision is reversed
- A new hard rule is agreed
- The project scope changes
- A new convention is established during a coding session
