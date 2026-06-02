# AGENTS.md — Master Plan for Game Marketing Campaign Optimizer

## Project Overview
**App:** Game Marketing Campaign Optimizer  
**Goal:** ML-powered Streamlit dashboard for mobile game marketing analysts — campaign overview, player retention/LTV predictions, and ad budget optimization in one free web app  
**Stack:** Python 3.12 · Streamlit · XGBoost · lifetimes (BG/NBD) · SciPy · Plotly · joblib  
**Deployment:** Streamlit Community Cloud (free)  
**Current Phase:** Phase 4 — Deploy  

---

## How I Should Think
1. **Understand Intent First** — Before answering, identify what the user actually needs
2. **Ask If Unsure** — If critical information is missing, ask before proceeding
3. **Plan Before Coding** — Propose a brief plan and wait for approval, then implement
4. **Verify After Changes** — Run the app or test the output after each feature; fix before moving on
5. **Explain Trade-offs** — When recommending something, mention the alternatives and why this is better

---

## Plan → Execute → Verify
1. **Plan:** Outline a brief approach and ask for approval before writing any code
2. **Execute:** Implement one feature/file at a time — never two simultaneously
3. **Verify:** After each file, either run it (`python src/filename.py`) or start the app (`streamlit run app/streamlit_app.py`) to confirm it works before moving on

---

## What NOT To Do
- Do NOT delete or overwrite files without explicit confirmation
- Do NOT add features not listed in the current phase (no auth, no live APIs, no extra charts)
- Do NOT use `matplotlib` — Plotly only, always `st.plotly_chart()`
- Do NOT use `method='simplex'` in any SciPy call — always `method='highs'`
- Do NOT use `print()` for logging — use Python `logging` module
- Do NOT hardcode file paths — load from `config/config.yaml`
- Do NOT skip `@st.cache_data` / `@st.cache_resource` on any data or model loader
- Do NOT commit `models/*.joblib` files to git — add to `.gitignore`
- Do NOT add new pip dependencies without updating `requirements.txt`

---

## Context Files
Load only when needed — do not load all at once:
- `agent_docs/tech_stack.md` — Full library list, versions, install commands, critical flags
- `agent_docs/code_patterns.md` — Code style, patterns, caching rules, anti-patterns
- `agent_docs/product_requirements.md` — Features, user stories, acceptance criteria
- `agent_docs/testing.md` — Verification strategy and manual test checklist
- `agent_docs/project_brief.md` — Persistent project rules and conventions

---

## Current State (Update This After Each Session!)
**Last Updated:** June 2, 2026  
**Working On:** Phase 4 — Deploy (docs & prep complete; user actions below)  
**Recently Completed:** Phase 4 prep — README, pinned requirements, bootstrap script, `.streamlit/config.toml`, local smoke test  
**Blocked By:** None — **User:** `git init`, GitHub repo, push, connect at share.streamlit.io; include artifacts on deploy (see README Option A)  

---

## Roadmap

### Phase 1: Data Layer (Day 1)
- [x] Create project folder structure
- [x] Create `config/config.yaml`
- [x] Create `requirements.txt`
- [x] Implement `src/data_generation.py` — acquisition dataset (100k users)
- [x] Implement `src/data_generation.py` — transaction dataset (purchase history)
- [x] Validate: check correlations are realistic (sessions → retention → LTV)
- [x] Create `.gitignore`

### Phase 2: ML Models (Days 2–3)
- [x] Implement `src/preprocessing.py` — feature pipeline (ColumnTransformer)
- [x] Implement `src/train_retention_model.py` — XGBoost classifier, save to `models/`
- [x] Implement `src/train_ltv_model.py` — XGBoost regressor, save to `models/`
- [x] Implement `src/train_bgnbd_model.py` — BG/NBD + Gamma-Gamma, save to `models/`
- [x] Validate: retention ROC-AUC > 0.75, LTV R² > 0.70

### Phase 3: Streamlit App (Days 4–6)
- [x] Scaffold `app/streamlit_app.py` with multipage navigation
- [x] Build Page 1: Campaign Overview (KPI cards + 2 Plotly charts)
- [x] Build Page 2: Player Predictions (input form + model output)
- [x] Build Page 3: Budget Optimizer (sliders + SciPy + Plotly pie/bar)
- [x] Implement `src/budget_optimizer.py`
- [x] Add `@st.cache_data` / `@st.cache_resource` throughout
- [x] Validate: all pages load, predictions return, optimizer runs

### Phase 4: Deploy (Day 7)
- [x] Finalize `requirements.txt` (pinned versions; streamlit==1.55.0)
- [x] Write `README.md` (portfolio-quality, explains business value)
- [x] Deploy prep: `.gitignore`, `scripts/bootstrap_artifacts.py`, `.streamlit/config.toml`
- [x] Local smoke test: `py -m streamlit run app/streamlit_app.py`
- [ ] **User:** Push to GitHub (public repo) — no git remote initialized yet
- [ ] **User:** Deploy on Streamlit Community Cloud (`app/streamlit_app.py`; see README)
- [ ] **User:** Validate public URL + cold start (after push + force-add artifacts or bootstrap on server)

---

## Success Criteria
- Retention model ROC-AUC > 0.75
- LTV model R² > 0.70
- All 3 Streamlit pages functional
- Budget optimizer returns valid allocations
- App live on Streamlit Community Cloud
- README explains business value to a hiring manager
