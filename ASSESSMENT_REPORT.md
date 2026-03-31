# NIFTY ML System — File Assessment Report

**Generated:** 2026-03-30  
**Purpose:** Pre-change assessment of every file. Categories: KEEP | DELETE | SIMPLIFY | MOVE  
**Rule:** No changes happen until this report is reviewed and confirmed.

---

## Summary Counts

| Category | Count |
|----------|-------|
| KEEP     | ~30 files |
| DELETE   | ~35 files |
| SIMPLIFY | ~6 files  |
| MOVE     | ~7 files  |

---

## Root Level

| File / Dir | Category | Reason |
|---|---|---|
| `.env` | **KEEP** (gitignored) | Contains credentials — must remain gitignored, never committed |
| `.env.example` | **KEEP** | Documents required env vars; safe to commit |
| `.gitignore` | **SIMPLIFY** | Currently only 2 lines; expand significantly before publish |
| `.pyre_configuration` | **DELETE** | Pyre type-checker config; not needed for publication |
| `.vscode/` | **DELETE** | Editor config; personal, should not be committed |
| `.DS_Store` | **DELETE** | macOS metadata; should be gitignored |
| `Implementation_Plan.md` | **DELETE** | Internal dev planning doc; superseded by README |
| `NIFTY_AGENT_HANDOFF-2.md` | **DELETE** | Internal agent handoff doc; not for public consumption |
| `UPSTOX_DATA_INGESTION.md` | **DELETE** | Upstox-specific ingestion doc; references live creds system |
| `README.md` | **NEW** | Does not exist yet; must be created for GitHub publication |
| `ASSESSMENT_REPORT.md` | **DELETE** (post-publish) | This file; internal only |
| `package-lock.json` | **DELETE** | Root-level lock file with no corresponding package.json |
| `pyproject.toml` | **KEEP** | Python package config |
| `pyrightconfig.json` | **DELETE** | Type-checker config; editor tooling only |
| `requirements.txt` | **KEEP** | Python dependencies |
| `setup_env.py` | **KEEP** | Environment setup helper |
| `walkthrough.md.resolved` | **DELETE** | Internal AI-resolved walkthrough; not for public repo |
| `venv/` | **DELETE** (from git) | Virtual env must never be committed; add to .gitignore |
| `outputs/` | **KEEP** (partially) | Contains visualizations subdirectory; add to .gitignore |
| `.pytest_cache/` | **DELETE** (from git) | Auto-generated; add to .gitignore |

---

## backend/

### backend/ (root files)

| File | Category | Reason |
|---|---|---|
| `__init__.py` | **KEEP** | Package marker |
| `_check_data.py` | **DELETE** | Debug/utility script; not part of system |
| `config.py` | **SIMPLIFY** | Keep but strip unused trading/scheduler constants |
| `database.py` | **DELETE** | SQLite ORM for paper trading, signals, trades; not needed |
| `db.py` | **DELETE** | Wrapper around database.py; both deleted |
| `financials_cache.db` | **DELETE** (from git) | SQLite file; add `*.db` to .gitignore |
| `main.py` | **SIMPLIFY** | Strip to chart-only; replace per spec |
| `replay.py` | **DELETE** | Full replay engine (706 lines); paper trading feature |
| `scheduler.py` | **DELETE** | APScheduler for live trading cron jobs |
| `smoke_test.py` | **KEEP** | Useful for verifying backend health |

### backend/adapters/

| File | Category | Reason |
|---|---|---|
| `questdb_adapter.py` | **KEEP** | Core — fetches candles; needed by both API and research scripts |

### backend/agent/

| File | Category | Reason |
|---|---|---|
| `__init__.py` | **DELETE** | No longer needed when agent/ is removed |
| `__innit__.py` | **DELETE** | Typo duplicate; dead file |
| `agent.py` | **DELETE** | Live trading agent orchestration |
| `confidence_validator.py` | **DELETE** | Trading confidence validation; live system only |
| `core_agent.py` | **DELETE** | Core trading agent loop |
| `decision_engine.py` | **DELETE** | Trading decision orchestration |
| `dual_explainer.py` | **DELETE** | Trade explanation generator |
| `dual_model_agent.py` | **DELETE** | Dual-model trading agent |
| `explainer.py` | **DELETE** | Signal explainer; live trading only |
| `indicator_selector.py` | **DELETE** | Dynamic indicator selection; live system only |
| `movement_predictor.py` | **DELETE** | Thin wrapper; duplicates ml/ functionality |
| `position_sizer.py` | **DELETE** | Position sizing; live trading only |
| `regime_classifier.py` | **DELETE** | Regime classification; live trading only |
| `risk_assessor.py` | **DELETE** | Risk assessment; live trading only |
| `signal_generator.py` | **DELETE** | Signal generation orchestration; live trading |

> **Verdict:** Entire `backend/agent/` directory — DELETE

### backend/data/

| File | Category | Reason |
|---|---|---|
| `evaluation.db` | **DELETE** (from git) | SQLite DB; add `*.db` to .gitignore |
| `event_calendar.json` | **DELETE** | Live trading calendar; not needed |
| `model_runtime.db` | **DELETE** (from git) | SQLite DB; add `*.db` to .gitignore |
| `paper_trading.db` | **DELETE** (from git) | SQLite DB for paper trades; add `*.db` to .gitignore |

> **Note:** The `data/` directory itself can remain as an empty placeholder if needed.

### backend/data_ingestion/

| File | Category | Reason |
|---|---|---|
| `__init__.py` | **DELETE** | Package marker; dir being deleted |
| `backfill_upstox.py` | **DELETE** | Upstox historical backfill; uses live API credentials |
| `daily_ingest.py` | **DELETE** | Daily data ingestion from Upstox |
| `env_loader.py` | **DELETE** | Upstox env loader |
| `live_feed.py` | **DELETE** | Upstox WebSocket live feed; uses credentials |
| `morning_check.py` | **DELETE** | Session pre-market check; live trading |
| `questdb_writer.py` | **DELETE** | Writes candles to QuestDB; data already in place |
| `upstox_auth.py` | **DELETE** | Upstox OAuth implementation; **SECURITY RISK** — contains auth logic tied to exposed API key |

> **Verdict:** Entire `backend/data_ingestion/` directory — DELETE

### backend/ml/

| File | Category | Reason |
|---|---|---|
| `__init__.py` | **KEEP** | Package marker |
| `__innit__.py` | **DELETE** | Typo duplicate; dead file |
| `data_loader.py` | **KEEP** | Loads training/vault data; needed by research scripts |
| `ensemble_predictor.py` | **KEEP** | Core ML inference; needed by research scripts |
| `feature_engineer.py` | **DELETE** | Thin stub/alias; 413 bytes, likely just re-exports |
| `feature_engineering.py` | **KEEP** | Core 21-feature pipeline; DO NOT MODIFY |
| `feature_engineering.py.backup` | **DELETE** | Backup file; not for publication |
| `feature_engineering.py.backup.Feb13` | **DELETE** | Backup file; not for publication |
| `labels.py` | **KEEP** | Signal outcome labelling; needed by research scripts |
| `labels.py.backup` | **DELETE** | Backup file; not for publication |
| `multi_horizon_predictor.py` | **KEEP** | 30m ensemble predictor; referenced by training |
| `regime_trainer.py` | **DELETE** | Empty file (0 bytes) |
| `run_pipeline.py` | **KEEP** | Pipeline runner utility |
| `training_pipeline.py` | **KEEP** | Model training pipeline; needed for retraining |
| `training_pipeline.py.backup` | **DELETE** | Backup file |
| `training_pipeline.py.backup.feb13` | **DELETE** | Backup file |

### backend/models/

| File | Category | Reason |
|---|---|---|
| `meta_filter_3model_banknifty.joblib` | **KEEP** (local only) | Production model; **NEVER COMMIT** — add `*.joblib` to .gitignore |
| `meta_filter_3model_nifty.joblib` | **KEEP** (local only) | Production model; **NEVER COMMIT** |
| `meta_filter_banknifty_down.joblib` | **KEEP** (local only) | Directional filter; **NEVER COMMIT** |
| `meta_filter_banknifty_up.joblib` | **KEEP** (local only) | Directional filter; **NEVER COMMIT** |
| `meta_filter_ensemble_banknifty.joblib` | **KEEP** (local only) | Ensemble meta-filter; **NEVER COMMIT** |
| `meta_filter_ensemble_nifty.joblib` | **KEEP** (local only) | Ensemble meta-filter; **NEVER COMMIT** |
| `meta_filter_nifty_down.joblib` | **KEEP** (local only) | Directional filter; **NEVER COMMIT** |
| `meta_filter_nifty_up.joblib` | **KEEP** (local only) | Directional filter; **NEVER COMMIT** |
| `movement_predictor_banknifty_15m.joblib` | **KEEP** (local only) | Foundation model; **NEVER COMMIT** |
| `movement_predictor_banknifty_30m.joblib` | **KEEP** (local only) | Foundation model; **NEVER COMMIT** |
| `movement_predictor_banknifty_5m.joblib` | **KEEP** (local only) | Foundation model; **NEVER COMMIT** |
| `movement_predictor_nifty_15m.joblib` | **KEEP** (local only) | Foundation model; **NEVER COMMIT** |
| `movement_predictor_nifty_30m.joblib` | **KEEP** (local only) | Foundation model; **NEVER COMMIT** |
| `movement_predictor_nifty_5m.joblib` | **KEEP** (local only) | Foundation model; **NEVER COMMIT** |
| `risk_assessor_banknifty_15m.joblib` | **KEEP** (local only) | Risk model; **NEVER COMMIT** |
| `risk_assessor_banknifty_5m.joblib` | **KEEP** (local only) | Risk model; **NEVER COMMIT** |
| `risk_assessor_nifty_15m.joblib` | **KEEP** (local only) | Risk model; **NEVER COMMIT** |
| `risk_assessor_nifty_5m.joblib` | **KEEP** (local only) | Risk model; **NEVER COMMIT** |
| `Archive/` | **DELETE** (from git) | Old model versions; should be gitignored |
| `training_logs/` | **DELETE** (from git) | Training run logs; add to .gitignore |

> **CRITICAL:** All 18 `.joblib` files total ~15MB. `*.joblib` MUST be in .gitignore. Document training instructions in README.

### backend/routers/

| File | Category | Reason |
|---|---|---|
| `__init__.py` | **KEEP** | Package marker |
| `agent.py` | **DELETE** | Agent API endpoints; live trading only |
| `candles.py` | **KEEP** | Core — serves historical OHLCV data to frontend |
| `health.py` | **KEEP** | Health check endpoint |
| `performance.py` | **DELETE** | Performance metrics API; paper trading only |
| `replay.py` | **DELETE** | Replay API endpoints; live trading only |
| `settings.py` | **DELETE** | Runtime settings API; paper trading only |
| `signals.py` | **DELETE** | Signal recording API; paper trading only |
| `stats.py` | **DELETE** | Trading stats API; paper trading only |
| `trades.py` | **DELETE** | Trade recording API; paper trading only |

### backend/services/

| File | Category | Reason |
|---|---|---|
| `__init__.py` | **DELETE** | Package marker; dir being deleted |
| `market_context.py` | **DELETE** | Market context service; live trading |
| `meta_filter_service.py` | **DELETE** | Meta-filter inference service wrapping models (research scripts import from ml/ directly) |
| `paper_trading_service.py` | **DELETE** | Paper trade execution; live trading |
| `recalibration_scheduler.py` | **DELETE** | Live model recalibration scheduler |
| `risk_manager.py` | **DELETE** | Live risk management |
| `runtime_settings.py` | **DELETE** | Runtime settings store; live trading |
| `session_prefetch.py` | **DELETE** | Pre-market session prep; live trading |
| `signal_service.py` | **DELETE** | Signal detection service; live trading |
| `stats_service.py` | **DELETE** | Trading statistics aggregation; live trading |
| `trade_monitor.py` | **DELETE** | Trade monitoring; live trading |

> **Verdict:** Entire `backend/services/` directory — DELETE

### backend/training/

| File | Category | Reason |
|---|---|---|
| `__init__.py` | **KEEP** | Package marker |
| `train_multihorizon.py` | **KEEP** | Training script for multi-horizon models |
| `validate_multihorizon.py` | **KEEP** | Validation script |

---

## frontend/

### frontend/ (root)

| File | Category | Reason |
|---|---|---|
| `index.html` | **KEEP** | Entry point |
| `package.json` | **KEEP** | Dependencies including Lightweight Charts |
| `package-lock.json` | **KEEP** | Lock file |
| `vite.config.js` | **KEEP** | Vite + proxy config; DO NOT TOUCH |
| `tailwind.config.js` | **KEEP** | Tailwind configuration |
| `postcss.config.js` | **KEEP** | PostCSS config |

### frontend/src/

| File | Category | Reason |
|---|---|---|
| `main.jsx` | **KEEP** | React entry point |
| `App.jsx` | **SIMPLIFY** | The authenticated portion is stripped to `<ChartPage />`; the auth guard and login page stay |
| `index.css` | **KEEP** | Global styles |

### frontend/src/auth/

| File | Category | Reason |
|---|---|---|
| `GoogleAuth.jsx` | **KEEP — DO NOT TOUCH** | Sacred; Google OAuth implementation |

### frontend/src/api/

| File | Category | Reason |
|---|---|---|
| `client.js` | **SIMPLIFY** | Keep candles fetch; remove signal/trade/stats/replay/settings fetch functions |

### frontend/src/hooks/

| File | Category | Reason |
|---|---|---|
| `usePolling.js` | **DELETE** | Polling hook for live data; not needed in chart-only mode |
| `useStats.js` | **DELETE** | Stats polling hook; not needed |

### frontend/src/pages/

| File | Category | Reason |
|---|---|---|
| `Dashboard.jsx` | **DELETE** | Full trading dashboard with signals/trades/stats panels |
| `ModelPerformance.jsx` | **DELETE** | Model performance page with replay; not needed |

> **NEW:** A `ChartPage.jsx` must be created here replacing both deleted pages.

### frontend/src/components/

| File | Category | Reason |
|---|---|---|
| `AlertPanel.jsx` | **DELETE** | Signal alerts; live trading only |
| `DrawdownMeter.jsx` | **DELETE** | Drawdown display; live trading only |
| `LiveChart.jsx` | **KEEP** | Core — candlestick chart with Lightweight Charts; this IS the chart |
| `MetaFilterToggle.jsx` | **DELETE** | Meta-filter on/off toggle; live trading only |
| `PerformanceSection.jsx` | **DELETE** | Performance metrics display; live trading only |
| `ReplayPanel.jsx` | **DELETE** | Replay controls; live trading only |
| `SettingsPanel.jsx` | **DELETE** | Settings drawer; live trading only |
| `SignalFeed.jsx` | **DELETE** | Signal feed display; live trading only |
| `StatsBar.jsx` | **DELETE** | Stats bar; live trading only |
| `TradeTable.jsx` | **DELETE** | Trade table; live trading only |

> **KEEP ONLY:** `LiveChart.jsx`

### frontend/src/utils/

| File | Category | Reason |
|---|---|---|
| `indicators.js` | **KEEP** | EMA, SMA, Bollinger, VWAP, RSI, MACD calculations; needed by ChartPage |

---

## scripts/ (top-level)

| File | Category | Reason |
|---|---|---|
| `analyze_movement_performance.py` | **KEEP** | Performance analysis utility; relevant to research |
| `generate_performance_visualizations.py` | **KEEP** | Visualization generator; relevant to research |
| `train_foundation_models.py` | **KEEP** | Foundation model training; needed for reproducibility |
| `train_meta_filter.py` | **KEEP** | Meta-filter training; needed for reproducibility |
| `train_meta_filter.py.splitbrain.md` | **DELETE** | Internal AI deliberation doc; not for publication |
| `validate_ensemble.py` | **KEEP** | Ensemble validation script |
| `validate_meta_filter_2025.py` | **KEEP** | 2025 vault validation script |

---

## data/ (top-level)

| File / Dir | Category | Reason |
|---|---|---|
| `Splitting.ipynb` | **KEEP** | Data splitting notebook; research artifact |
| `data/` | **KEEP** | Nested data directory |
| `processed/` | **KEEP** (local only) | Parquet files for NIFTY/BANKNIFTY; add `*.parquet` to .gitignore for large files |
| `raw/` | **KEEP** (local only) | Raw spot data; add `*.parquet`, `*.csv` to .gitignore |

> **Note:** Parquet and large CSV files should NOT be committed. Add to .gitignore.

---

## questdb/

| File / Dir | Category | Reason |
|---|---|---|
| `docker-compose.yml` | **KEEP** | QuestDB setup; needed for dev environment |
| `import/NIFTY.parquet` | **DELETE** (from git) | Large data file; gitignore |
| `data/` | **DELETE** (from git) | QuestDB runtime data; should be gitignored entirely |
| `data/import/` | **DELETE** (from git) | Import parquet files; gitignore |
| `data/db/` | **DELETE** (from git) | QuestDB database files; gitignore |
| `data/conf/` | **KEEP** | QuestDB config (server.conf, mime.types) |
| `data/public/` | **DELETE** (from git) | QuestDB web console; large bundled JS/fonts |

---

## New Directories to Create

| Directory | Category | Reason |
|---|---|---|
| `research/` | **NEW** | All 6 research scripts |
| `research/outputs/` | **NEW** | Research outputs (mostly gitignored) |

---

## Security Findings

> **ACTION REQUIRED before any git push:**

1. `.env` exists at root — contains credentials. Currently gitignored. Confirm it stays gitignored.
2. `backend/data_ingestion/upstox_auth.py` — contains Upstox OAuth token handling logic. No hardcoded credentials found in a quick scan, but verify `ac74d824` and `wenkf9i5te` are not present in any tracked file.
3. `questdb/data/import/trades/paper_trades.json` — may contain historical trade data. Check contents before committing.
4. `backend/data/*.db` — SQLite files; should be gitignored.

---

## Files Requiring Action in .gitignore

```
# Add all of these:
.env
*.env
venv/
node_modules/
__pycache__/
*.pyc
*.joblib
*.db
.DS_Store
.pytest_cache/
outputs/
questdb/data/db/
questdb/data/public/
questdb/data/import/
questdb/import/
data/processed/
data/raw/
data/data/
backend/models/Archive/
backend/models/training_logs/
research/outputs/*.png
research/outputs/test_models/
.pyre_configuration
pyrightconfig.json
walkthrough.md.resolved
```

---

## Final Decision Summary

### What stays in the repository after cleanup

**Backend (kept):**  
`main.py` (simplified), `config.py` (simplified), `adapters/questdb_adapter.py`, `routers/health.py`, `routers/candles.py`, `ml/feature_engineering.py`, `ml/ensemble_predictor.py`, `ml/labels.py`, `ml/data_loader.py`, `ml/multi_horizon_predictor.py`, `ml/training_pipeline.py`, `ml/run_pipeline.py`, `training/train_multihorizon.py`, `training/validate_multihorizon.py`

**Frontend (kept):**  
Auth system (`auth/GoogleAuth.jsx`), chart component (`LiveChart.jsx`), indicators (`utils/indicators.js`), new `ChartPage.jsx` (to create), simplified `App.jsx`, `api/client.js` (simplified)

**Scripts (kept):**  
All 6 training/validation scripts except `.splitbrain.md` internal doc

**Research (new):**  
6 research scripts + README + outputs directory

**Root:**  
`README.md` (new), `requirements.txt`, `setup_env.py`, `.env.example`, `.gitignore` (expanded), `pyproject.toml`, `questdb/docker-compose.yml`

---

**Status: AWAITING HUMAN REVIEW — No changes have been made.**
