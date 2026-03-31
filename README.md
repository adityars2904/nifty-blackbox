# NIFTY ML Research System

Machine learning system for NIFTY/BANKNIFTY directional prediction using a multi-horizon ensemble + asymmetric meta-filter architecture.

## Architecture

```
Candle Data (QuestDB)
    ↓
Feature Engineering (50+ technical indicators)
    ↓
Multi-Horizon Ensemble (5m × 15m XGBoost, weighted 30/70)
    ↓
Asymmetric Meta-Filter (directional UP/DOWN models)
    ↓
Signal Approval / Rejection
```

### Models

| Model | Purpose | Features |
|-------|---------|----------|
| `movement_predictor_*_5m.joblib` | 5-minute directional prediction | 50+ technical features |
| `movement_predictor_*_15m.joblib` | 15-minute directional prediction | Same feature set |
| `meta_filter_*_up.joblib` | Meta-filter for UP signals | 21 meta-features |
| `meta_filter_*_down.joblib` | Meta-filter for DOWN signals | 21 meta-features |
| `meta_filter_ensemble_*.joblib` | Pooled meta-filter (fallback) | 19-21 features |

## Project Structure

```
├── backend/
│   ├── main.py                  # FastAPI chart-only server
│   ├── config.py                # Central configuration
│   ├── adapters/
│   │   └── questdb_adapter.py   # QuestDB HTTP adapter
│   ├── ml/
│   │   ├── data_loader.py       # QuestDB data loading
│   │   ├── feature_engineering.py  # 50+ technical indicators
│   │   ├── labels.py            # Movement labelling
│   │   ├── ensemble_predictor.py   # Multi-horizon ensemble
│   │   └── training_pipeline.py    # Model training
│   ├── models/                  # .joblib model files (gitignored)
│   ├── routers/
│   │   ├── candles.py           # OHLCV + VWAP endpoint
│   │   └── health.py            # Health check
│   └── services/
│       └── meta_filter_service.py  # Model inference wrapper
├── frontend/                    # React + Vite chart viewer
│   ├── src/
│   │   ├── App.jsx              # Auth gate + chart layout
│   │   ├── pages/ChartPage.jsx  # Candle chart page
│   │   ├── components/LiveChart.jsx  # TradingView chart
│   │   └── auth/GoogleAuth.jsx  # Google OAuth
│   └── vite.config.js
├── research/                    # Standalone research scripts
│   ├── 01_expectancy_analysis.py
│   ├── 02_friction_analysis.py
│   ├── 03_feature_fragility.py
│   ├── 04_regime_analysis.py
│   ├── 05_vectorbt_backtest.py
│   ├── 06_factor_ic.py
│   └── outputs/                 # CSV results (gitignored)
├── scripts/                     # Training & validation scripts
│   ├── train_meta_filter.py
│   └── validate_meta_filter_2025.py
└── data/processed/              # Parquet files (gitignored)
```

## Quick Start

### Prerequisites
- Python 3.12+ with venv
- QuestDB running on `localhost:9000`
- Node.js 18+ (for frontend)

### Backend
```bash
cd backend
source ../venv/bin/activate
uvicorn main:app --reload --port 8000
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

### Research Scripts
```bash
cd research
source ../venv/bin/activate
python 01_expectancy_analysis.py   # → outputs/expectancy_results.csv
python 02_friction_analysis.py     # → outputs/friction_analysis.csv
python 03_feature_fragility.py     # → outputs/fragility_results.csv
python 04_regime_analysis.py       # → outputs/regime_analysis.csv
python 05_vectorbt_backtest.py     # → outputs/equity_curve.csv
python 06_factor_ic.py             # → outputs/factor_ic.csv
```

## Key Design Decisions

1. **Asymmetric thresholds**: UP model at 0.60, DOWN model at 0.52 — reflects empirical finding that DOWN signals need less filtering.
2. **15m veto power**: If the 15-minute model predicts NEUTRAL, the signal is rejected regardless of 5m confidence.
3. **Feature count flexibility**: Meta-filter reads feature names from the model itself (`model.get_booster().feature_names`), so 19-feature and 21-feature models coexist without code changes.
4. **No database dependency in research**: `meta_filter_service.py` uses neutral defaults for all database-dependent features; research scripts override via kwargs.

## Data

- **Training**: 2022-04-11 to 2024-04-30
- **Validation**: 2024-05-01 to 2024-12-31
- **Vault (OOS)**: 2025-01-01 to 2025-12-10

All candle data lives in QuestDB. Parquet fallback files are in `data/processed/`.
