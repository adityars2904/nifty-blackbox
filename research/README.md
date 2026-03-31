# Research Suite

Standalone quant scripts for evaluating the NIFTY ML prediction pipeline.  
Each script imports from `backend/ml/` and `backend/services/meta_filter_service.py` directly.

## Prerequisites

- QuestDB running with vault data (2025-01-01 onwards)
- All model `.joblib` files in `backend/models/`
- Python venv activated (`source ../venv/bin/activate`)

## Scripts

| # | Script | Description | Depends On |
|---|--------|-------------|------------|
| 01 | `01_expectancy_analysis.py` | Per-directional-model trade expectancy, profit factor | QuestDB |
| 02 | `02_friction_analysis.py` | Transaction cost impact (3 scenarios) | Script 01 output |
| 03 | `03_feature_fragility.py` | Test `consecutive_wins` dependency via retrain | QuestDB |
| 04 | `04_regime_analysis.py` | ADX/vol regime × meta-filter performance | QuestDB |
| 05 | `05_vectorbt_backtest.py` | Full pipeline backtest with equity curve | QuestDB |
| 06 | `06_factor_ic.py` | Feature IC and rank IC analysis | QuestDB |

## How to Run

```bash
cd research

# Run in order (02 depends on 01's output)
python 01_expectancy_analysis.py
python 02_friction_analysis.py
python 03_feature_fragility.py
python 04_regime_analysis.py
python 05_vectorbt_backtest.py
python 06_factor_ic.py
```

## Output

All results are saved to `research/outputs/`:

| File | Source |
|------|--------|
| `expectancy_results.csv` | Script 01 |
| `friction_analysis.csv` | Script 02 |
| `fragility_results.csv` | Script 03 |
| `regime_analysis.csv` | Script 04 |
| `equity_curve.csv` | Script 05 |
| `backtest_summary.csv` | Script 05 |
| `factor_ic.csv` | Script 06 |
| `test_models/` | Script 03 (never deployed) |
