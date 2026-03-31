#!/usr/bin/env python3
"""
06_factor_ic.py
================
Compute Information Coefficient (IC) and rank IC for each meta-filter
feature against continuous PnL (Expectancy).

Now uses Newey-West standard errors to account for autocorrelated 
returns from overlapping trades.

Usage:
    cd research && python 06_factor_ic.py
"""

from __future__ import annotations
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import joblib
from scipy.stats import spearmanr, pearsonr, t as t_dist

from backend.ml.data_loader import fetch_candles
from backend.ml.feature_engineering import add_features, MOVEMENT_FEATURES
from backend.ml.labels import label_movement
from backend.ml.ensemble_predictor import EnsemblePredictor

sys.path.insert(0, str(project_root / "scripts"))
from validate_meta_filter_2025 import generate_ensemble_signals, build_meta_features, label_signal_outcomes
from research.core_execution import simulate_portfolio

# ============================================================================
# CONFIG
# ============================================================================

MODEL_DIR   = str(project_root / "backend" / "models")
VAULT_START = "2025-01-01 09:15:00"
VAULT_END   = "2025-12-10 15:30:00"
OUTPUT_DIR  = Path(__file__).parent / "outputs"


def compute_nw_pvalue(x, y, max_lag=10):
    """
    Computes Newey-West standard errors for a single independent variable (Factor).
    Since both vectors are standardized, OLS Beta = Pearson IC.
    Returns the robust p-value for the Factor IC.
    """
    N = len(x)
    
    # Standardize
    x_s = (x - np.mean(x)) / np.std(x)
    y_s = (y - np.mean(y)) / np.std(y)
    
    # OLS coefficient is Pearson IC
    beta = np.sum(x_s * y_s) / np.sum(x_s**2)
    residuals = y_s - beta * x_s
    
    # HAC (Newey-West) Covariance
    Q = 0
    # Center 
    for i in range(N):
        Q += (residuals[i]**2) * (x_s[i]**2)
        
    # Lagged
    for l in range(1, max_lag + 1):
        w = 1 - l / (max_lag + 1)
        for i in range(l, N):
            Q += w * residuals[i] * residuals[i-l] * 2 * (x_s[i] * x_s[i-l])
            
    Q = Q / N
    X_XX_inv = 1.0 / (np.sum(x_s**2) / N)
    
    variance = (X_XX_inv * Q * X_XX_inv) / N
    se = np.sqrt(variance) if variance > 0 else 1e-9
    
    t_stat = beta / se
    p_val = t_dist.sf(np.abs(t_stat), df=N-2) * 2
    
    return beta, p_val


def compute_ic_table(signals, feature_names):
    """Compute Pearson IC, Spearman Rank IC, and Newey-West p-values per feature vs PnL."""
    y = signals["pnl_pts"].values.astype(float)
    rows = []

    for feat in feature_names:
        x = signals[feat].values.astype(float)
        valid = np.isfinite(x) & np.isfinite(y)
        if valid.sum() < 30:
            continue

        x_v, y_v = x[valid], y[valid]

        pearson_ic, _ = pearsonr(x_v, y_v)
        spearman_ic, spearman_p = spearmanr(x_v, y_v)
        
        # Newey-West for the continuous PnL target
        nw_beta, nw_p = compute_nw_pvalue(x_v, y_v, max_lag=10)

        rows.append({
            "Feature":      feat,
            "Pearson IC":   round(pearson_ic, 4),
            "NW p-value":   f"{nw_p:.4f}",
            "Rank IC":      round(spearman_ic, 4),
            "Rank p":       f"{spearman_p:.4f}",
            "|NW IC|":      round(abs(nw_beta), 4),
            "Significant":  "✅" if nw_p < 0.05 else "—",
        })

    if not rows:
        return pd.DataFrame()
        
    df = pd.DataFrame(rows).sort_values("|NW IC|", ascending=False)
    return df


def main():
    print("=" * 80)
    print("FACTOR IC ANALYSIS — Expectancy Targets + Newey-West SE")
    print(f"Period: {VAULT_START} to {VAULT_END}")
    print("=" * 80)
    print()

    all_ics = []

    for symbol in ["NIFTY", "BANKNIFTY"]:
        print(f"\nEvaluating {symbol}...")
        raw_5m  = fetch_candles(symbol, timeframe="5m",  start_date=VAULT_START, end_date=VAULT_END)
        raw_15m = fetch_candles(symbol, timeframe="15m", start_date=VAULT_START, end_date=VAULT_END)

        if len(raw_5m) == 0:
            continue

        feat_5m = add_features(raw_5m).copy()
        feat_5m["label"] = label_movement(feat_5m)
        feat_5m = feat_5m.dropna(subset=MOVEMENT_FEATURES + ["label", "atr"]).reset_index(drop=True)
        feat_15m = add_features(raw_15m).dropna(subset=MOVEMENT_FEATURES).reset_index(drop=True)

        ensemble = EnsemblePredictor.load(MODEL_DIR, symbol=symbol)
        signals = generate_ensemble_signals(feat_5m, feat_15m, ensemble)

        for direction in ["UP", "DOWN"]:
            dir_signals = signals[signals["direction"] == direction].reset_index(drop=True)
            if len(dir_signals) < 30:
                continue

            dir_signals["win"] = label_signal_outcomes(dir_signals, feat_5m)
            dir_signals = build_meta_features(dir_signals, feat_5m, symbol)

            # Get continuous PnL points via core_execution simulator 
            # We force accept all signals to measure factor significance
            t_df, _ = simulate_portfolio(dir_signals, feat_5m)
            if len(t_df) == 0:
                continue
                
            # Merge PnL points cleanly
            t_df["timestamp"] = pd.to_datetime(t_df["entry_ts"]).dt.tz_localize(None)
            dir_signals["timestamp"] = pd.to_datetime(dir_signals["timestamp"]).dt.tz_localize(None)
            dir_signals = dir_signals.merge(t_df[["timestamp", "pnl_pts"]], on="timestamp", how="inner")

            if len(dir_signals) < 30:
                continue

            # Get feature names from the production model
            meta_path = project_root / "backend" / "models" / f"meta_filter_{symbol.lower()}_{direction.lower()}.joblib"
            if not meta_path.exists():
                meta_path = project_root / "backend" / "models" / f"meta_filter_ensemble_{symbol.lower()}.joblib"
            if not meta_path.exists():
                continue

            loaded = joblib.load(meta_path)
            meta_model = loaded["model"] if isinstance(loaded, dict) and "model" in loaded else loaded
            feature_names = list(meta_model.get_booster().feature_names)

            ic_table = compute_ic_table(dir_signals, feature_names)
            if len(ic_table) == 0:
                continue
                
            ic_table.insert(0, "Symbol", symbol)
            ic_table.insert(1, "Direction", direction)

            print(f"\n  {symbol} {direction} — Top 5 by |NW IC|:")
            top5 = ic_table.head(5)
            for _, r in top5.iterrows():
                print(f"    {r['Feature']:30s}  NW IC={r['Pearson IC']:+.4f}  NW-p={r['NW p-value']}  {r['Significant']}")

            all_ics.append(ic_table)

    if not all_ics:
        print("\n❌ No IC results.")
        return

    combined = pd.concat(all_ics, ignore_index=True)

    print("\n" + "=" * 80)
    print("FULL FACTOR IC TABLE")
    print("=" * 80)
    print(combined.to_string(index=False))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "factor_ic.csv"
    combined.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")

    print("\n" + "=" * 80)
    print("CROSS-MODEL ROBUSTNESS (Significant in ALL models)")
    print("=" * 80)
    sig_counts = combined[combined["Significant"] == "✅"].groupby("Feature").size()
    n_models = len(combined.groupby(["Symbol", "Direction"]).size())
    consistent = sig_counts[sig_counts == n_models]
    if len(consistent) > 0:
        for feat, _ in consistent.items():
            avg_ic = combined[combined["Feature"] == feat]["|NW IC|"].mean()
            print(f"  {feat:30s}  avg |NW IC| = {avg_ic:.4f}")
    else:
        print("  No features are significantly predicting PnL across all 4 models.")
    print()


if __name__ == "__main__":
    main()

