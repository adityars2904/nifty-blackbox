#!/usr/bin/env python3
"""
02_friction_analysis.py
========================
Model realistic transaction costs through dynamic slippage simulation.

Usage:
    cd research && python 02_friction_analysis.py
"""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.ml.data_loader import fetch_candles
from backend.ml.feature_engineering import add_features, MOVEMENT_FEATURES
from backend.ml.labels import label_movement
from backend.ml.ensemble_predictor import EnsemblePredictor
from scripts.validate_meta_filter_2025 import generate_ensemble_signals, build_meta_features, label_signal_outcomes
from research.core_execution import simulate_portfolio, THRESHOLDS, RISK_PER_TRADE, DEFAULT_CAPITAL

# ============================================================================
# CONFIG & SCENARIOS
# ============================================================================
MODEL_DIR = str(project_root / "backend" / "models")
VAULT_START = "2025-01-01 09:15:00"
VAULT_END = "2025-12-10 15:30:00"
OUTPUT_DIR = Path(__file__).parent / "outputs"

# Define fraction of ATR to use as slippage per leg
FRICTION_SCENARIOS = {
    "optimistic":  0.02,  # 2% of ATR 
    "realistic":   0.05,  # 5% of ATR
    "pessimistic": 0.10,  # 10% of ATR
}


def analyze_friction(symbol):
    print(f"\nLoading {symbol} vault data...")
    raw_5m = fetch_candles(symbol, timeframe="5m", start_date=VAULT_START, end_date=VAULT_END)
    raw_15m = fetch_candles(symbol, timeframe="15m", start_date=VAULT_START, end_date=VAULT_END)

    if len(raw_5m) == 0 or len(raw_15m) == 0:
        print(f"  ❌ No data returned from QuestDB for {symbol}")
        return []

    feat_5m = add_features(raw_5m).copy()
    feat_5m["label"] = label_movement(feat_5m)
    feat_5m = feat_5m.dropna(subset=MOVEMENT_FEATURES + ["label", "atr"]).reset_index(drop=True)
    feat_15m = add_features(raw_15m).dropna(subset=MOVEMENT_FEATURES).reset_index(drop=True)

    ensemble = EnsemblePredictor.load(MODEL_DIR, symbol=symbol)
    ens_signals = generate_ensemble_signals(feat_5m, feat_15m, ensemble)

    results = []
    
    for direction in ["UP", "DOWN"]:
        dir_signals = ens_signals[ens_signals["direction"] == direction].reset_index(drop=True)
        if len(dir_signals) == 0:
            continue

        dir_signals["win"] = label_signal_outcomes(dir_signals, feat_5m)
        dir_signals = build_meta_features(dir_signals, feat_5m, symbol)

        threshold = THRESHOLDS.get((symbol, direction), 0.55)
        meta_path = project_root / "backend" / "models" / f"meta_filter_{symbol.lower()}_{direction.lower()}.joblib"
        if not meta_path.exists():
            meta_path = project_root / "backend" / "models" / f"meta_filter_ensemble_{symbol.lower()}.joblib"

        loaded = joblib.load(meta_path)
        meta_model = loaded["model"] if isinstance(loaded, dict) and "model" in loaded else loaded
        feature_names = list(meta_model.get_booster().feature_names)

        X = dir_signals[feature_names].astype(float)
        probs = meta_model.predict_proba(X)[:, 1]
        approved = dir_signals[probs >= threshold].copy()

        if len(approved) == 0:
            continue

        # Zero friction baseline for reference
        t_base, p_base = simulate_portfolio(approved, feat_5m, slippage_frac=0.0)
        base_exp = t_base["pnl_pts"].mean() if len(t_base) > 0 else 0

        for scenario_name, slip_frac in FRICTION_SCENARIOS.items():
            t_df, p_df = simulate_portfolio(approved, feat_5m, slippage_frac=slip_frac)
            if len(t_df) == 0:
                continue
                
            win_rate = (t_df["pnl_pts"] > 0).mean() * 100
            exp_pts = t_df["pnl_pts"].mean()
            survives = exp_pts > 0
            
            # Dynamic BE win rate approx
            # avg_loss = W * avg_win => W = avg_loss / (avg_win + avg_loss)
            wins = t_df[t_df["pnl_pts"] > 0]["pnl_pts"]
            losses = abs(t_df[t_df["pnl_pts"] <= 0]["pnl_pts"])
            aw = wins.mean() if len(wins) > 0 else 0
            al = losses.mean() if len(losses) > 0 else 0
            be_wr = (al / (aw + al)) * 100 if (aw + al) > 0 else 100
            
            avg_fric = t_df["friction_pts"].mean()

            results.append({
                "Symbol": symbol,
                "Direction": direction,
                "Scenario": scenario_name,
                "Baseline Exp (pts)": round(base_exp, 1),
                "Avg Friction (pts)": round(avg_fric, 1),
                "Adj Expectancy": round(exp_pts, 1),
                "BE Win Rate": f"{be_wr:.1f}%",
                "Actual Win Rate": f"{win_rate:.1f}%",
                "Survives": "✅ YES" if survives else "❌ NO"
            })

    return results


def main():
    print("=" * 80)
    print("FRICTION ANALYSIS — True Path Simulation")
    print("=" * 80)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []

    for symbol in ["NIFTY", "BANKNIFTY"]:
        res = analyze_friction(symbol)
        all_results.extend(res)

    if not all_results:
        print("\n❌ No trades evaluated.")
        return

    results_df = pd.DataFrame(all_results)
    print("\n" + results_df.to_string(index=False))

    out_path = OUTPUT_DIR / "friction_analysis.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")

    # Verdict
    print("\n" + "=" * 80)
    print("VERDICT PER MODEL PER SCENARIO")
    print("=" * 80)
    for _, r in results_df.iterrows():
        print(f"  {r['Symbol']:10s} {r['Direction']:5s} [{r['Scenario']:12s}]  "
              f"Adj Exp: {r['Adj Expectancy']:+6.1f}  → {r['Survives']}")


if __name__ == "__main__":
    main()

