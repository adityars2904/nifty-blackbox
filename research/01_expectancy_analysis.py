#!/usr/bin/env python3
"""
01_expectancy_analysis.py
==========================
Analyzes the theoretical expectancy of the system using true event-driven portfolio trades,
rather than label-based static points.
"""

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

MODEL_DIR = str(project_root / "backend" / "models")
VAULT_START = "2025-01-01 09:15:00"
VAULT_END = "2025-12-10 15:30:00"
OUTPUT_DIR = Path(__file__).parent / "outputs"


def analyze_symbol(symbol):
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
    print(f"✅ Loaded 5m model:  {MODEL_DIR}/movement_predictor_{symbol.lower()}_5m.joblib")
    print(f"✅ Loaded 15m model: {MODEL_DIR}/movement_predictor_{symbol.lower()}_15m.joblib")

    ens_signals = generate_ensemble_signals(feat_5m, feat_15m, ensemble)

    results = []
    for direction in ["UP", "DOWN"]:
        print(f"\n  Computing {symbol} {direction}...")
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
        
        jsd_idx = feature_names.index("JSD") if "JSD" in feature_names else -1
        if jsd_idx >= 0:
            jsd_vals = dir_signals[feature_names[jsd_idx]]
            print(f"  Feature #{jsd_idx} (JSD): mean={jsd_vals.mean():.4f}  max={jsd_vals.max():.4f}")

        if len(approved) == 0:
            print("  0 signals approved")
            continue

        trades_df, portfolio_df = simulate_portfolio(approved, feat_5m, initial_capital=DEFAULT_CAPITAL)
        
        if len(trades_df) == 0:
            continue

        trades = len(trades_df)
        win_rate = (trades_df["pnl_pts"] > 0).mean() * 100
        wins = trades_df[trades_df["pnl_pts"] > 0]["pnl_pts"].mean()
        losses = abs(trades_df[trades_df["pnl_pts"] <= 0]["pnl_pts"].mean())
        expectancy_pts = trades_df["pnl_pts"].mean()
        
        pnl_rupees_win = trades_df[trades_df["pnl_rupees"] > 0]["pnl_rupees"].sum()
        pnl_rupees_loss = abs(trades_df[trades_df["pnl_rupees"] <= 0]["pnl_rupees"].sum())
        pf = pnl_rupees_win / pnl_rupees_loss if pnl_rupees_loss > 0 else float('inf')
        
        total_days = pd.to_datetime(trades_df["entry_ts"]).dt.date.nunique()
        spd = trades / total_days if total_days > 0 else 0

        print(f"    Trades: {trades}  Win%: {win_rate:.1f}%  Expectancy: +{expectancy_pts:.1f} pts  PF: {pf:.2f}")

        results.append({
            "Symbol": symbol,
            "Direction": direction,
            "Threshold": threshold,
            "Trades": trades,
            "Win%": round(win_rate, 1),
            "Avg Win (pts)": round(wins, 1),
            "Avg Loss (pts)": round(losses, 1),
            "Expectancy (pts)": round(expectancy_pts, 1),
            "Profit Factor": round(pf, 2),
            "Signals/Day": round(spd, 1)
        })

    return results


def main():
    print("=" * 70)
    print("EXPECTANCY ANALYSIS — Vault 2025")
    print(f"Period: {VAULT_START} to {VAULT_END}")
    print(f"SL = 1.0×ATR, TP = 1.5×ATR, True Portfolio Path Sim")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []
    
    for symbol in ["NIFTY", "BANKNIFTY"]:
        res = analyze_symbol(symbol)
        all_results.extend(res)
        
    if not all_results:
        print("\n❌ No trades evaluated.")
        return

    df_res = pd.DataFrame(all_results)
    
    print("\n" + "=" * 70)
    print("EXPECTANCY TABLE")
    print("=" * 70)
    print(df_res.to_string(index=False))
    
    out_path = OUTPUT_DIR / "expectancy_results.csv"
    df_res.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")
    
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    for _, row in df_res.iterrows():
        status = "✅ POSITIVE" if row["Expectancy (pts)"] > 0 else "❌ NEGATIVE"
        print(f"  {row['Symbol']:<10} {row['Direction']:<6} Expectancy: {row['Expectancy (pts)']:+5.1f} pts → {status}")


if __name__ == "__main__":
    main()
