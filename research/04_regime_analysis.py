#!/usr/bin/env python3
"""
04_regime_analysis.py
======================
Rule-based regime classifier using ADX and volatility on rolling T-1 data.
Evaluates portfolio simulation with and without Regime-Based Position Sizing.

Usage:
    cd research && python 04_regime_analysis.py
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
from research.core_execution import simulate_portfolio, THRESHOLDS

MODEL_DIR   = str(project_root / "backend" / "models")
VAULT_START = "2025-01-01 09:15:00"
VAULT_END   = "2025-12-10 15:30:00"
OUTPUT_DIR  = Path(__file__).parent / "outputs"

# ADX / volatility thresholds for regime classification
ADX_TRENDING_THRESHOLD   = 25   # ADX > 25 → trend
ADX_CHOPPY_THRESHOLD     = 20   # ADX < 20 → choppy
VOL_EXPANSION_THRESHOLD  = 1.3  # ATR/ATR_20 > 1.3 → volatile

REGIME_SCALARS = {
    "TRENDING": 1.0,
    "VOLATILE": 0.5,
    "CHOPPY":   0.5,
    "MIXED":    0.8
}


def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high  = df["high"].astype(float)
    low   = df["low"].astype(float)
    close = df["close"].astype(float)

    plus_dm  = high.diff()
    minus_dm = -low.diff()
    plus_dm  = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)

    atr      = tr.rolling(period).mean()
    plus_di  = 100 * (plus_dm.rolling(period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr)

    dx  = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    adx = dx.rolling(period).mean()
    return adx


def classify_regime(adx_val: float, vol_ratio: float) -> str:
    """Classify regime purely based on scalars."""
    if pd.isna(adx_val) or pd.isna(vol_ratio):
        return "MIXED"
    if vol_ratio > VOL_EXPANSION_THRESHOLD:
        return "VOLATILE"
    if adx_val > ADX_TRENDING_THRESHOLD:
        return "TRENDING"
    if adx_val < ADX_CHOPPY_THRESHOLD:
        return "CHOPPY"
    return "MIXED"


def main():
    print("=" * 80)
    print("REGIME ANALYSIS — Dynamic Position Sizing Simulation")
    print(f"Period: {VAULT_START} to {VAULT_END}")
    print("=" * 80)

    all_results = []

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

        # -----------------------------------------------------------------
        # No Lookahead Bias Regime
        # -----------------------------------------------------------------
        feat_5m["adx"] = compute_adx(feat_5m)
        feat_5m["atr_75"] = feat_5m["atr"].rolling(75).mean() # roughly 1 day
        feat_5m["vol_ratio"] = feat_5m["atr"] / feat_5m["atr_75"].replace(0, np.nan)
        
        # Calculate trailing 75-bar averages (approx 1 day) shifted by 1 to strictly prevent looking at current bar
        feat_5m["trailing_adx"] = feat_5m["adx"].rolling(75).mean().shift(1)
        feat_5m["trailing_vol"] = feat_5m["vol_ratio"].rolling(75).mean().shift(1)
        
        feat_5m["regime"] = feat_5m.apply(lambda r: classify_regime(r["trailing_adx"], r["trailing_vol"]), axis=1)

        regime_dist = feat_5m["regime"].value_counts()
        print(f"  Regime distribution:")
        for regime, count in regime_dist.items():
            print(f"    {regime:10s}: {count:4d} bars ({count/len(feat_5m):.1%})")

        ensemble = EnsemblePredictor.load(MODEL_DIR, symbol=symbol)
        signals = generate_ensemble_signals(feat_5m, feat_15m, ensemble)

        for direction in ["UP", "DOWN"]:
            dir_signals = signals[signals["direction"] == direction].reset_index(drop=True)
            if len(dir_signals) == 0:
                continue

            dir_signals["win"] = label_signal_outcomes(dir_signals, feat_5m)
            dir_signals = build_meta_features(dir_signals, feat_5m, symbol)
            
            # Merge trailing regime tags
            if "regime" in dir_signals.columns:
                dir_signals = dir_signals.drop(columns=["regime"])
            dir_signals = dir_signals.merge(feat_5m[["timestamp", "regime"]], on="timestamp", how="left")

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

            # Standard Simulation
            t_fixed, p_fixed = simulate_portfolio(approved, feat_5m)
            exp_fixed = t_fixed["pnl_pts"].mean() if len(t_fixed) > 0 else 0
            
            # Regime-Sized Simulation
            approved["risk_scalar"] = approved["regime"].map(REGIME_SCALARS).fillna(1.0)
            t_reg, p_reg = simulate_portfolio(approved, feat_5m)
            exp_reg = t_reg["pnl_pts"].mean() if len(t_reg) > 0 else 0
            
            pnl_fixed = p_fixed["capital"].iloc[-1] - p_fixed["capital"].iloc[0] if len(p_fixed) > 0 else 0
            pnl_reg = p_reg["capital"].iloc[-1] - p_reg["capital"].iloc[0] if len(p_reg) > 0 else 0
            
            improvement = (pnl_reg - pnl_fixed) / abs(pnl_fixed) * 100 if pnl_fixed != 0 else 0

            print(f"  {direction:5s}: Base PnL = {pnl_fixed:,.0f} | Regime PnL = {pnl_reg:,.0f} ({improvement:+.1f}%)")

            all_results.append({
                "Symbol": symbol,
                "Direction": direction,
                "Trades": len(t_reg),
                "Base Exp (pts)": round(exp_fixed, 1),
                "Regime Exp (pts)": round(exp_reg, 1),
                "Base Net PnL": round(pnl_fixed, 1),
                "Regime Net PnL": round(pnl_reg, 1),
                "PnL Improv": f"{improvement:+.1f}%"
            })

    if not all_results:
        print("\n❌ No results.")
        return

    df = pd.DataFrame(all_results)
    print("\n" + "=" * 80)
    print("REGIME SIZING PERFORMANCE METRICS")
    print("=" * 80)
    print(df.to_string(index=False))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "regime_analysis.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()

