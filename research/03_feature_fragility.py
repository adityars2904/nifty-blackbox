#!/usr/bin/env python3
"""
03_feature_fragility.py
========================
Test the consecutive_wins dependency. Train each directional model with and without
this feature on 2024 training data, then measure the net Expectancy (PnL) drop visually
via full simulation path on 2025 vault data.

Usage:
    cd research && python 03_feature_fragility.py
"""

from __future__ import annotations
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.ml.data_loader import fetch_candles
from backend.ml.feature_engineering import add_features, MOVEMENT_FEATURES
from backend.ml.labels import label_movement
from backend.ml.ensemble_predictor import EnsemblePredictor
from scripts.validate_meta_filter_2025 import generate_ensemble_signals, label_signal_outcomes, build_meta_features
from research.core_execution import simulate_portfolio, THRESHOLDS

MODEL_DIR = str(project_root / "backend" / "models")
OUTPUT_DIR = Path(__file__).parent / "outputs"

TRAIN_START = "2024-05-01 09:15:00"
TRAIN_END   = "2024-12-31 15:30:00"
VAULT_START = "2025-01-01 09:15:00"
VAULT_END   = "2025-12-10 15:30:00"


def simulate_model_path(X_train, y_train, X_vault, vault_signals, vault_feat_5m, threshold):
    """Trains xgboost on X_train, y_train. Test on X_vault using simulate_portfolio."""
    model = xgb.XGBClassifier(
        objective="binary:logistic", eval_metric="logloss",
        max_depth=4, learning_rate=0.05, n_estimators=200,
        random_state=42, use_label_encoder=False
    )
    model.fit(X_train, y_train, verbose=False)
    
    probs = model.predict_proba(X_vault)[:, 1]
    approved = vault_signals[probs >= threshold].copy()
    
    if len(approved) == 0:
        return 0.0, 0
        
    t_df, _ = simulate_portfolio(approved, vault_feat_5m)
    if len(t_df) == 0:
        return 0.0, 0
        
    exp = float(t_df["pnl_pts"].mean())
    return exp, len(t_df)

def main():
    print("=" * 70)
    print("FEATURE FRAGILITY — EXPECTANCY DROP ANALYSIS")
    print("=" * 70)
    print()

    results = []

    for symbol in ["NIFTY", "BANKNIFTY"]:
        print(f"\nLoading {symbol} Train & Vault data...")
        raw_train_5m  = fetch_candles(symbol, timeframe="5m",  start_date=TRAIN_START, end_date=TRAIN_END)
        raw_train_15m = fetch_candles(symbol, timeframe="15m", start_date=TRAIN_START, end_date=TRAIN_END)
        raw_vault_5m  = fetch_candles(symbol, timeframe="5m",  start_date=VAULT_START, end_date=VAULT_END)
        raw_vault_15m = fetch_candles(symbol, timeframe="15m", start_date=VAULT_START, end_date=VAULT_END)

        if len(raw_train_5m) == 0 or len(raw_vault_5m) == 0:
            print(f"  ❌ Missing data for {symbol}, skipping")
            continue

        train_5m = add_features(raw_train_5m).copy()
        train_5m["label"] = label_movement(train_5m)
        train_5m = train_5m.dropna(subset=MOVEMENT_FEATURES + ["label", "atr"]).reset_index(drop=True)
        train_15m = add_features(raw_train_15m).dropna(subset=MOVEMENT_FEATURES).reset_index(drop=True)

        vault_5m = add_features(raw_vault_5m).copy()
        vault_5m["label"] = label_movement(vault_5m)
        vault_5m = vault_5m.dropna(subset=MOVEMENT_FEATURES + ["label", "atr"]).reset_index(drop=True)
        vault_15m = add_features(raw_vault_15m).dropna(subset=MOVEMENT_FEATURES).reset_index(drop=True)

        ensemble = EnsemblePredictor.load(MODEL_DIR, symbol=symbol)
        
        train_sigs = generate_ensemble_signals(train_5m, train_15m, ensemble)
        vault_sigs = generate_ensemble_signals(vault_5m, vault_15m, ensemble)

        for direction in ["UP", "DOWN"]:
            train_dir = train_sigs[train_sigs["direction"] == direction].reset_index(drop=True)
            vault_dir = vault_sigs[vault_sigs["direction"] == direction].reset_index(drop=True)
            
            if len(train_dir) < 50 or len(vault_dir) < 10:
                print(f"  ⚠️  {symbol} {direction}: Not enough signals, skipping")
                continue

            train_dir["win"] = label_signal_outcomes(train_dir, train_5m)
            train_dir = build_meta_features(train_dir, train_5m, symbol)
            
            vault_dir["win"] = label_signal_outcomes(vault_dir, vault_5m)
            vault_dir = build_meta_features(vault_dir, vault_5m, symbol)

            # Look up features from the reference joblib
            meta_path = project_root / "backend" / "models" / f"meta_filter_{symbol.lower()}_{direction.lower()}.joblib"
            if not meta_path.exists():
                meta_path = project_root / "backend" / "models" / f"meta_filter_ensemble_{symbol.lower()}.joblib"

            if not meta_path.exists():
                print(f"  ⚠️  {meta_path.name} not found, skipping")
                continue

            loaded = joblib.load(meta_path)
            ref_model = loaded["model"] if isinstance(loaded, dict) and "model" in loaded else loaded
            full_feats = list(ref_model.get_booster().feature_names)
            
            if "consecutive_wins" not in full_feats:
                # If the meta-filter doesn't have it, skip
                continue

            threshold = THRESHOLDS.get((symbol, direction), 0.55)

            y_train = train_dir["win"].astype(int)
            
            # FULL MODEL
            X_tr_full = train_dir[full_feats].astype(float)
            X_va_full = vault_dir[full_feats].astype(float)
            full_exp, full_t = simulate_model_path(X_tr_full, y_train, X_va_full, vault_dir, vault_5m, threshold)

            # REDUCED MODEL
            red_feats = [f for f in full_feats if f != "consecutive_wins"]
            X_tr_red = train_dir[red_feats].astype(float)
            X_va_red = vault_dir[red_feats].astype(float)
            red_exp, red_t = simulate_model_path(X_tr_red, y_train, X_va_red, vault_dir, vault_5m, threshold)

            drop_pts = full_exp - red_exp

            if drop_pts > 5.0 and red_exp < 0:
                verdict = "CRITICALLY FRAGILE"
            elif drop_pts > 2.0:
                verdict = "FRAGILE"
            elif drop_pts < -2.0:
                verdict = "BETTER WITHOUT"
            else:
                verdict = "ROBUST"

            print(f"  {symbol:10s} {direction:5s} | Full Exp: {full_exp:+.1f} | No CW: {red_exp:+.1f} | "
                  f"Drop: {drop_pts:+.1f} pts → {verdict}")

            results.append({
                "Model": f"{symbol} {direction}",
                "Full Exp (pts)": round(full_exp, 1),
                "No-CW Exp (pts)": round(red_exp, 1),
                "Drop (pts)": round(drop_pts, 1),
                "Trades Full/Red": f"{full_t}/{red_t}",
                "Verdict": verdict
            })

    if not results:
        print("\n❌ No models dependent on consecutive_wins found.")
        return

    df = pd.DataFrame(results)
    print("\n" + "=" * 70)
    print("FRAGILITY TABLE — Expectancy Drop")
    print("=" * 70)
    print(df.to_string(index=False))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "fragility_results_expectancy.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}\n")

if __name__ == "__main__":
    main()

