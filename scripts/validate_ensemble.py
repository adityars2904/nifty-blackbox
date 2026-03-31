#!/usr/bin/env python3
"""
Validate Ensemble Predictor Performance
=========================================
Tests the 5m + 15m ensemble against validation data.
Compares ensemble vs individual model performance.

Usage:
    cd /path/to/PYTHON_FinProj
    python scripts/validate_ensemble.py --symbol NIFTY
    python scripts/validate_ensemble.py --symbol BANKNIFTY
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.metrics import classification_report, precision_score, recall_score

from backend.ml.data_loader import load_validation_data
from backend.ml.feature_engineering import add_features, MOVEMENT_FEATURES
from backend.ml.labels import label_movement
from backend.ml.ensemble_predictor import EnsemblePredictor


# ============================================================================
# HELPERS - explicit numpy conversions to fix all Pylance type errors
# ============================================================================

def _to_int(x: object) -> npt.NDArray[np.int_]:
    if isinstance(x, pd.Series):
        return x.to_numpy().astype(int)
    return np.asarray(x, dtype=int)


def _to_float(x: object) -> npt.NDArray[np.float64]:
    if isinstance(x, pd.Series):
        return x.to_numpy().astype(np.float64)
    return np.asarray(x, dtype=np.float64)


def evaluate_model(
    name: str,
    preds: npt.NDArray[np.int_],
    y_true: npt.NDArray[np.int_],
) -> dict:
    preds_np  = np.asarray(preds,  dtype=int)
    y_true_np = np.asarray(y_true, dtype=int)

    accuracy = float((preds_np == y_true_np).mean())

    prec_arr = np.asarray(
        precision_score(y_true_np, preds_np, average=None, zero_division=0),
        dtype=np.float64,
    )
    rec_arr = np.asarray(
        recall_score(y_true_np, preds_np, average=None, zero_division=0),
        dtype=np.float64,
    )

    up_prec   = float(prec_arr[2]) if len(prec_arr) > 2 else 0.0
    down_prec = float(prec_arr[0]) if len(prec_arr) > 0 else 0.0
    up_rec    = float(rec_arr[2])  if len(rec_arr)  > 2 else 0.0
    down_rec  = float(rec_arr[0])  if len(rec_arr)  > 0 else 0.0

    dir_mask     = (preds_np != 1) & (y_true_np != 1)
    n_dir        = int(dir_mask.sum())
    dir_accuracy = float((preds_np[dir_mask] == y_true_np[dir_mask]).mean()) if n_dir > 0 else 0.0
    up_bias      = float((preds_np == 2).mean())

    return {
        "name":           name,
        "accuracy":       accuracy,
        "up_precision":   up_prec,
        "down_precision": down_prec,
        "up_recall":      up_rec,
        "down_recall":    down_rec,
        "dir_accuracy":   dir_accuracy,
        "up_bias":        up_bias,
        "n_trades":       n_dir,
    }


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Ensemble Models")
    parser.add_argument(
        "--symbol", 
        type=str, 
        default="NIFTY", 
        choices=["NIFTY", "BANKNIFTY"],
        help="Trading symbol to validate"
    )
    args = parser.parse_args()

    print("=" * 70)
    print(f"ENSEMBLE PREDICTOR VALIDATION - {args.symbol}")
    print("=" * 70)
    print()

    # STEP 1: Load validation data
    print("STEP 1: Loading validation data...")
    val_5m  = load_validation_data(symbol=args.symbol, timeframe="5m")
    val_15m = load_validation_data(symbol=args.symbol, timeframe="15m")
    print(f"  5m:  {len(val_5m):,} candles")
    print(f"  15m: {len(val_15m):,} candles")
    print()

    # STEP 2: Features and labels
    print("STEP 2: Generating features and labels...")

    val_5m_feat  = add_features(val_5m).copy()
    val_5m_feat["label"] = label_movement(val_5m_feat)
    val_5m_clean = val_5m_feat.dropna(subset=MOVEMENT_FEATURES + ["label"]).copy()
    # We keep X_5m for length checking, but prediction uses dynamic features below
    X_5m  = val_5m_clean[MOVEMENT_FEATURES].astype(float)
    y_5m  = _to_int(val_5m_clean["label"])

    val_15m_feat  = add_features(val_15m).copy()
    val_15m_feat["label"] = label_movement(val_15m_feat)
    val_15m_clean = val_15m_feat.dropna(subset=MOVEMENT_FEATURES + ["label"]).copy()
    # We keep X_15m for length checking, but prediction uses dynamic features below
    X_15m  = val_15m_clean[MOVEMENT_FEATURES].astype(float)
    y_15m  = _to_int(val_15m_clean["label"])

    print(f"  5m clean:  {len(X_5m):,} samples")
    print(f"  15m clean: {len(X_15m):,} samples")
    print()

    # STEP 3: Load ensemble
    print("STEP 3: Loading ensemble predictor...")
    try:
        ensemble = EnsemblePredictor.load("backend/models", symbol=args.symbol)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print(f"   Make sure you have trained models for {args.symbol} first!")
        print(f"   Run: python scripts/train_foundation_models.py --symbol {args.symbol} --timeframe 5m")
        return

    print()

    # STEP 4: Predictions
    print("STEP 4: Generating predictions...")

    # 🔧 FIX: Extract the exact feature names the ensemble models were trained on.
    ens_features_5m = ensemble.model_5m.get_booster().feature_names
    ens_features_15m = ensemble.model_15m.get_booster().feature_names

    probs_5m  = np.asarray(ensemble.model_5m.predict_proba(val_5m_clean[ens_features_5m].astype(float)),   dtype=np.float64)
    probs_15m = np.asarray(ensemble.model_15m.predict_proba(val_15m_clean[ens_features_15m].astype(float)), dtype=np.float64)

    preds_5m_all  = np.argmax(probs_5m,  axis=1).astype(int)
    preds_15m_all = np.argmax(probs_15m, axis=1).astype(int)

    val_5m_clean["prob_down_5m"]    = probs_5m[:, 0]
    val_5m_clean["prob_neutral_5m"] = probs_5m[:, 1]
    val_5m_clean["prob_up_5m"]      = probs_5m[:, 2]

    val_15m_clean["prob_down_15m"]    = probs_15m[:, 0]
    val_15m_clean["prob_neutral_15m"] = probs_15m[:, 1]
    val_15m_clean["prob_up_15m"]      = probs_15m[:, 2]
    val_15m_clean["pred_15m_dir"]     = preds_15m_all

    val_5m_clean  = val_5m_clean.sort_values("timestamp").reset_index(drop=True)
    val_15m_clean = val_15m_clean.sort_values("timestamp").reset_index(drop=True)

    merged: pd.DataFrame = pd.merge_asof(
        val_5m_clean,
        val_15m_clean[[
            "timestamp",
            "prob_down_15m", "prob_neutral_15m", "prob_up_15m",
            "pred_15m_dir",
        ]],
        on="timestamp",
        direction="backward",
    ).dropna(subset=["prob_down_15m"])

    w5  = float(ensemble.weight_5m)
    w15 = float(ensemble.weight_15m)

    merged["ens_prob_down"]    = merged["prob_down_5m"]    * w5 + merged["prob_down_15m"]    * w15
    merged["ens_prob_neutral"] = merged["prob_neutral_5m"] * w5 + merged["prob_neutral_15m"] * w15
    merged["ens_prob_up"]      = merged["prob_up_5m"]      * w5 + merged["prob_up_15m"]      * w15

    ens_probs        = np.asarray(merged[["ens_prob_down", "ens_prob_neutral", "ens_prob_up"]].values, dtype=np.float64)
    probs_5m_mat     = np.asarray(merged[["prob_down_5m", "prob_neutral_5m", "prob_up_5m"]].values,   dtype=np.float64)

    preds_ens        = np.argmax(ens_probs,    axis=1).astype(int)
    conf_ens         = np.max(ens_probs,       axis=1).astype(np.float64)
    preds_5m_aligned = np.argmax(probs_5m_mat, axis=1).astype(int)
    pred_15m_aligned = np.asarray(merged["pred_15m_dir"].values, dtype=int)
    y_ens            = np.asarray(merged["label"].values,        dtype=int)

    print(f"  Aligned samples for ensemble: {len(merged):,}")
    print()

    # STEP 5: Compare performance
    print("=" * 70)
    print("RESULTS COMPARISON")
    print("=" * 70)
    print()

    results = [
        evaluate_model("5m alone",        preds_5m_aligned, y_ens),
        evaluate_model("15m alone",        pred_15m_aligned, y_ens),
        evaluate_model("Ensemble (30/70)", preds_ens,        y_ens),
    ]

    print(f"{'Metric':<28} {'5m alone':>10} {'15m alone':>10} {'Ensemble':>10}")
    print("-" * 62)
    for key, label in [
        ("accuracy",       "Overall Accuracy"),
        ("up_precision",   "UP Precision"),
        ("down_precision", "DOWN Precision"),
        ("up_recall",      "UP Recall"),
        ("down_recall",    "DOWN Recall"),
        ("dir_accuracy",   "Directional Accuracy"),
        ("up_bias",        "UP Prediction Rate"),
    ]:
        v0 = float(results[0][key])
        v1 = float(results[1][key])
        v2 = float(results[2][key])
        print(f"{label:<28} {v0:>10.3f} {v1:>10.3f} {v2:>10.3f}")

    print(f"{'Directional Trades':<28} {results[0]['n_trades']:>10,} {results[1]['n_trades']:>10,} {results[2]['n_trades']:>10,}")
    print()

    # STEP 6: Veto analysis
    print("=" * 70)
    print("VETO ANALYSIS")
    print("=" * 70)
    print()

    veto_15m_neutral  = pred_15m_aligned == 1
    veto_low_conf     = conf_ens < float(ensemble.min_confidence)
    veto_disagreement = (
        (preds_5m_aligned != 1)
        & (pred_15m_aligned != 1)
        & (preds_5m_aligned != pred_15m_aligned)
    )
    trade_mask = (
        ~veto_15m_neutral
        & ~veto_low_conf
        & ~veto_disagreement
        & (preds_ens != 1)
    )

    total = len(preds_ens)
    print(f"Total samples:             {total:>8,}")
    print(f"Veto - 15m NEUTRAL:        {int(veto_15m_neutral.sum()):>8,} ({float(veto_15m_neutral.mean())*100:.1f}%)")
    print(f"Veto - Low confidence:     {int(veto_low_conf.sum()):>8,} ({float(veto_low_conf.mean())*100:.1f}%)")
    print(f"Veto - 5m/15m disagree:    {int(veto_disagreement.sum()):>8,} ({float(veto_disagreement.mean())*100:.1f}%)")
    print(f"Signals that pass veto:    {int(trade_mask.sum()):>8,} ({float(trade_mask.mean())*100:.1f}%)")
    print()

    n_passed = int(trade_mask.sum())
    if n_passed > 0:
        preds_v = preds_ens[trade_mask]
        y_v     = y_ens[trade_mask]

        acc_v   = float((preds_v == y_v).mean())
        prec_v  = np.asarray(precision_score(y_v, preds_v, average=None, zero_division=0), dtype=np.float64)
        rec_v   = np.asarray(recall_score(y_v, preds_v, average=None, zero_division=0),    dtype=np.float64)

        print("Performance on signals that PASS all veto checks:")
        print()
        print(f"  Accuracy:       {acc_v:.4f} ({acc_v*100:.2f}%)")
        if len(prec_v) > 2:
            up_p  = float(prec_v[2])
            dn_p  = float(prec_v[0])
            up_r  = float(rec_v[2])
            dn_r  = float(rec_v[0])
            avg_p = (up_p + dn_p) / 2.0
            print(f"  UP Precision:   {up_p:.4f}")
            print(f"  DOWN Precision: {dn_p:.4f}")
            print(f"  UP Recall:      {up_r:.4f}")
            print(f"  DOWN Recall:    {dn_r:.4f}")
            print(f"  Avg Directional Precision: {avg_p:.4f} ({avg_p*100:.2f}%)")
    print()

    # STEP 7: Full classification report
    print("=" * 70)
    print("FULL ENSEMBLE CLASSIFICATION REPORT")
    print("=" * 70)
    print()
    print(classification_report(
        y_ens, preds_ens,
        target_names=["DOWN", "NEUTRAL", "UP"],
        digits=3,
    ))

    # STEP 8: Recommendation
    print("=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)
    print()

    avg_dir_prec = (results[2]["up_precision"] + results[2]["down_precision"]) / 2.0
    best_model   = max(results, key=lambda x: (x["up_precision"] + x["down_precision"]) / 2.0)
    best_prec    = (best_model["up_precision"] + best_model["down_precision"]) / 2.0

    if avg_dir_prec >= 0.42:
        print(f"✅ ENSEMBLE READY FOR META-LABELING!")
        print(f"   Average directional precision: {avg_dir_prec:.3f}")
        print()
        print("   Next step: python scripts/train_meta_filter.py")

    elif avg_dir_prec >= 0.38:
        print(f"🟡 ENSEMBLE IS MARGINAL ({avg_dir_prec:.3f})")
        print(f"   Best individual model: {best_model['name']} ({best_prec:.3f})")
        print()
        print("   Options:")
        print("   1. Use 15m alone as primary model")
        print("   2. Retrain 5m with purged features and re-validate")
        print("   3. Accept 38-40% and proceed to meta-filter anyway")

    else:
        print(f"❌ ENSEMBLE UNDERPERFORMING ({avg_dir_prec:.3f})")
        print()
        print("   Steps to fix:")
        print("   1. cp feature_engineering_PURGED.py backend/ml/feature_engineering.py")
        print("   2. python scripts/train_foundation_models.py --symbol {args.symbol} --timeframe 5m")
        print("   3. python scripts/validate_ensemble.py --symbol {args.symbol}")

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()