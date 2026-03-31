#!/usr/bin/env python3
"""
09_label_redesign_comparison.py
================================
Tests whether path-dependent labels produce better-calibrated base models
than the current point-in-time label, and whether the improvement propagates
through the full pipeline to better vault meta-filter win rates.

THREE VARIANTS TRAINED AND COMPARED
-------------------------------------
  CURRENT  — Point-in-time 3-class label (existing production design)
             UP   = future_close[+6] > close + 0.5×ATR
             DOWN = future_close[+6] < close - 0.5×ATR
             NEUTRAL = otherwise

  VARIANT_A — Path-dependent binary label (fixes target mismatch)
             WIN  = TP (1.5×ATR) hit before SL (1.0×ATR) within 12 bars
             LOSS = SL hit first, or neither hit within 12 bars
             → Binary XGBoost, same 17 features

  VARIANT_B — Path-dependent 3-class label (keeps NEUTRAL concept)
             UP      = TP hit before SL within 12 bars AND direction was UP
             DOWN    = TP hit before SL within 12 bars AND direction was DOWN
             NEUTRAL = neither TP nor SL hit within 12 bars (time expiry)
             → 3-class XGBoost, same 17 features

WHAT WE MEASURE
----------------
For each variant we measure THREE things in sequence:

  1. BASE MODEL QUALITY
     - For CURRENT/VARIANT_B: 3-class accuracy on 2024 val period
     - For VARIANT_A: binary accuracy + AUC on 2024 val period
     - Key metric: directional precision (UP+DOWN only, excluding NEUTRAL)
     - Calibration: Brier score of P(correct direction) against actual outcome

  2. PROBABILITY CALIBRATION
     - For each base model, generate probabilities on 2025 vault candles
     - Bin probabilities into deciles and measure actual win rate per bin
     - A well-calibrated model should show monotonic win rate across bins
     - Current model's calibration against PATH-DEPENDENT outcomes
       (not its own training labels) is expected to be poor

  3. FULL PIPELINE META-FILTER WIN RATE
     - Run each base model through the FULL signal pipeline:
       ensemble veto → signal generation → meta-filter approval
     - Compare approved signal win rate on vault data
     - This is the ground truth metric that matters for deployment

All three variants use IDENTICAL:
  - 17 MOVEMENT_FEATURES (no feature changes)
  - Training period: 2022-04-11 to 2024-04-30
  - Validation period: 2024-05-01 to 2024-12-31
  - Judgment period: 2025-01-01 to 2025-12-10 (vault)
  - XGBoost hyperparameters (same as production)
  - Meta-filter models (existing production models — not retrained here)

IMPORTANT: This script DOES NOT overwrite any production model files.
Experimental models are saved to research/outputs/label_comparison_models/
The production models in backend/models/ are untouched throughout.

Usage:
    cd research && python 09_label_redesign_comparison.py

    # To run only one symbol (faster iteration):
    cd research && python 09_label_redesign_comparison.py --symbol NIFTY

    # To skip the expensive full pipeline meta-filter step:
    cd research && python 09_label_redesign_comparison.py --skip-pipeline
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.ml.data_loader import fetch_candles
from backend.ml.feature_engineering import add_features, MOVEMENT_FEATURES
from backend.ml.labels import label_movement
from backend.ml.ensemble_predictor import EnsemblePredictor, DIRECTION_MAP
from scripts.validate_meta_filter_2025 import (
    generate_ensemble_signals,
    build_meta_features,
    label_signal_outcomes,
)
from research.core_execution import simulate_portfolio, THRESHOLDS

OUTPUT_DIR   = Path(__file__).parent / "outputs"
MODEL_CACHE  = OUTPUT_DIR / "label_comparison_models"
PROD_MODELS  = project_root / "backend" / "models"

TRAIN_START = "2022-04-11 09:15:00"
TRAIN_END   = "2024-04-30 15:30:00"
VAL_START   = "2024-05-01 09:15:00"
VAL_END     = "2024-12-31 15:30:00"
VAULT_START = "2025-01-01 09:15:00"
VAULT_END   = "2025-12-10 15:30:00"

TP_MULT     = 1.5   # matches meta-filter evaluation exactly
SL_MULT     = 1.0   # matches meta-filter evaluation exactly
HORIZON     = 12    # bars — matches label_signal_outcomes() in validate script

DIRECTION_MAP_INV = {"DOWN": 0, "NEUTRAL": 1, "UP": 2}

XGB_BASE_PARAMS = dict(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    gamma=0.1,
    eval_metric="mlogloss",
    random_state=42,
    verbosity=0,
)


# ════════════════════════════════════════════════════════════════════════════
# LABEL FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════

def label_current(df: pd.DataFrame) -> pd.Series:
    """
    Production label: point-in-time 3-class.
    Thin wrapper around the existing label_movement() function
    (horizon=6, atr_mult=0.5) for explicit comparison.
    """
    return label_movement(df, horizon=6, atr_mult=0.5)


def label_path_binary(df: pd.DataFrame) -> pd.Series:
    """
    VARIANT_A: Path-dependent binary label.
    WIN=1 if TP (1.5×ATR) is hit before SL (1.0×ATR) within HORIZON bars.
    LOSS=0 otherwise (SL hit first, or time expiry).
    Labels the direction implicitly: for each candle we check both UP and
    DOWN paths — label is assigned based on whichever direction would result
    in a WIN. If both or neither would win, label is LOSS(0).

    For base model training we need to know whether this candle's price
    action resolves as a win for the direction the ensemble would predict.
    Since we don't know ensemble direction at label time, we label
    conservatively: WIN=1 only if the price moves cleanly in one direction
    (TP hit before SL) without ambiguity.
    """
    close = df["close"].astype(float).values
    high  = df["high"].astype(float).values
    low   = df["low"].astype(float).values
    atr   = df["atr"].astype(float).values if "atr" in df.columns else \
        np.full(len(df), 10.0)

    labels = np.zeros(len(df), dtype=float)

    for i in range(len(df) - HORIZON):
        if np.isnan(atr[i]) or atr[i] <= 0:
            labels[i] = np.nan
            continue

        entry    = close[i]
        tp_up    = entry + TP_MULT * atr[i]
        sl_up    = entry - SL_MULT * atr[i]
        tp_down  = entry - TP_MULT * atr[i]
        sl_down  = entry + SL_MULT * atr[i]

        won_up   = False
        won_down = False

        for fwd in range(i + 1, min(i + 1 + HORIZON, len(df))):
            h, l = high[fwd], low[fwd]
            # UP path
            if not won_up:
                if h >= tp_up and l > sl_up:
                    won_up = True
                elif l <= sl_up:
                    pass  # UP path lost — don't mark won_up
            # DOWN path
            if not won_down:
                if l <= tp_down and h < sl_down:
                    won_down = True
                elif h >= sl_down:
                    pass  # DOWN path lost

        # WIN = exactly one direction wins cleanly
        if won_up and not won_down:
            labels[i] = 1.0
        elif won_down and not won_up:
            labels[i] = 1.0
        else:
            labels[i] = 0.0

    # Last HORIZON bars cannot be labeled
    labels[len(df) - HORIZON:] = np.nan
    return pd.Series(labels, index=df.index)


def label_path_3class(df: pd.DataFrame) -> pd.Series:
    """
    VARIANT_B: Path-dependent 3-class label.
    UP      = TP hit on UP path before SL, SL on DOWN path not triggered first
    DOWN    = TP hit on DOWN path before SL, SL on UP path not triggered first
    NEUTRAL = Neither TP nor SL hit within HORIZON bars (time expiry)

    Key difference from current: UP/DOWN assignment is path-dependent, not
    point-in-time. A candle labeled UP means buying here would win before
    losing. NEUTRAL means the market went nowhere decisive.
    """
    close = df["close"].astype(float).values
    high  = df["high"].astype(float).values
    low   = df["low"].astype(float).values
    atr   = df["atr"].astype(float).values if "atr" in df.columns else \
        np.full(len(df), 10.0)

    labels = np.full(len(df), np.nan)

    for i in range(len(df) - HORIZON):
        if np.isnan(atr[i]) or atr[i] <= 0:
            continue

        entry   = close[i]
        tp_up   = entry + TP_MULT * atr[i]
        sl_up   = entry - SL_MULT * atr[i]
        tp_down = entry - TP_MULT * atr[i]
        sl_down = entry + SL_MULT * atr[i]

        up_result   = None   # "win" or "loss"
        down_result = None

        for fwd in range(i + 1, min(i + 1 + HORIZON, len(df))):
            h, l = high[fwd], low[fwd]

            # Resolve UP path if not yet decided
            if up_result is None:
                if h >= tp_up and l <= sl_up:
                    up_result = "loss"   # same-bar collision → conservative = loss
                elif h >= tp_up:
                    up_result = "win"
                elif l <= sl_up:
                    up_result = "loss"

            # Resolve DOWN path if not yet decided
            if down_result is None:
                if l <= tp_down and h >= sl_down:
                    down_result = "loss"  # same-bar collision → conservative = loss
                elif l <= tp_down:
                    down_result = "win"
                elif h >= sl_down:
                    down_result = "loss"

            if up_result is not None and down_result is not None:
                break

        # Time expiry: unresolved path = loss (never reached TP in time)
        if up_result is None:
            up_result = "loss"
        if down_result is None:
            down_result = "loss"

        # Assign class
        if up_result == "win" and down_result == "loss":
            labels[i] = 2.0   # UP
        elif down_result == "win" and up_result == "loss":
            labels[i] = 0.0   # DOWN
        elif up_result == "loss" and down_result == "loss":
            labels[i] = 1.0   # NEUTRAL (time expiry or both lose)
        else:
            # Both win simultaneously (rare, volatile candle)
            labels[i] = 1.0   # NEUTRAL — ambiguous, don't trade

    labels[len(df) - HORIZON:] = np.nan
    return pd.Series(labels, index=df.index)


# ════════════════════════════════════════════════════════════════════════════
# MODEL TRAINING
# ════════════════════════════════════════════════════════════════════════════

def train_base_model(
    X: pd.DataFrame,
    y: pd.Series,
    variant: str,
    n_splits: int = 5,
    verbose: bool = True,
) -> tuple[XGBClassifier, dict]:
    """
    Train a base model with walk-forward CV.
    variant: "current" | "binary" | "3class"
    Returns (final_model, cv_metrics).
    """
    is_binary  = variant == "binary"
    n_classes  = 2 if is_binary else 3

    objective  = "binary:logistic" if is_binary else "multi:softprob"
    params     = {**XGB_BASE_PARAMS, "objective": objective}
    if not is_binary:
        params["num_class"] = n_classes

    classes    = np.unique(y)
    cw         = compute_class_weight("balanced", classes=classes, y=y)
    w_dict     = {int(c): float(w) for c, w in zip(classes, cw)}
    weights    = np.array([w_dict[int(v)] for v in y])

    tscv       = TimeSeriesSplit(n_splits=n_splits)
    fold_accs  = []
    fold_aucs  = []

    for fold, (tr_idx, te_idx) in enumerate(tscv.split(X), 1):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]
        w_tr       = weights[tr_idx]

        m = XGBClassifier(**params)
        m.fit(X_tr, y_tr, sample_weight=w_tr)

        preds = m.predict(X_te)
        acc   = float(accuracy_score(y_te, preds))
        fold_accs.append(acc)

        probs = m.predict_proba(X_te)
        if is_binary:
            auc = float(roc_auc_score(y_te, probs[:, 1]))
        else:
            try:
                auc = float(roc_auc_score(
                    y_te, probs, multi_class="ovr", average="macro"
                ))
            except Exception:
                auc = float("nan")
        fold_aucs.append(auc)

        if verbose:
            print(f"      Fold {fold}: acc={acc:.4f}  auc={auc:.4f}  "
                  f"[{len(tr_idx)}/{len(te_idx)}]")

    cv_metrics = {
        "cv_acc_mean": float(np.mean(fold_accs)),
        "cv_acc_std":  float(np.std(fold_accs)),
        "cv_auc_mean": float(np.nanmean(fold_aucs)),
    }
    if verbose:
        print(f"      CV mean: acc={cv_metrics['cv_acc_mean']:.4f}  "
              f"auc={cv_metrics['cv_auc_mean']:.4f}")

    # Final model on all training data
    final = XGBClassifier(**params)
    final.fit(X, y, sample_weight=weights)
    return final, cv_metrics


# ════════════════════════════════════════════════════════════════════════════
# CALIBRATION MEASUREMENT
# ════════════════════════════════════════════════════════════════════════════

def measure_calibration(
    model: XGBClassifier,
    feat_df: pd.DataFrame,
    variant: str,
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Bin model's directional confidence into deciles.
    For each bin, measure actual path-dependent win rate.
    A well-calibrated model shows monotonically increasing win rate.

    Returns DataFrame with columns: bin_center, mean_confidence,
    actual_win_rate, n_samples.
    """
    is_binary = variant == "binary"

    # Features
    X = feat_df[MOVEMENT_FEATURES].astype(float)
    probs = model.predict_proba(X)

    # Path-dependent binary outcome for calibration judgment
    # (always uses the same standard regardless of training label)
    path_labels = label_path_binary(feat_df)
    valid = path_labels.notna()

    if is_binary:
        confidence = probs[:, 1]   # P(WIN)
        outcome    = path_labels.fillna(0).astype(float).values
    else:
        # For 3-class: directional confidence = max(P(UP), P(DOWN))
        # outcome = 1 if the highest-confidence direction actually won
        pred_dirs  = np.argmax(probs, axis=1)   # 0=DOWN,1=NEUTRAL,2=UP
        confidence = np.max(probs, axis=1)

        # Map: did path-dependent label match predicted direction?
        path_3class = label_path_3class(feat_df).fillna(1)
        outcome = (pred_dirs == path_3class.values).astype(float)

    # Bin by confidence
    conf_valid = confidence[valid]
    out_valid  = outcome[valid]

    bins = np.percentile(conf_valid, np.linspace(0, 100, n_bins + 1))
    bins = np.unique(bins)  # remove duplicates at extremes

    rows = []
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf_valid >= lo) & (conf_valid < hi)
        if mask.sum() == 0:
            continue
        rows.append({
            "bin_center":      round((lo + hi) / 2, 3),
            "mean_confidence": round(float(conf_valid[mask].mean()), 3),
            "actual_win_rate": round(float(out_valid[mask].mean()), 3),
            "n_samples":       int(mask.sum()),
        })

    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════════════
# FULL PIPELINE: substitute model → run ensemble → meta-filter → vault win rate
# ════════════════════════════════════════════════════════════════════════════

def run_full_pipeline(
    symbol: str,
    model_5m: XGBClassifier,
    model_15m: XGBClassifier,
    feat_vault_5m: pd.DataFrame,
    feat_vault_15m: pd.DataFrame,
    variant_label: str,
) -> dict:
    """
    Substitute experimental base models into the ensemble, run signal
    generation and meta-filter approval on vault data, return win rate metrics.
    Uses production meta-filter models unchanged.
    """
    # Build a temporary EnsemblePredictor with our experimental models
    from backend.ml.ensemble_predictor import EnsemblePredictor
    exp_ensemble = EnsemblePredictor(
        model_5m=model_5m,
        model_15m=model_15m,
    )

    # Run signal generation (mirrors validate_meta_filter_2025.py logic)
    try:
        signals = generate_ensemble_signals(
            feat_vault_5m, feat_vault_15m, exp_ensemble
        )
    except Exception as e:
        return {"error": str(e), "variant": variant_label, "symbol": symbol}

    if len(signals) == 0:
        return {
            "variant":          variant_label,
            "symbol":           symbol,
            "n_signals_base":   0,
            "n_approved":       0,
            "base_win_rate":    float("nan"),
            "approved_win_rate": float("nan"),
            "filter_edge_pp":   float("nan"),
            "signals_per_day":  float("nan"),
        }

    signals["win"] = label_signal_outcomes(signals, feat_vault_5m)
    signals = build_meta_features(signals, feat_vault_5m, symbol)

    trading_days = pd.to_datetime(
        feat_vault_5m["timestamp"]
    ).dt.date.nunique()

    results = {}
    approved_all = []

    for direction in ["UP", "DOWN"]:
        dir_sigs = signals[signals["direction"] == direction].reset_index(drop=True)
        if len(dir_sigs) == 0:
            continue

        threshold = THRESHOLDS.get((symbol, direction), 0.55)
        meta_path = PROD_MODELS / f"meta_filter_{symbol.lower()}_{direction.lower()}.joblib"
        if not meta_path.exists():
            meta_path = PROD_MODELS / f"meta_filter_ensemble_{symbol.lower()}.joblib"
        if not meta_path.exists():
            continue

        loaded     = joblib.load(meta_path)
        meta_model = loaded["model"] if isinstance(loaded, dict) else loaded
        feat_names = list(meta_model.get_booster().feature_names)

        missing = [f for f in feat_names if f not in dir_sigs.columns]
        if missing:
            continue

        probs    = meta_model.predict_proba(dir_sigs[feat_names].astype(float))[:, 1]
        approved = dir_sigs[probs >= threshold].copy()
        approved_all.append(approved)

    if not approved_all:
        return {
            "variant":           variant_label,
            "symbol":            symbol,
            "n_signals_base":    len(signals),
            "n_approved":        0,
            "base_win_rate":     round(float(signals["win"].mean()), 3),
            "approved_win_rate": float("nan"),
            "filter_edge_pp":    float("nan"),
            "signals_per_day":   0.0,
        }

    approved_df = pd.concat(approved_all).sort_values("timestamp").reset_index(drop=True)

    base_wr     = float(signals["win"].mean())
    approved_wr = float(approved_df["win"].mean())
    spd         = len(approved_df) / max(trading_days, 1)

    return {
        "variant":           variant_label,
        "symbol":            symbol,
        "n_signals_base":    len(signals),
        "n_approved":        len(approved_df),
        "base_win_rate":     round(base_wr,     3),
        "approved_win_rate": round(approved_wr, 3),
        "filter_edge_pp":    round((approved_wr - base_wr) * 100, 1),
        "signals_per_day":   round(spd, 1),
        "trading_days":      trading_days,
    }


# ════════════════════════════════════════════════════════════════════════════
# DIRECTIONAL PRECISION HELPER
# ════════════════════════════════════════════════════════════════════════════

def directional_precision(
    model: XGBClassifier, X: pd.DataFrame, y: pd.Series, variant: str
) -> dict:
    """
    For base model quality: measure precision on directional calls only.
    For binary model, precision = P(path wins | model predicts WIN).
    For 3-class, precision = P(direction correct | model predicts UP or DOWN).
    """
    probs = model.predict_proba(X)
    preds = model.predict(X)
    y_arr = y.values

    if variant == "binary":
        # Predicted WIN = high confidence
        high_conf = probs[:, 1] >= 0.50
        if high_conf.sum() == 0:
            return {"dir_precision": float("nan"), "n_directional": 0}
        prec = float((y_arr[high_conf] == 1).mean())
        return {"dir_precision": round(prec, 4), "n_directional": int(high_conf.sum())}
    else:
        # Directional = predicted UP(2) or DOWN(0)
        dir_mask = preds != 1
        if dir_mask.sum() == 0:
            return {"dir_precision": float("nan"), "n_directional": 0}
        prec = float((preds[dir_mask] == y_arr[dir_mask]).mean())
        return {"dir_precision": round(prec, 4), "n_directional": int(dir_mask.sum())}


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol",        default=None,
                        choices=["NIFTY", "BANKNIFTY"],
                        help="Run for one symbol only (default: both)")
    parser.add_argument("--skip-pipeline", action="store_true",
                        help="Skip full meta-filter pipeline step (faster)")
    args = parser.parse_args()

    symbols = [args.symbol] if args.symbol else ["NIFTY", "BANKNIFTY"]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_CACHE.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("LABEL REDESIGN COMPARISON — Base Model Training & Vault Evaluation")
    print(f"Train:  {TRAIN_START} → {TRAIN_END}")
    print(f"Val:    {VAL_START}   → {VAL_END}")
    print(f"Vault:  {VAULT_START} → {VAULT_END}")
    print(f"Labels: CURRENT (point-in-time) vs VARIANT_A (binary path-dep) "
          f"vs VARIANT_B (3-class path-dep)")
    print(f"Features: {len(MOVEMENT_FEATURES)} (unchanged — {MOVEMENT_FEATURES[:3]}…)")
    print("=" * 80)

    all_base_metrics  = []
    all_calib         = []
    all_pipeline      = []

    for symbol in symbols:
        print(f"\n{'═'*70}")
        print(f"  SYMBOL: {symbol}")
        print(f"{'═'*70}")

        # ── Load data ───────────────────────────────────────────────────────
        print(f"\n  Loading training data…")
        t0 = time.time()
        train_raw = fetch_candles(
            symbol, timeframe="5m",
            start_date=TRAIN_START, end_date=TRAIN_END,
        )
        val_raw   = fetch_candles(
            symbol, timeframe="5m",
            start_date=VAL_START, end_date=VAL_END,
        )
        print(f"  Train: {len(train_raw):,} candles  "
              f"Val: {len(val_raw):,} candles  "
              f"({time.time()-t0:.1f}s)")

        if len(train_raw) < 5000:
            print(f"  ❌ Insufficient training data for {symbol}, skipping")
            continue

        # Feature engineering (shared across all variants)
        print(f"  Computing features…")
        train_feat = add_features(train_raw).copy()
        val_feat   = add_features(val_raw).copy()

        # Ensure ATR is available (needed by path-dependent labellers)
        for df in [train_feat, val_feat]:
            if "atr" not in df.columns:
                close = df["close"].astype(float)
                high  = df["high"].astype(float)
                low   = df["low"].astype(float)
                tr = pd.concat([
                    high - low,
                    (high - close.shift(1)).abs(),
                    (low  - close.shift(1)).abs(),
                ], axis=1).max(axis=1)
                df["atr"] = tr.rolling(14).mean()

        # ── Generate labels ─────────────────────────────────────────────────
        print(f"  Generating labels (path-dependent labels take ~30s)…")
        t0 = time.time()

        train_feat["label_current"] = label_current(train_feat)
        val_feat["label_current"]   = label_current(val_feat)
        print(f"    CURRENT labels done ({time.time()-t0:.1f}s)")

        t0 = time.time()
        train_feat["label_binary"]  = label_path_binary(train_feat)
        val_feat["label_binary"]    = label_path_binary(val_feat)
        print(f"    VARIANT_A binary labels done ({time.time()-t0:.1f}s)")

        t0 = time.time()
        train_feat["label_3class"]  = label_path_3class(train_feat)
        val_feat["label_3class"]    = label_path_3class(val_feat)
        print(f"    VARIANT_B 3-class labels done ({time.time()-t0:.1f}s)")

        # Label distribution report
        for lname in ["label_current", "label_binary", "label_3class"]:
            dist = train_feat[lname].value_counts().sort_index()
            total = dist.sum()
            dist_str = "  ".join(
                f"{int(k)}={'NEUTRAL' if k==1 and lname!='label_binary' else ('WIN' if lname=='label_binary' and k==1 else 'class')}"
                f":{int(v)}({v/total*100:.0f}%)"
                for k, v in dist.items()
            )
            print(f"    {lname:18s}: {dist_str}")

        # ── Clean datasets per label ─────────────────────────────────────────
        required_feats = MOVEMENT_FEATURES

        def clean(df, label_col):
            return df.dropna(subset=required_feats + [label_col]).copy()

        tr_cur  = clean(train_feat, "label_current")
        tr_bin  = clean(train_feat, "label_binary")
        tr_3cl  = clean(train_feat, "label_3class")
        va_cur  = clean(val_feat,   "label_current")
        va_bin  = clean(val_feat,   "label_binary")
        va_3cl  = clean(val_feat,   "label_3class")

        X_tr_cur  = tr_cur[MOVEMENT_FEATURES].astype(float)
        X_tr_bin  = tr_bin[MOVEMENT_FEATURES].astype(float)
        X_tr_3cl  = tr_3cl[MOVEMENT_FEATURES].astype(float)
        X_va_cur  = va_cur[MOVEMENT_FEATURES].astype(float)
        X_va_bin  = va_bin[MOVEMENT_FEATURES].astype(float)
        X_va_3cl  = va_3cl[MOVEMENT_FEATURES].astype(float)

        y_tr_cur  = tr_cur["label_current"].astype(int)
        y_tr_bin  = tr_bin["label_binary"].astype(int)
        y_tr_3cl  = tr_3cl["label_3class"].astype(int)
        y_va_cur  = va_cur["label_current"].astype(int)
        y_va_bin  = va_bin["label_binary"].astype(int)
        y_va_3cl  = va_3cl["label_3class"].astype(int)

        print(f"\n  Effective training sizes: "
              f"current={len(X_tr_cur):,}  "
              f"binary={len(X_tr_bin):,}  "
              f"3class={len(X_tr_3cl):,}")

        # ── TRAIN all three variants ─────────────────────────────────────────
        trained_models = {}

        for variant, X_tr, y_tr, X_va, y_va, va_df in [
            ("current", X_tr_cur, y_tr_cur, X_va_cur, y_va_cur, va_cur),
            ("binary",  X_tr_bin, y_tr_bin, X_va_bin, y_va_bin, va_bin),
            ("3class",  X_tr_3cl, y_tr_3cl, X_va_3cl, y_va_3cl, va_3cl),
        ]:
            print(f"\n  ── Training VARIANT: {variant.upper()} ──")
            model, cv_m = train_base_model(X_tr, y_tr, variant, n_splits=5)

            # Val metrics
            val_preds = model.predict(X_va)
            val_acc   = float(accuracy_score(y_va, val_preds))
            dir_m     = directional_precision(model, X_va, y_va, variant)

            # Brier score (calibration quality against path-dependent outcome)
            val_probs = model.predict_proba(X_va)
            # Path-dep binary for calibration reference
            path_bin_va = label_path_binary(va_df).fillna(0)
            path_bin_va = path_bin_va.reindex(y_va.index).fillna(0)

            if variant == "binary":
                brier = float(brier_score_loss(
                    path_bin_va.astype(int), val_probs[:, 1]
                ))
            else:
                # Use max directional prob vs path-dep binary for cross-comparison
                max_dir_prob = np.max(val_probs[:, [0, 2]], axis=1)  # max(P_DOWN, P_UP)
                brier = float(brier_score_loss(
                    path_bin_va.astype(int), max_dir_prob
                ))

            row = {
                "symbol":         symbol,
                "variant":        variant,
                "train_samples":  len(X_tr),
                "val_samples":    len(X_va),
                "cv_acc_mean":    round(cv_m["cv_acc_mean"], 4),
                "cv_auc_mean":    round(cv_m["cv_auc_mean"], 4),
                "val_accuracy":   round(val_acc, 4),
                "dir_precision":  round(dir_m["dir_precision"], 4),
                "n_directional":  dir_m["n_directional"],
                "brier_vs_path":  round(brier, 4),
            }
            all_base_metrics.append(row)
            trained_models[variant] = model

            print(f"    val_acc={val_acc:.4f}  "
                  f"dir_precision={dir_m['dir_precision']:.4f}  "
                  f"brier(vs path)={brier:.4f}")

            # Save experimental model
            save_path = MODEL_CACHE / f"{symbol.lower()}_{variant}_5m.joblib"
            joblib.dump({"model": model, "variant": variant, "symbol": symbol,
                         "label_params": {"tp": TP_MULT, "sl": SL_MULT, "horizon": HORIZON}},
                        save_path)

        # ── CALIBRATION on vault period ─────────────────────────────────────
        print(f"\n  Loading vault data for calibration check…")
        vault_raw_5m = fetch_candles(
            symbol, timeframe="5m",
            start_date=VAULT_START, end_date=VAULT_END,
        )
        vault_feat_5m = add_features(vault_raw_5m).copy()
        if "atr" not in vault_feat_5m.columns:
            close = vault_feat_5m["close"].astype(float)
            high  = vault_feat_5m["high"].astype(float)
            low   = vault_feat_5m["low"].astype(float)
            tr    = pd.concat([
                high - low,
                (high - close.shift(1)).abs(),
                (low  - close.shift(1)).abs(),
            ], axis=1).max(axis=1)
            vault_feat_5m["atr"] = tr.rolling(14).mean()

        vault_feat_5m = vault_feat_5m.dropna(
            subset=MOVEMENT_FEATURES + ["atr"]
        ).reset_index(drop=True)

        for variant, model in trained_models.items():
            calib = measure_calibration(model, vault_feat_5m, variant)
            if len(calib) > 0:
                calib.insert(0, "symbol", symbol)
                calib.insert(1, "variant", variant)
                all_calib.append(calib)

                # Monotonicity score: fraction of adjacent bin pairs where
                # win_rate increases with confidence
                wrs = calib["actual_win_rate"].values
                if len(wrs) >= 3:
                    mono_pairs = sum(
                        1 for i in range(len(wrs) - 1) if wrs[i + 1] >= wrs[i]
                    )
                    mono_score = mono_pairs / (len(wrs) - 1)
                else:
                    mono_score = float("nan")

                print(f"    {variant:8s}: calibration monotonicity={mono_score:.2f} "
                      f"(1.0 = perfect, 0.5 = random)")

        # ── FULL PIPELINE (optional) ─────────────────────────────────────────
        if not args.skip_pipeline:
            print(f"\n  Running full pipeline comparison on vault…")
            vault_raw_15m = fetch_candles(
                symbol, timeframe="15m",
                start_date=VAULT_START, end_date=VAULT_END,
            )
            vault_feat_15m = add_features(vault_raw_15m).dropna(
                subset=MOVEMENT_FEATURES
            ).reset_index(drop=True)

            vault_feat_5m_labeled = vault_feat_5m.copy()
            vault_feat_5m_labeled["label"] = label_movement(vault_feat_5m_labeled)
            vault_feat_5m_labeled = vault_feat_5m_labeled.dropna(
                subset=MOVEMENT_FEATURES + ["label", "atr"]
            ).reset_index(drop=True)

            # CURRENT: use production model (ground truth comparison)
            print(f"    [CURRENT] Using production 5m model…")
            prod_ensemble = EnsemblePredictor.load(str(PROD_MODELS), symbol=symbol)
            pipeline_current = run_full_pipeline(
                symbol, prod_ensemble.model_5m, prod_ensemble.model_15m,
                vault_feat_5m_labeled, vault_feat_15m, "current_production"
            )
            all_pipeline.append(pipeline_current)
            print(f"    CURRENT:   approved={pipeline_current.get('n_approved', 0)}  "
                  f"win_rate={pipeline_current.get('approved_win_rate', 'n/a')}")

            # VARIANT_A and VARIANT_B: use experimental 5m model
            # (15m model unchanged — isolates the 5m label effect)
            for variant_key, variant_label in [
                ("binary", "variant_a_binary"),
                ("3class", "variant_b_3class"),
            ]:
                exp_model_5m = trained_models[variant_key]
                print(f"    [{variant_label}] Running pipeline…")
                result = run_full_pipeline(
                    symbol,
                    exp_model_5m,
                    prod_ensemble.model_15m,   # 15m unchanged
                    vault_feat_5m_labeled,
                    vault_feat_15m,
                    variant_label,
                )
                all_pipeline.append(result)
                print(f"    {variant_label}: "
                      f"approved={result.get('n_approved', 0)}  "
                      f"win_rate={result.get('approved_win_rate', 'n/a')}")

    # ── Save outputs ─────────────────────────────────────────────────────────
    if all_base_metrics:
        pd.DataFrame(all_base_metrics).to_csv(
            OUTPUT_DIR / "label_comparison_base_metrics.csv", index=False
        )
    if all_calib:
        pd.concat(all_calib, ignore_index=True).to_csv(
            OUTPUT_DIR / "label_comparison_calibration.csv", index=False
        )
    if all_pipeline:
        pd.DataFrame(all_pipeline).to_csv(
            OUTPUT_DIR / "label_comparison_pipeline.csv", index=False
        )

    # ── Final comparison table ────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("FINAL COMPARISON TABLE")
    print("=" * 80)

    if all_base_metrics:
        bm_df = pd.DataFrame(all_base_metrics)
        print("\n  BASE MODEL QUALITY (validation period 2024):")
        print(bm_df[[
            "symbol", "variant", "val_accuracy", "dir_precision",
            "brier_vs_path", "cv_auc_mean",
        ]].to_string(index=False))

    if all_pipeline:
        pl_df = pd.DataFrame(all_pipeline)
        print("\n  FULL PIPELINE WIN RATE (vault 2025):")
        print(pl_df[[
            "symbol", "variant", "n_signals_base", "n_approved",
            "base_win_rate", "approved_win_rate", "filter_edge_pp",
            "signals_per_day",
        ]].to_string(index=False))

    # Verdict
    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)

    if all_pipeline and len(all_pipeline) >= 3:
        pl_df = pd.DataFrame(all_pipeline)
        for symbol in symbols:
            sym_df = pl_df[pl_df["symbol"] == symbol]
            if len(sym_df) < 2:
                continue

            current_row = sym_df[sym_df["variant"] == "current_production"]
            binary_row  = sym_df[sym_df["variant"] == "variant_a_binary"]
            class3_row  = sym_df[sym_df["variant"] == "variant_b_3class"]

            if len(current_row) == 0:
                continue

            current_wr = current_row["approved_win_rate"].values[0]

            print(f"\n  {symbol}:")
            print(f"    Production (current):   {current_wr:.1%} win rate")

            for row, label in [(binary_row, "VARIANT_A binary"),
                                (class3_row, "VARIANT_B 3-class")]:
                if len(row) == 0:
                    continue
                wr   = row["approved_win_rate"].values[0]
                diff = (wr - current_wr) * 100
                spd  = row["signals_per_day"].values[0]

                if diff > 2.0 and spd >= 4.0:
                    verdict = "✅ RECOMMENDED — clear improvement, adequate signal count"
                elif diff > 1.0 and spd >= 4.0:
                    verdict = "🟡 MARGINAL — worth full retrain validation"
                elif diff < -1.0:
                    verdict = "❌ WORSE — do not retrain"
                else:
                    verdict = "— NEUTRAL — no meaningful difference"

                print(f"    {label}: {wr:.1%} ({diff:+.1f}pp)  "
                      f"spd={spd:.1f}  → {verdict}")

    print(f"\n  Experimental models saved to: {MODEL_CACHE}/")
    print(f"  Full results: {OUTPUT_DIR}/label_comparison_*.csv")
    print("=" * 80)
    print("""
DECISION FRAMEWORK:
  +2pp or more on approved_win_rate AND signals/day >= 4.0:
    → Retrain both 5m AND 15m base models with this label design
    → Then retrain all four directional meta-filters with new base outputs
    → Validate new meta-filters on vault before deploying

  +1 to +2pp:
    → Run the 15m model through the same comparison (this script only
      swaps the 5m model; the 15m is a separate experiment)
    → If 15m also improves, combined effect may clear the +2pp bar

  No improvement or worse:
    → Label design is not the bottleneck
    → Focus on meta-filter feature replacement (JSD → replacement candidate)
    → The base model target mismatch hypothesis is incorrect for this dataset
""")


if __name__ == "__main__":
    main()


# =============================================================================
# WHAT EACH METRIC TELLS YOU
# =============================================================================
#
# dir_precision (validation period):
#   Production target: was 42.7% in Feb 2026.
#   Binary model: precision = P(trade wins | model says WIN).
#   3-class model: precision = P(direction correct | model predicts UP or DOWN).
#   Both measured against PATH-DEPENDENT outcomes, not their own training label.
#
# brier_vs_path:
#   Lower is better. Brier score of the model's directional probability
#   against path-dependent binary outcomes.
#   Current model expected to score ~0.28-0.32 (poorly calibrated).
#   Binary model expected to score ~0.22-0.26 (better calibrated by design).
#   If current model scores LOWER than binary, the label mismatch hypothesis
#   is wrong — the current label is somehow better calibrated despite
#   being point-in-time.
#
# calibration monotonicity:
#   Score of 1.0 = perfect: higher confidence bins have higher win rates.
#   Score of 0.5 = random: no relationship between confidence and outcome.
#   Current model expected: ~0.6-0.7 against path-dependent outcomes.
#   Binary model expected: ~0.8-0.9.
#   Key insight: if binary model monotonicity >> current model monotonicity,
#   the meta-filter receives cleaner probability signals, which means its
#   threshold decisions are more reliable.
#
# approved_win_rate (vault 2025):
#   THE key metric. This is what matters for deployment.
#   Measures end-to-end: does a better base model probability calibration
#   actually propagate to better meta-filter approved signal quality?
#   If yes: retrain. If no: something else is the bottleneck.
#
# =============================================================================
