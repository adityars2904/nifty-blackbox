#!/usr/bin/env python3
"""
train_meta_filter.py
=============================================================================
Phase 3: Meta-Labeling — Second-Layer Signal Filter (Ensemble Only)

Meta-Features (21 total — at the standing ceiling per handoff Section 2.1)
---------------------------------------------------------------------------
  Features 1-19:  Original validated set — DO NOT REORDER
  Feature 20:     jsd — Jensen-Shannon Divergence between 5m and 15m
                  probability distributions. Normalised to [0,1] by log(2).
  Feature 21:     bn_nifty_ratio_zscore — Rolling 20-period Z-score of the
                  BANKNIFTY/NIFTY close ratio.

Usage
-----
    python scripts/train_meta_filter.py --symbol NIFTY
    python scripts/train_meta_filter.py --symbol NIFTY --direction UP
    python scripts/train_meta_filter.py --symbol NIFTY --direction DOWN
    python scripts/train_meta_filter.py --symbol BANKNIFTY
    python scripts/train_meta_filter.py --symbol BANKNIFTY --direction UP
    python scripts/train_meta_filter.py --symbol BANKNIFTY --direction DOWN

Output
------
    --direction ALL:  backend/models/meta_filter_ensemble_{symbol}.joblib
    --direction UP:   backend/models/meta_filter_{symbol}_up.joblib
    --direction DOWN: backend/models/meta_filter_{symbol}_down.joblib
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "backend"))

import joblib
import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier

from backend.ml.data_loader import load_validation_data
from backend.ml.feature_engineering import add_features, MOVEMENT_FEATURES
from backend.ml.labels import label_movement
from backend.ml.ensemble_predictor import EnsemblePredictor
from backend.ml.multi_horizon_predictor import MultiHorizonPredictor


# ============================================================================
# CONFIG
# ============================================================================

MODEL_DIR       = "backend/models"
N_SPLITS        = 4
FORWARD_CANDLES = 12
ATR_STOP_MULT   = 1.0
ATR_TARGET_MULT = 1.5

# ============================================================================
# META-FEATURES — 21 total (standing ceiling)
#
# The order here is the exact order XGBoost is trained on.
# meta_filter_service.py and replay.py MUST retrieve this order via
# model.get_booster().feature_names — never hardcode positions independently.
# ============================================================================

META_FEATURES = [
    # ── Original 19 (DO NOT REORDER) ─────────────────────────────────────────
    "ens_confidence",
    "conf_5m",
    "conf_15m",
    "prob_gap",
    "p_neutral",
    "vol_ratio",
    "vol_expansion",
    "regime_trend",
    "z_score_20",
    "close_position",
    "z_score_distance_from_vwap",
    "time_sin",
    "regime_time_of_day",
    "recent_win_rate_5",
    "recent_win_rate_10",
    "consecutive_losses",
    "consecutive_wins",
    "signals_today",
    "bars_since_last_signal",
    # ── Feature #20: JSD disagreement ────────────────────────────────────────
    "jsd",
    # ── Feature #21: BankNifty/NIFTY ratio Z-score ───────────────────────────
    "bn_nifty_ratio_zscore",
]


# ============================================================================
# HELPERS
# ============================================================================

def _compute_jsd(probs_5m: np.ndarray, probs_15m: np.ndarray) -> float:
    """
    Jensen-Shannon Divergence between two 3-class probability distributions.
    Normalised to [0, 1] by dividing by log(2).
    """
    p = np.clip(probs_5m,  1e-10, None)
    q = np.clip(probs_15m, 1e-10, None)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    return float((0.5 * entropy(p, m) + 0.5 * entropy(q, m)) / np.log(2))


# ============================================================================
# STEP 1 — WIN / LOSS LABELLER
# ============================================================================

def label_signal_outcomes(signals: pd.DataFrame, feat_5m: pd.DataFrame) -> pd.Series:
    """Simulate path-dependent WIN/LOSS for each signal. Returns pd.Series of 0/1."""
    feat_5m = feat_5m.copy().reset_index(drop=True)
    feat_ts = np.asarray(
        pd.to_datetime(feat_5m["timestamp"]).dt.tz_localize(None).values,
        dtype="datetime64[ns]",
    )
    outcomes = []

    for _, row in signals.iterrows():
        direction = row["direction"]
        atr       = float(row["atr"])

        if atr <= 0 or np.isnan(atr):
            outcomes.append(0)
            continue

        stop_pts   = atr * ATR_STOP_MULT
        target_pts = atr * ATR_TARGET_MULT

        sig_ts_naive = pd.to_datetime(row["timestamp"]).tz_localize(None)
        idx   = int(np.searchsorted(feat_ts, np.datetime64(sig_ts_naive), side="left"))
        entry = float(feat_5m.iloc[idx]["close"]) if idx < len(feat_5m) else np.nan

        if np.isnan(entry):
            outcomes.append(0)
            continue

        result  = 0
        end_idx = min(idx + 1 + FORWARD_CANDLES, len(feat_5m))

        for fwd in range(idx + 1, end_idx):
            hi = float(feat_5m.iloc[fwd]["high"])
            lo = float(feat_5m.iloc[fwd]["low"])
            if direction == "UP":
                if hi >= entry + target_pts:
                    result = 1
                    break
                if lo <= entry - stop_pts:
                    break
            else:
                if lo <= entry - target_pts:
                    result = 1
                    break
                if hi >= entry + stop_pts:
                    break

        outcomes.append(result)

    return pd.Series(outcomes, index=signals.index, name="win")


# ============================================================================
# STEP 2a — RATIO Z-SCORE (Feature #21)
# ============================================================================

def _add_ratio_zscore(signals_df: pd.DataFrame, feat_5m: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Compute rolling 20-period BankNifty/NIFTY ratio Z-score for each signal
    row and attach as 'bn_nifty_ratio_zscore'. Falls back to 0.0 on any error.

    For NIFTY: ratio = BANKNIFTY_close / NIFTY_close
    For BANKNIFTY: ratio = BANKNIFTY_close / NIFTY_close  (same formula;
                   companion symbol is NIFTY)
    """
    df = signals_df.copy()
    try:
        companion_sym = "BANKNIFTY" if symbol == "NIFTY" else "NIFTY"
        companion_raw  = load_validation_data(symbol=companion_sym, timeframe="5m")
        companion_feat = add_features(companion_raw).dropna(subset=["close"]).reset_index(drop=True)

        base_ts = pd.to_datetime(feat_5m["timestamp"]).dt.tz_localize(None)
        comp_ts = pd.to_datetime(companion_feat["timestamp"]).dt.tz_localize(None)

        base_df = pd.DataFrame({"ts": base_ts,  "close_base": feat_5m["close"].values})
        comp_df = pd.DataFrame({"ts": comp_ts,  "close_comp": companion_feat["close"].values})

        merged = pd.merge_asof(
            base_df.sort_values("ts"),
            comp_df.sort_values("ts"),
            on="ts", direction="backward",
        ).dropna(subset=["close_comp"])

        # Always ratio = BANKNIFTY / NIFTY regardless of which symbol is base
        if symbol == "NIFTY":
            merged["ratio"] = merged["close_comp"] / merged["close_base"]
        else:
            merged["ratio"] = merged["close_base"] / merged["close_comp"]

        merged["ratio_z"] = (
            (merged["ratio"] - merged["ratio"].rolling(20).mean())
            / merged["ratio"].rolling(20).std().replace(0, np.nan)
        ).fillna(0.0)

        sig_ts  = pd.to_datetime(df["timestamp"]).dt.tz_localize(None).values.astype("datetime64[ns]")
        mrg_ts  = merged["ts"].values.astype("datetime64[ns]")
        # Ensure both are numpy arrays for searchsorted compatibility
        sig_ts_np = np.array(sig_ts)
        mrg_ts_np = np.array(mrg_ts)
        idx     = np.clip(np.searchsorted(mrg_ts_np, sig_ts_np, side="right") - 1, 0, len(merged) - 1)
        df["bn_nifty_ratio_zscore"] = merged["ratio_z"].values[idx].astype(float)

        n_nz = (df["bn_nifty_ratio_zscore"] != 0).sum()
        print(f"  Feature #21 (bn_nifty_ratio_zscore): {n_nz:,}/{len(df):,} rows non-zero")

    except Exception as exc:
        print(f"  ⚠️  bn_nifty_ratio_zscore failed: {exc} — defaulting to 0.0")
        df["bn_nifty_ratio_zscore"] = 0.0

    return df


# ============================================================================
# STEP 2b — META FEATURE BUILDER
# ============================================================================

def build_meta_features(signals: pd.DataFrame, feat_5m: pd.DataFrame) -> pd.DataFrame:
    """Attach all 21 meta-features to the signals DataFrame."""
    df = signals.copy().reset_index(drop=True)

    # prob_gap and p_neutral
    prob_cols = df[["prob_down", "prob_neutral", "prob_up"]].values
    sorted_p  = np.sort(prob_cols, axis=1)
    df["prob_gap"]  = sorted_p[:, 2] - sorted_p[:, 1]
    df["p_neutral"] = df["prob_neutral"]

    # Rolling win history
    wins  = df["win"].values
    rwr5  = np.full(len(df), 0.5)
    rwr10 = np.full(len(df), 0.5)
    closs = np.zeros(len(df))
    cwins = np.zeros(len(df))

    for i in range(len(df)):
        if i >= 5:
            rwr5[i]  = float(np.mean(np.asarray(wins[i-5:i],  dtype=float)))
        if i >= 10:
            rwr10[i] = float(np.mean(np.asarray(wins[i-10:i], dtype=float)))
        if i > 0:
            last   = wins[i - 1]
            streak = 0
            for j in range(i - 1, -1, -1):
                if wins[j] == last:
                    streak += 1
                else:
                    break
            if last == 0:
                closs[i] = streak
            else:
                cwins[i] = streak

    df["recent_win_rate_5"]  = rwr5
    df["recent_win_rate_10"] = rwr10
    df["consecutive_losses"] = closs
    df["consecutive_wins"]   = cwins

    # signals_today
    ts = pd.to_datetime(df["timestamp"])
    df["_date"]        = ts.dt.date
    df["signals_today"] = df.groupby("_date").cumcount()
    df = df.drop(columns=["_date"])

    # bars_since_last_signal
    feat_ts_vals = np.asarray(
        pd.to_datetime(feat_5m["timestamp"]).dt.tz_localize(None).values,
        dtype="datetime64[ns]",
    )
    sig_ts_vals = np.asarray(
        pd.to_datetime(df["timestamp"]).dt.tz_localize(None).values,
        dtype="datetime64[ns]",
    )
    sig_idx    = np.searchsorted(feat_ts_vals, sig_ts_vals, side="left")
    bars_since = np.zeros(len(df))
    for i in range(1, len(df)):
        bars_since[i] = float(sig_idx[i] - sig_idx[i - 1])
    df["bars_since_last_signal"] = bars_since

    # Feature #20: JSD
    has_5m  = all(c in df.columns for c in ["p_down_5m",  "p_neutral_5m",  "p_up_5m"])
    has_15m = all(c in df.columns for c in ["p_down_15m", "p_neutral_15m", "p_up_15m"])

    if has_5m and has_15m:
        p5  = df[["p_down_5m",  "p_neutral_5m",  "p_up_5m"]].values.astype(float)
        p15 = df[["p_down_15m", "p_neutral_15m", "p_up_15m"]].values.astype(float)
        jsd_vals = np.array([_compute_jsd(p5[i], p15[i]) for i in range(len(df))])
        print(f"  Feature #20 (JSD): mean={jsd_vals.mean():.4f}  max={jsd_vals.max():.4f}")
    else:
        jsd_vals = np.zeros(len(df))
        print("  ⚠️  Feature #20 (JSD): prob columns missing — set to 0.0")
    df["jsd"] = jsd_vals

    # Feature #21: already attached by generate_ensemble_signals via _add_ratio_zscore
    if "bn_nifty_ratio_zscore" not in df.columns:
        df["bn_nifty_ratio_zscore"] = 0.0
        print("  ⚠️  Feature #21 (bn_nifty_ratio_zscore): missing — set to 0.0")

    return df


# ============================================================================
# STEP 3 — ENSEMBLE SIGNAL GENERATORS
# ============================================================================

def generate_ensemble_signals(
    feat_5m: pd.DataFrame,
    feat_15m: pd.DataFrame,
    ensemble: EnsemblePredictor,
    symbol: str = "NIFTY",
) -> pd.DataFrame:
    """
    Replay 2-model ensemble veto logic. Preserves per-timeframe probability
    columns so build_meta_features() can compute JSD (Feature #20).
    Attaches bn_nifty_ratio_zscore (Feature #21) via _add_ratio_zscore().
    """
    ens_feats = ensemble.model_5m.get_booster().feature_names

    p5  = np.asarray(ensemble.model_5m.predict_proba(feat_5m[ens_feats].astype(float)),  dtype=np.float64)
    p15 = np.asarray(ensemble.model_15m.predict_proba(feat_15m[ens_feats].astype(float)), dtype=np.float64)

    f5  = feat_5m.copy()
    f15 = feat_15m.copy()
    f5[["p_down_5m",   "p_neutral_5m",  "p_up_5m"]]  = p5
    f15[["p_down_15m", "p_neutral_15m", "p_up_15m"]] = p15

    merged = pd.merge_asof(
        f5.sort_values("timestamp"),
        f15[["timestamp","p_down_15m","p_neutral_15m","p_up_15m"]].sort_values("timestamp"),
        on="timestamp", direction="backward",
    ).dropna(subset=["p_down_15m"])

    merged["prob_down"]    = merged["p_down_5m"]    * 0.30 + merged["p_down_15m"]    * 0.70
    merged["prob_neutral"] = merged["p_neutral_5m"] * 0.30 + merged["p_neutral_15m"] * 0.70
    merged["prob_up"]      = merged["p_up_5m"]      * 0.30 + merged["p_up_15m"]      * 0.70

    probs = merged[["prob_down","prob_neutral","prob_up"]].values
    merged["ens_confidence"] = np.max(probs, axis=1)
    merged["ensemble_dir"]   = np.argmax(probs, axis=1)
    merged["ensemble_dir"]   = merged["ensemble_dir"].map({0:"DOWN",1:"NEUTRAL",2:"UP"})
    merged["conf_5m"]        = np.max(p5, axis=1)[:len(merged)]

    ts5  = np.array(pd.to_datetime(merged["timestamp"]).values, dtype="datetime64[ns]")
    ts15 = np.array(pd.to_datetime(feat_15m["timestamp"]).values, dtype="datetime64[ns]")
    i15  = np.clip(np.searchsorted(ts15, ts5, side="right") - 1, 0, len(p15) - 1)
    merged["conf_15m"] = np.max(p15[i15], axis=1)
    merged["pred_15m"] = np.argmax(p15[i15], axis=1)

    mask = (
        (merged["ensemble_dir"] != "NEUTRAL") &
        (merged["ens_confidence"] >= ensemble.min_confidence) &
        (merged["pred_15m"] != 1)
    )
    signals = merged[mask].copy().rename(columns={"ensemble_dir": "direction"})
    signals = _add_ratio_zscore(signals, feat_5m, symbol)

    print(f"  Signals: {len(signals):,}  UP={( signals['direction']=='UP').sum():,}  DOWN={(signals['direction']=='DOWN').sum():,}")
    return signals.reset_index(drop=True)


def generate_3model_signals(
    feat_5m: pd.DataFrame,
    feat_15m: pd.DataFrame,
    feat_30m: pd.DataFrame,
    predictor: MultiHorizonPredictor,
    symbol: str = "NIFTY",
) -> pd.DataFrame:
    """3-model variant. Same logic as generate_ensemble_signals but with 30m."""
    f5m  = predictor.model_5m.get_booster().feature_names  or MOVEMENT_FEATURES
    f15m = predictor.model_15m.get_booster().feature_names or MOVEMENT_FEATURES
    f30m = predictor.model_30m.get_booster().feature_names or MOVEMENT_FEATURES

    p5  = np.asarray(predictor.model_5m.predict_proba(feat_5m[f5m].astype(float)),   dtype=np.float64)
    p15 = np.asarray(predictor.model_15m.predict_proba(feat_15m[f15m].astype(float)), dtype=np.float64)
    p30 = np.asarray(predictor.model_30m.predict_proba(feat_30m[f30m].astype(float)), dtype=np.float64)

    fc5  = feat_5m.copy();  fc5[["p_down_5m",  "p_neutral_5m",  "p_up_5m"]]  = p5
    fc15 = feat_15m.copy(); fc15[["p_down_15m","p_neutral_15m","p_up_15m"]] = p15
    fc30 = feat_30m.copy(); fc30[["p_down_30m","p_neutral_30m","p_up_30m"]] = p30

    merged = pd.merge_asof(fc5.sort_values("timestamp"),
        fc15[["timestamp","p_down_15m","p_neutral_15m","p_up_15m"]].sort_values("timestamp"),
        on="timestamp", direction="backward").dropna(subset=["p_down_15m"])
    merged = pd.merge_asof(merged.sort_values("timestamp"),
        fc30[["timestamp","p_down_30m","p_neutral_30m","p_up_30m"]].sort_values("timestamp"),
        on="timestamp", direction="backward").dropna(subset=["p_down_30m"])

    w5, w15, w30 = predictor.weight_5m, predictor.weight_15m, predictor.weight_30m
    merged["prob_down"]    = merged["p_down_5m"]*w5    + merged["p_down_15m"]*w15    + merged["p_down_30m"]*w30
    merged["prob_neutral"] = merged["p_neutral_5m"]*w5 + merged["p_neutral_15m"]*w15 + merged["p_neutral_30m"]*w30
    merged["prob_up"]      = merged["p_up_5m"]*w5      + merged["p_up_15m"]*w15      + merged["p_up_30m"]*w30

    probs = merged[["prob_down","prob_neutral","prob_up"]].values
    merged["ens_confidence"] = np.max(probs, axis=1)
    merged["ensemble_dir"]   = pd.Series(np.argmax(probs, axis=1)).map({0:"DOWN",1:"NEUTRAL",2:"UP"}).values
    merged["conf_5m"]        = np.max(p5, axis=1)[:len(merged)]

    ts5  = np.array(pd.to_datetime(merged["timestamp"]).values, dtype="datetime64[ns]")
    ts15 = np.array(pd.to_datetime(feat_15m["timestamp"]).values, dtype="datetime64[ns]")
    i15  = np.clip(np.searchsorted(ts15, ts5, side="right") - 1, 0, len(p15) - 1)
    merged["conf_15m"] = np.max(p15[i15], axis=1)
    merged["pred_15m"] = np.argmax(p15[i15], axis=1)

    mask = (
        (merged["ensemble_dir"] != "NEUTRAL") &
        (merged["ens_confidence"] >= predictor.min_confidence) &
        (merged["pred_15m"] != 1)
    )
    signals = merged[mask].copy().rename(columns={"ensemble_dir": "direction"})
    signals = _add_ratio_zscore(signals, feat_5m, symbol)

    print(f"  Signals: {len(signals):,}  UP={(signals['direction']=='UP').sum():,}  DOWN={(signals['direction']=='DOWN').sum():,}")
    return signals.reset_index(drop=True)


# ============================================================================
# STEP 4 — META-FILTER TRAINER
# ============================================================================

def train_meta_filter(signal_df: pd.DataFrame) -> tuple:
    missing   = [f for f in META_FEATURES if f not in signal_df.columns]
    available = [f for f in META_FEATURES if f in signal_df.columns]
    if missing:
        print(f"  ⚠️  Missing features: {missing}")

    X = signal_df[available].astype(float)
    y = signal_df["win"].astype(int)

    base_rate = float(y.mean())
    print(f"\n  Signal set: {len(X):,} | WIN={int(y.sum())} ({base_rate:.1%}) | "
          f"Features={len(available)}")

    if len(X) < 100:
        return None, {"error": "insufficient_signals", "n_signals": len(X)}

    classes     = np.unique(y)
    cw          = compute_class_weight("balanced", classes=classes, y=y)
    weight_dict = {int(c): float(w) for c, w in zip(classes, cw)}
    weights     = np.array([weight_dict[int(v)] for v in y])

    tscv         = TimeSeriesSplit(n_splits=N_SPLITS)
    fold_metrics = []

    print(f"\n  Walk-Forward CV ({N_SPLITS} folds):")
    for fold_num, (tr_idx, te_idx) in enumerate(tscv.split(X), 1):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]

        if len(np.unique(y_te)) < 2:
            print(f"  Fold {fold_num}: skipped")
            continue

        m = XGBClassifier(
            objective="binary:logistic", n_estimators=200, max_depth=3,
            learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
            min_child_weight=10, gamma=0.2, eval_metric="logloss",
            random_state=42, verbosity=0,
        )
        m.fit(X_tr, y_tr, sample_weight=weights[tr_idx])

        probs   = m.predict_proba(X_te)[:, 1]
        preds   = (probs >= 0.50).astype(int)
        prec    = float(precision_score(y_te, preds, zero_division=0))
        rec     = float(recall_score(y_te,   preds, zero_division=0))
        auc     = float(roc_auc_score(np.asarray(y_te, dtype=int), probs))
        hc      = probs >= 0.55
        hc_prec = float(precision_score(y_te[hc], preds[hc], zero_division=0)) if hc.sum() > 0 else 0.0
        hc_rate = float(hc.mean())

        print(f"  Fold {fold_num}: Prec={prec:.3f} Rec={rec:.3f} AUC={auc:.3f} "
              f"HC@55%={hc_prec:.3f} HC-Rate={hc_rate:.1%} "
              f"[{len(tr_idx)}/{len(te_idx)}]")
        fold_metrics.append({"precision":prec,"recall":rec,"auc":auc,"hc_prec":hc_prec,"hc_rate":hc_rate})

    if not fold_metrics:
        return None, {"error": "no_valid_folds"}

    mp  = float(np.mean([m["precision"] for m in fold_metrics]))
    ma  = float(np.mean([m["auc"]       for m in fold_metrics]))
    mhp = float(np.mean([m["hc_prec"]   for m in fold_metrics]))
    mhr = float(np.mean([m["hc_rate"]   for m in fold_metrics]))
    print(f"\n  CV Mean → Prec={mp:.3f} AUC={ma:.3f} HC@55%={mhp:.3f} HC-Rate={mhr:.1%}")
    print(f"\n  Training final model on all {len(X):,} signals…")

    final = XGBClassifier(
        objective="binary:logistic", n_estimators=200, max_depth=3,
        learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        min_child_weight=10, gamma=0.2, eval_metric="logloss",
        random_state=42, verbosity=0,
    )
    final.fit(X, y, sample_weight=weights)

    return final, {
        "n_signals":       len(X),
        "base_win_rate":   base_rate,
        "cv_prec_mean":    mp,
        "cv_auc_mean":     ma,
        "cv_hc_prec_mean": mhp,
        "cv_hc_rate_mean": mhr,
        "features_used":   available,
    }


# ============================================================================
# STEP 5 — FEATURE IMPORTANCE
# ============================================================================

def print_feature_importance(model, feature_names: list, top_n: int = 12) -> None:
    pairs  = sorted(zip(feature_names, model.feature_importances_), key=lambda x: x[1], reverse=True)
    shown  = set()
    print(f"\n  Top {top_n} meta-features by importance:")
    for name, imp in pairs[:top_n]:
        bar = "█" * int(imp * 200)
        tag = "  ← NEW" if name in ("jsd", "bn_nifty_ratio_zscore") else ""
        print(f"    {name:<35} {imp:.4f}  {bar}{tag}")
        shown.add(name)
    for name, imp in pairs:
        if name in ("jsd", "bn_nifty_ratio_zscore") and name not in shown:
            bar = "█" * int(imp * 200)
            print(f"    {name:<35} {imp:.4f}  {bar}  ← NEW")


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol",    default="NIFTY",  choices=["NIFTY","BANKNIFTY"])
    parser.add_argument("--models",    default=2,         type=int, choices=[2,3])
    parser.add_argument("--direction", default="ALL",     choices=["ALL","UP","DOWN"])
    args = parser.parse_args()

    SYMBOL, N_BASE_MODELS, DIRECTION = args.symbol, args.models, args.direction

    print("=" * 70)
    print(f"🔮 META-FILTER TRAINING — {SYMBOL} "
          f"({N_BASE_MODELS}-model, direction={DIRECTION}, features=21)")
    print("=" * 70)

    # Load data
    print("\nLoading validation data…")
    try:
        val_5m  = load_validation_data(symbol=SYMBOL, timeframe="5m")
        val_15m = load_validation_data(symbol=SYMBOL, timeframe="15m")
    except Exception as e:
        print(f"  ⚠️  QuestDB failed: {e}")
        val_5m = val_15m = pd.DataFrame()

    if len(val_5m) == 0 or len(val_15m) == 0:
        for sym, tf, attr in [("5m", "val_5m", "val_5m"), ("15m", "val_15m", "val_15m")]:
            p = Path(f"data/val_{sym}_{SYMBOL.lower()}.parquet")
            if p.exists():
                if tf == "val_5m":
                    val_5m  = pd.read_parquet(p)
                else:
                    val_15m = pd.read_parquet(p)
        if len(val_5m) == 0 or len(val_15m) == 0:
            print("❌ No validation data available.")
            return

    feat_5m  = add_features(val_5m).copy()
    feat_5m["label"] = label_movement(feat_5m)
    feat_5m  = feat_5m.dropna(subset=MOVEMENT_FEATURES + ["label","atr"]).reset_index(drop=True)
    feat_15m = add_features(val_15m).dropna(subset=MOVEMENT_FEATURES).reset_index(drop=True)
    print(f"  5m: {len(feat_5m):,}  15m: {len(feat_15m):,}")

    feat_30m = None
    if N_BASE_MODELS == 3:
        try:
            val_30m = load_validation_data(symbol=SYMBOL, timeframe="30m")
        except Exception:
            val_30m = pd.DataFrame()
        if len(val_30m) == 0:
            p30 = Path(f"data/val_30m_{SYMBOL.lower()}.parquet")
            val_30m = pd.read_parquet(p30) if p30.exists() else pd.DataFrame()
        if len(val_30m) == 0:
            print("❌ No 30m data.")
            return
        feat_30m = add_features(val_30m).dropna(subset=MOVEMENT_FEATURES).reset_index(drop=True)
        print(f"  30m: {len(feat_30m):,}")

    # Generate signals
    if N_BASE_MODELS == 3:
        predictor   = MultiHorizonPredictor.load(MODEL_DIR, symbol=SYMBOL)
        assert feat_30m is not None
        ens_signals = generate_3model_signals(feat_5m, feat_15m, feat_30m, predictor, SYMBOL)
    else:
        ensemble    = EnsemblePredictor.load(MODEL_DIR, symbol=SYMBOL)
        ens_signals = generate_ensemble_signals(feat_5m, feat_15m, ensemble, SYMBOL)

    ens_signals["win"] = label_signal_outcomes(ens_signals, feat_5m)
    print(f"  Base win rate: {ens_signals['win'].mean():.1%}")

    # Direction filter
    if DIRECTION != "ALL":
        ens_signals = ens_signals[ens_signals["direction"] == DIRECTION].reset_index(drop=True)
        print(f"  Filtered to {DIRECTION}: {len(ens_signals):,} signals")
        if len(ens_signals) < 100:
            print("❌ Too few signals.")
            return

    ens_signals = build_meta_features(ens_signals, feat_5m)

    # Train
    model_ens, metrics_ens = train_meta_filter(ens_signals)
    if model_ens is None:
        print("❌ Training failed.")
        return

    print_feature_importance(model_ens, metrics_ens["features_used"])

    # Save
    if DIRECTION == "UP":
        path = f"{MODEL_DIR}/meta_filter_{SYMBOL.lower()}_up.joblib"
    elif DIRECTION == "DOWN":
        path = f"{MODEL_DIR}/meta_filter_{SYMBOL.lower()}_down.joblib"
    elif N_BASE_MODELS == 3:
        path = f"{MODEL_DIR}/meta_filter_3model_{SYMBOL.lower()}.joblib"
    else:
        path = f"{MODEL_DIR}/meta_filter_ensemble_{SYMBOL.lower()}.joblib"

    joblib.dump({
        "model":         model_ens,
        "metrics":       metrics_ens,
        "symbol":        SYMBOL,
        "n_base_models": N_BASE_MODELS,
        "direction":     DIRECTION,
        "n_features":    len(metrics_ens["features_used"]),
    }, path)
    print(f"\n  💾 Saved: {path}  ({len(metrics_ens['features_used'])} features)")

    # Diagnostics
    X_ens      = ens_signals[metrics_ens["features_used"]].astype(float)
    meta_probs = model_ens.predict_proba(X_ens)[:, 1]
    thr        = 0.60 if DIRECTION == "UP" else 0.52 if DIRECTION == "DOWN" else 0.55
    approved   = ens_signals[meta_probs >= thr]
    print(f"\n  Approved @ {thr}: {len(approved):,} ({len(approved)/len(ens_signals)*100:.1f}%)")
    if len(approved) > 0:
        print(f"  Approved win rate: {approved['win'].mean():.1%}")

    # Gate
    hc = metrics_ens.get("cv_hc_prec_mean", 0)
    print(f"\n  Decision: {'✅ PASS' if hc >= 0.50 else '❌ FAIL'}  HC@55%={hc:.3f}")
    if hc >= 0.50:
        print("  Ready for vault validation.")


if __name__ == "__main__":
    main()


# =============================================================================
# CHANGES
# =============================================================================
# Added Feature #20 (jsd) and Feature #21 (bn_nifty_ratio_zscore) to
# META_FEATURES, bringing the total to 21 — the standing ceiling.
#
# Feature #20 — jsd:
#   Computed in build_meta_features() from p_down_5m / p_neutral_5m / p_up_5m
#   and p_down_15m / p_neutral_15m / p_up_15m columns that
#   generate_ensemble_signals() preserves from the merge_asof. Normalised to
#   [0,1] via _compute_jsd() which divides the raw JSD by log(2).
#
# Feature #21 — bn_nifty_ratio_zscore:
#   Computed by _add_ratio_zscore(), called at the end of both signal
#   generators. Loads the companion symbol data from QuestDB, merges by
#   timestamp, computes BANKNIFTY/NIFTY close ratio, applies a rolling-20
#   Z-score, and maps back to signal rows. Falls back to 0.0 on any error.
#
# Both signal generator functions now accept a `symbol` parameter so
# _add_ratio_zscore() knows which symbol is the base and which is the
# companion when computing the ratio.
#
# meta_filter_service.py and replay.py MUST use
#   model.get_booster().feature_names
# to select columns when calling predict_proba(). Never hardcode column
# positions independently — the feature names embedded in the model are
# the single source of truth.
#
# Never retrain movement_predictor_*.joblib under any circumstances.
# =============================================================================