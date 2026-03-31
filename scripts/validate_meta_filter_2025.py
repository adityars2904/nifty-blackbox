#!/usr/bin/env python3
"""
validate_meta_filter_2025.py
=============================================================================
Validate Meta-Filters on 2025 Vault Data (Out-of-Sample)

Purpose
-------
Tests meta-filter models on 2025-01-01 to 2025-12-10 (unseen vault data).

Decision Gate
-------------
  Pooled model:      HC Win Rate ≥ 60%
  UP model:          HC Win Rate ≥ 63%  AND signals ≥ 4.0/day
  DOWN model:        HC Win Rate ≥ 65%  AND signals ≥ 4.0/day
  Any model 55-60%:  Marginal, deploy with 0.25% risk
  Any model < 55%:   Do not deploy

Usage
-----
    python scripts/validate_meta_filter_2025.py --symbol NIFTY
    python scripts/validate_meta_filter_2025.py --symbol BANKNIFTY
    python scripts/validate_meta_filter_2025.py --symbol NIFTY --direction UP
    python scripts/validate_meta_filter_2025.py --symbol NIFTY --direction DOWN
    python scripts/validate_meta_filter_2025.py --symbol BANKNIFTY --direction UP
    python scripts/validate_meta_filter_2025.py --symbol BANKNIFTY --direction DOWN
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import joblib
import numpy as np
import pandas as pd
from scipy.stats import entropy

from backend.ml.data_loader import fetch_candles
from backend.ml.feature_engineering import add_features, MOVEMENT_FEATURES
from backend.ml.labels import label_movement
from backend.ml.ensemble_predictor import EnsemblePredictor


# ============================================================================
# CONFIG
# ============================================================================

MODEL_DIR = "backend/models"

VAULT_START = "2025-01-01 09:15:00"
VAULT_END   = "2025-12-10 15:30:00"

FORWARD_CANDLES = 12
ATR_STOP_MULT   = 1.0
ATR_TARGET_MULT = 1.5


# ============================================================================
# HELPERS — Features #20 and #21
# ============================================================================

def _compute_jsd(probs_5m: np.ndarray, probs_15m: np.ndarray) -> float:
    """
    Jensen-Shannon Divergence between two 3-class probability distributions.
    Normalised to [0, 1] by dividing by log(2).
    Matches train_meta_filter.py exactly.
    """
    p = np.clip(probs_5m,  1e-10, None)
    q = np.clip(probs_15m, 1e-10, None)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    return float((0.5 * entropy(p, m) + 0.5 * entropy(q, m)) / np.log(2))


def _add_ratio_zscore(
    signals_df: pd.DataFrame,
    feat_5m: pd.DataFrame,
    symbol: str,
) -> pd.DataFrame:
    """
    Compute rolling 20-period BankNifty/NIFTY ratio Z-score and attach as
    'bn_nifty_ratio_zscore'. Falls back to 0.0 on any error.
    Matches train_meta_filter.py exactly.
    """
    df = signals_df.copy()
    try:
        companion_sym  = "BANKNIFTY" if symbol == "NIFTY" else "NIFTY"
        companion_raw  = fetch_candles(
            companion_sym, timeframe="5m",
            start_date=VAULT_START, end_date=VAULT_END,
        )
        if len(companion_raw) == 0:
            raise ValueError(f"No vault data for {companion_sym}")

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

        # Always BANKNIFTY / NIFTY regardless of which symbol is being validated
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
# SIGNAL LABELLER
# ============================================================================

def label_signal_outcomes(signals: pd.DataFrame, feat_5m: pd.DataFrame) -> pd.Series:
    """Simulate path-dependent WIN/LOSS for each signal row."""
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
# META FEATURE BUILDER
# ============================================================================

def build_meta_features(
    signals: pd.DataFrame,
    feat_5m: pd.DataFrame,
    symbol: str,
) -> pd.DataFrame:
    """
    Attach all meta-features to the signals DataFrame.

    Computes the original 19 rolling/history features plus:
      Feature #20 (jsd):                from 5m/15m probability columns
      Feature #21 (bn_nifty_ratio_zscore): via _add_ratio_zscore()

    validate_meta_filter() selects only the features the loaded model
    expects via model.get_booster().feature_names.
    """
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

    # Feature #20: JSD from per-row 5m/15m probability columns
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

    # Feature #21: BankNifty/NIFTY ratio Z-score
    df = _add_ratio_zscore(df, feat_5m, symbol)

    return df


# ============================================================================
# SIGNAL GENERATOR
# ============================================================================

def generate_ensemble_signals(
    feat_5m: pd.DataFrame,
    feat_15m: pd.DataFrame,
    ensemble: EnsemblePredictor,
) -> pd.DataFrame:
    """
    Generate ensemble signals with veto logic.
    Preserves per-timeframe probability columns so build_meta_features()
    can compute JSD (Feature #20).
    """
    ensemble_features = ensemble.model_5m.get_booster().feature_names

    X_5m  = feat_5m[ensemble_features].astype(float)
    X_15m = feat_15m[ensemble_features].astype(float)

    p5  = np.asarray(ensemble.model_5m.predict_proba(X_5m),   dtype=np.float64)
    p15 = np.asarray(ensemble.model_15m.predict_proba(X_15m), dtype=np.float64)

    feat_5m_cp  = feat_5m.copy()
    feat_15m_cp = feat_15m.copy()
    # Preserve raw per-model probabilities — needed by build_meta_features for JSD
    feat_5m_cp[["p_down_5m",  "p_neutral_5m",  "p_up_5m"]]  = p5
    feat_15m_cp[["p_down_15m","p_neutral_15m", "p_up_15m"]] = p15

    merged = pd.merge_asof(
        feat_5m_cp.sort_values("timestamp"),
        feat_15m_cp[["timestamp","p_down_15m","p_neutral_15m","p_up_15m"]].sort_values("timestamp"),
        on="timestamp", direction="backward",
    ).dropna(subset=["p_down_15m"])

    w5, w15 = 0.30, 0.70
    merged["prob_down"]    = merged["p_down_5m"]    * w5 + merged["p_down_15m"]    * w15
    merged["prob_neutral"] = merged["p_neutral_5m"] * w5 + merged["p_neutral_15m"] * w15
    merged["prob_up"]      = merged["p_up_5m"]      * w5 + merged["p_up_15m"]      * w15

    probs_matrix = merged[["prob_down","prob_neutral","prob_up"]].values
    merged["ens_pred"]       = np.argmax(probs_matrix, axis=1)
    merged["ens_confidence"] = np.max(probs_matrix, axis=1)
    merged["ensemble_dir"]   = merged["ens_pred"].map({0:"DOWN",1:"NEUTRAL",2:"UP"})
    merged["conf_5m"]        = np.max(p5, axis=1)[:len(merged)]

    ts_5m_vals  = np.array(pd.to_datetime(merged["timestamp"]).values,  dtype="datetime64[ns]")
    ts_15m_vals = np.array(pd.to_datetime(feat_15m["timestamp"]).values, dtype="datetime64[ns]")
    idx_15m     = np.clip(np.searchsorted(ts_15m_vals, ts_5m_vals, side="right") - 1, 0, len(p15) - 1)
    merged["conf_15m"] = np.max(p15[idx_15m], axis=1)
    merged["pred_15m"] = np.argmax(p15[idx_15m], axis=1)

    signal_mask = (
        (merged["ensemble_dir"] != "NEUTRAL") &
        (merged["ens_confidence"] >= ensemble.min_confidence) &
        (merged["pred_15m"] != 1)
    )

    signals = merged[signal_mask].copy()
    signals = signals.rename(columns={"ensemble_dir": "direction"})
    return signals.reset_index(drop=True)


# ============================================================================
# VALIDATION CORE
# ============================================================================

def validate_meta_filter(
    signals: pd.DataFrame,
    meta_filter,
    base_name: str,
    threshold: float = 0.55,
) -> dict:
    """
    Apply meta-filter and compute validation metrics.

    Uses model.get_booster().feature_names to build the feature matrix —
    works correctly for both 19-feature and 21-feature models.
    Raises a clear error if any required features are missing.
    """
    # Use the model's own feature list — the single source of truth
    feature_names = list(meta_filter.get_booster().feature_names)

    missing = [f for f in feature_names if f not in signals.columns]
    if missing:
        raise ValueError(
            f"Signals DataFrame is missing {len(missing)} features the model requires: "
            f"{missing}\n"
            f"Ensure build_meta_features() is called before validate_meta_filter()."
        )

    X          = signals[feature_names].astype(float)
    meta_probs = meta_filter.predict_proba(X)[:, 1]

    base_win_rate = float(signals["win"].mean())
    base_n        = len(signals)

    hc_mask     = meta_probs >= threshold
    hc_sigs     = signals[hc_mask].copy()
    hc_win_rate = float(hc_sigs["win"].mean()) if len(hc_sigs) > 0 else 0.0
    hc_n        = len(hc_sigs)
    hc_rate     = float(hc_mask.mean())

    up_mask   = signals["direction"] == "UP"
    down_mask = signals["direction"] == "DOWN"

    up_base_wr   = float(signals[up_mask]["win"].mean())   if up_mask.sum()   > 0 else 0.0
    down_base_wr = float(signals[down_mask]["win"].mean()) if down_mask.sum() > 0 else 0.0

    up_hc    = hc_sigs[hc_sigs["direction"] == "UP"]
    down_hc  = hc_sigs[hc_sigs["direction"] == "DOWN"]
    up_hc_wr   = float(up_hc["win"].mean())   if len(up_hc)   > 0 else 0.0
    down_hc_wr = float(down_hc["win"].mean()) if len(down_hc) > 0 else 0.0

    rejected = signals[~hc_mask].copy()
    rej_wr   = float(rejected["win"].mean()) if len(rejected) > 0 else 0.0

    return {
        "base_name":       base_name,
        "threshold":       threshold,
        "feature_names":   feature_names,
        "n_features":      len(feature_names),
        "base_signals":    base_n,
        "base_win_rate":   base_win_rate,
        "hc_signals":      hc_n,
        "hc_win_rate":     hc_win_rate,
        "hc_signal_rate":  hc_rate,
        "up_base_n":       int(up_mask.sum()),
        "up_base_wr":      up_base_wr,
        "down_base_n":     int(down_mask.sum()),
        "down_base_wr":    down_base_wr,
        "up_hc_n":         len(up_hc),
        "up_hc_wr":        up_hc_wr,
        "down_hc_n":       len(down_hc),
        "down_hc_wr":      down_hc_wr,
        "rejected_n":      len(rejected),
        "rejected_wr":     rej_wr,
    }


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Meta-Filter (2025 Vault)")
    parser.add_argument("--symbol",    default="NIFTY",  choices=["NIFTY","BANKNIFTY"])
    parser.add_argument("--direction", default="ALL",    choices=["ALL","UP","DOWN"])
    args = parser.parse_args()
    SYMBOL    = args.symbol
    DIRECTION = args.direction

    if DIRECTION == "UP":
        meta_model_path = f"{MODEL_DIR}/meta_filter_{SYMBOL.lower()}_up.joblib"
        apply_threshold = 0.60
        gate_win_rate   = 0.63
        gate_label      = "≥ 63% (handoff Section 13)"
    elif DIRECTION == "DOWN":
        meta_model_path = f"{MODEL_DIR}/meta_filter_{SYMBOL.lower()}_down.joblib"
        apply_threshold = 0.52
        gate_win_rate   = 0.65
        gate_label      = "≥ 65% (handoff Section 13)"
    else:
        meta_model_path = f"{MODEL_DIR}/meta_filter_ensemble_{SYMBOL.lower()}.joblib"
        apply_threshold = 0.55
        gate_win_rate   = 0.60
        gate_label      = "≥ 60%"

    print("=" * 70)
    print(f"🧪 META-FILTER VALIDATION — 2025 VAULT DATA")
    print(f"   Symbol: {SYMBOL}  |  Direction: {DIRECTION}  |  Threshold: {apply_threshold:.2f}")
    print("=" * 70)
    print()
    print(f"Vault period: {VAULT_START} to {VAULT_END}")
    print("This is OUT-OF-SAMPLE data — not used in any training or validation.")
    print()

    # Load vault data
    print(f"Loading 2025 vault data for {SYMBOL} from QuestDB…")
    vault_5m  = fetch_candles(SYMBOL, timeframe="5m",  start_date=VAULT_START, end_date=VAULT_END)
    vault_15m = fetch_candles(SYMBOL, timeframe="15m", start_date=VAULT_START, end_date=VAULT_END)

    if len(vault_5m) == 0 or len(vault_15m) == 0:
        print(f"\n❌ No 2025 vault data found for {SYMBOL}")
        print(f"  5m rows: {len(vault_5m)}  15m rows: {len(vault_15m)}")
        return

    feat_5m  = add_features(vault_5m).copy()
    feat_5m["label"] = label_movement(feat_5m)
    feat_5m  = feat_5m.dropna(subset=MOVEMENT_FEATURES + ["label","atr"]).reset_index(drop=True)
    feat_15m = add_features(vault_15m).dropna(subset=MOVEMENT_FEATURES).reset_index(drop=True)

    trading_days = pd.to_datetime(feat_5m["timestamp"]).dt.date.nunique()
    print(f"  5m rows:  {len(feat_5m):,}  ({feat_5m['timestamp'].min()} to {feat_5m['timestamp'].max()})")
    print(f"  15m rows: {len(feat_15m):,}")
    print(f"  Trading days in vault: {trading_days}")
    print()

    # Load models
    print("Loading models…")
    ensemble = EnsemblePredictor.load(MODEL_DIR, symbol=SYMBOL)

    if not Path(meta_model_path).exists():
        print(f"\n❌ Meta-filter not found: {meta_model_path}")
        cmd = f"python scripts/train_meta_filter.py --symbol {SYMBOL}"
        if DIRECTION != "ALL":
            cmd += f" --direction {DIRECTION}"
        print(f"  Run: {cmd}")
        return

    loaded_data = joblib.load(meta_model_path)
    if isinstance(loaded_data, dict) and "model" in loaded_data:
        meta_filter_model = loaded_data["model"]
        train_baseline    = loaded_data["metrics"]["cv_hc_prec_mean"]
        saved_direction   = loaded_data.get("direction", "ALL")
        n_features_saved  = loaded_data.get("n_features", "?")
        print(f"  ✅ Loaded: {meta_model_path}")
        print(f"  📊 Training CV HC Precision: {train_baseline:.1%}")
        print(f"  🏷️  Model direction: {saved_direction}  |  Saved features: {n_features_saved}")
        if saved_direction != DIRECTION:
            print(f"  ⚠️  WARNING: model direction={saved_direction} but --direction={DIRECTION}")
    else:
        meta_filter_model = loaded_data
        train_baseline    = 0.687
        print(f"  ✅ Loaded: {meta_model_path} (legacy format)")

    # Handle both XGBoost Booster and legacy dict models
    if isinstance(meta_filter_model, dict):
        if "feature_names" in meta_filter_model:
            model_feature_names = list(meta_filter_model["feature_names"])
        else:
            raise ValueError("Meta-filter model dict does not have 'feature_names' key.")
    elif hasattr(meta_filter_model, "get_booster"):
        model_feature_names = list(meta_filter_model.get_booster().feature_names)
    elif hasattr(meta_filter_model, "feature_names"):
        model_feature_names = list(meta_filter_model.feature_names)
    else:
        raise ValueError("Meta-filter model does not have feature_names or get_booster().feature_names")
    print(f"  🔍 Model expects {len(model_feature_names)} features: "
          f"{model_feature_names[:3]} … {model_feature_names[-2:]}")
    print()

    # Generate signals
    print("=" * 70)
    print("GENERATING ENSEMBLE SIGNALS ON 2025 VAULT")
    print("=" * 70)
    print()

    ens_signals = generate_ensemble_signals(feat_5m, feat_15m, ensemble)
    print(f"  Total ensemble signals: {len(ens_signals):,}")
    print(f"  UP:   {(ens_signals['direction']=='UP').sum():,}")
    print(f"  DOWN: {(ens_signals['direction']=='DOWN').sum():,}")

    if DIRECTION != "ALL":
        before      = len(ens_signals)
        ens_signals = ens_signals[ens_signals["direction"] == DIRECTION].reset_index(drop=True)
        print(f"\n  Filtered to {DIRECTION} signals: {len(ens_signals):,} (was {before:,})")
        if len(ens_signals) == 0:
            print(f"  ❌ No {DIRECTION} signals found. Cannot validate.")
            return

    print("\n  Labelling trade outcomes (WIN/LOSS)…")
    ens_signals["win"] = label_signal_outcomes(ens_signals, feat_5m)

    # Attach all 21 meta-features
    ens_signals = build_meta_features(ens_signals, feat_5m, SYMBOL)

    # Confirm all model features are present
    still_missing = [f for f in model_feature_names if f not in ens_signals.columns]
    if still_missing:
        print(f"\n❌ Still missing features after build_meta_features: {still_missing}")
        return
    print(f"  Meta-features attached ({len(model_feature_names)} model features all present).")
    print()

    # Validate
    print("=" * 70)
    print(f"APPLYING META-FILTER  (threshold: {apply_threshold:.2f})")
    print("=" * 70)
    print()

    results = validate_meta_filter(
        ens_signals, meta_filter_model, "Ensemble", threshold=apply_threshold
    )

    signals_per_day_base = results["base_signals"] / trading_days if trading_days > 0 else 0.0
    signals_per_day_hc   = results["hc_signals"]   / trading_days if trading_days > 0 else 0.0

    print(f"BASE (no meta-filter):")
    print(f"  Total signals:     {results['base_signals']:,}  ({signals_per_day_base:.1f}/day)")
    print(f"  Base win rate:     {results['base_win_rate']:.1%}")
    print(f"  UP signals:        {results['up_base_n']:,}  (win rate: {results['up_base_wr']:.1%})")
    print(f"  DOWN signals:      {results['down_base_n']:,}  (win rate: {results['down_base_wr']:.1%})")
    print()

    print(f"META-FILTER @ {apply_threshold:.2f}:")
    print(f"  Approved signals:  {results['hc_signals']:,}  ({results['hc_signal_rate']:.1%} of base)")
    print(f"  Signals per day:   {signals_per_day_hc:.1f}/day  ← check ≥ 4.0/day gate")
    print(f"  HC win rate:       {results['hc_win_rate']:.1%}  ← KEY METRIC")
    if results["up_hc_n"] > 0:
        print(f"  UP approved:       {results['up_hc_n']:,}  (win rate: {results['up_hc_wr']:.1%})")
    if results["down_hc_n"] > 0:
        print(f"  DOWN approved:     {results['down_hc_n']:,}  (win rate: {results['down_hc_wr']:.1%})")
    print()

    print(f"REJECTED SIGNALS:")
    print(f"  Rejected count:    {results['rejected_n']:,}")
    print(f"  Rejected win rate: {results['rejected_wr']:.1%}")
    print(f"  Δ (HC - Rejected): {results['hc_win_rate'] - results['rejected_wr']:.1%}  (filter edge)")
    print()

    # Decision gate
    print("=" * 70)
    print("DECISION GATE")
    print("=" * 70)
    print()

    hc_wr      = results["hc_win_rate"]
    gate_pass  = hc_wr >= gate_win_rate
    count_pass = signals_per_day_hc >= 4.0

    if gate_pass and count_pass:
        print(f"  ✅ VALIDATED — ready for deployment")
        print(f"     HC Win Rate:     {hc_wr:.1%}  (gate: {gate_label}  ✅)")
        print(f"     Signals per day: {signals_per_day_hc:.1f}  (gate: ≥ 4.0/day  ✅)")
        print()
        print("  Deploy directly (shadow mode skipped per decision).")

    elif gate_pass and not count_pass:
        print(f"  ⚠️  WIN RATE PASSES but signal count is low")
        print(f"     HC Win Rate:     {hc_wr:.1%}  (gate: {gate_label}  ✅)")
        print(f"     Signals per day: {signals_per_day_hc:.1f}  (gate: ≥ 4.0/day  ❌)")
        print()
        print("  Consider lowering threshold slightly. Hard floor is 0.45.")

    elif not gate_pass and hc_wr >= 0.55:
        print(f"  🟡 MARGINAL — HC Win Rate: {hc_wr:.1%}  (gate: {gate_label})")
        print()
        print("  Options: deploy at 0.25% risk, lower threshold, or retrain.")

    else:
        print(f"  ❌ FAILED — HC Win Rate: {hc_wr:.1%}  (gate: {gate_label})")
        print()
        print("  DO NOT deploy. Investigate regime shift or overfitting.")

    print()
    print("=" * 70)
    print("COMPARISON TO TRAINING BASELINE")
    print("=" * 70)
    print()
    print(f"  Features used:            {results['n_features']}")
    print(f"  Training CV HC Precision: {train_baseline:.1%}")
    print(f"  2025 Vault HC Win Rate:   {hc_wr:.1%}")
    print(f"  Gap:                      {train_baseline - hc_wr:+.1%}")
    print()
    gap = abs(train_baseline - hc_wr)
    if gap > 0.10:
        print("  ⚠️  Gap > 10pp — possible regime shift or overfitting.")
    elif gap > 0.05:
        print("  🟡 Gap 5–10pp — acceptable variance for a different year.")
    else:
        print("  ✅ Gap < 5pp — strong generalisation across regimes.")

    print()
    print("=" * 70)
    print(f"VAULT VALIDATION COMPLETE — {SYMBOL} {DIRECTION}")
    print("=" * 70)


if __name__ == "__main__":
    main()


# =============================================================================
# CHANGES
# =============================================================================
# Root cause of "feature_names mismatch" / "expected jsd, bn_nifty_ratio_zscore
# in input data" error:
#
#   validate_meta_filter() was building X using a hardcoded META_FEATURES list
#   of 19 names. The newly retrained 21-feature models store their feature list
#   inside the XGBoost booster. XGBoost validates feature names on every
#   predict_proba() call and raises ValueError if names or count do not match.
#
# Fix 1 — validate_meta_filter() uses model.get_booster().feature_names:
#   feature_names = list(meta_filter.get_booster().feature_names)
#   X = signals[feature_names].astype(float)
#   This selects exactly the right features in the right order regardless of
#   whether the model has 19 or 21 features. The hardcoded META_FEATURES
#   constant has been removed from this file entirely.
#
# Fix 2 — build_meta_features() now computes Features #20 and #21:
#   - Added `symbol` parameter.
#   - Feature #20 (jsd): computed from p_down_5m / p_neutral_5m / p_up_5m
#     and p_down_15m / p_neutral_15m / p_up_15m columns. These columns are
#     now preserved by generate_ensemble_signals() (they were computed but
#     previously dropped before the signal mask was applied).
#   - Feature #21 (bn_nifty_ratio_zscore): computed by _add_ratio_zscore(),
#     which loads the companion symbol's vault data from QuestDB and computes
#     the rolling-20 BankNifty/NIFTY ratio Z-score. Falls back to 0.0 on
#     any error.
#
# Fix 3 — generate_ensemble_signals() now preserves p_down_5m, p_neutral_5m,
#   p_up_5m, p_down_15m, p_neutral_15m, p_up_15m on the returned DataFrame.
#   The veto logic is unchanged.
#
# Fix 4 — A pre-flight check after build_meta_features() prints any features
#   that are still missing before calling validate_meta_filter(), providing
#   a clear error message instead of an XGBoost stack trace.
#
# NO MODEL IMPACT. Read-only validation script.
# All .joblib files remain unchanged after running this script.
# =============================================================================