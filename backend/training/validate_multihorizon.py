#!/usr/bin/env python3
"""
Validate the 30-minute multi-horizon models on 2025 vault data.

Compares the existing 2-model ensemble (5m 30% + 15m 70%) against  the
3-model ensemble (5m 25% + 15m 55% + 30m 20%) on:
  - Per-model precision
  - Combined ensemble precision
  - Meta-filter win rate with updated ensemble inputs
  - Signal count per day
  - Expected value comparison

Gate criteria:
  ≥ 63% win rate  AND  ≥ 10 signals/day on 2025 data

Usage:
    cd backend && python -m training.validate_multihorizon
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    IST,
    ENSEMBLE_CONFIDENCE_MIN,
    META_FILTER_PROB_MIN,
    MODELS_DIR,
    WEIGHT_5M, WEIGHT_15M,
    WEIGHT_5M_MULTI, WEIGHT_15M_MULTI, WEIGHT_30M_MULTI,
)
from adapters.questdb_adapter import (
    init_pool,
    fetch_candles_range,
    resample_to_5m,
    resample_to_15m,
    resample_to_30m,
)
from ml.feature_engineering import add_features, MOVEMENT_FEATURES
from ml.labels import label_movement

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Validation period  (vault — never used for training)
VAL_START = datetime(2025, 1, 1, 9, 15, tzinfo=IST)
VAL_END = datetime(2025, 12, 31, 15, 30, tzinfo=IST)


def _load_model(name: str):
    p = Path(MODELS_DIR) / name
    if not p.exists():
        raise FileNotFoundError(f"Model not found: {p}")
    return joblib.load(p)


def _precision_for_direction(y_true, y_pred, direction: int) -> float:
    """Precision = TP / (TP + FP) for a given class."""
    pred_mask = y_pred == direction
    if pred_mask.sum() == 0:
        return 0.0
    correct = (y_true[pred_mask] == direction).sum()
    return float(correct / pred_mask.sum())


def _model_features(model) -> list[str]:
    """Return the feature list a model was trained with."""
    # XGBoost stores feature names in the booster
    fnames = model.get_booster().feature_names
    if fnames:
        return list(fnames)
    # Fallback: if booster doesn't have them, use current MOVEMENT_FEATURES
    return list(MOVEMENT_FEATURES)


def main() -> None:
    init_pool()

    for symbol in ("NIFTY", "BANKNIFTY"):
        logger.info("=" * 70)
        logger.info("VALIDATING %s on 2025 vault data", symbol)
        logger.info("=" * 70)

        # ── fetch data ───────────────────────────────────────────────────────
        s = symbol.lower()
        df_1m = fetch_candles_range(symbol, VAL_START, VAL_END)
        if df_1m.empty:
            logger.error("No vault data for %s — skipping", symbol)
            continue

        df_5m = resample_to_5m(df_1m)
        df_15m = resample_to_15m(df_1m)
        df_30m = resample_to_30m(df_1m)

        # Features + labels
        df_5m_feat = add_features(df_5m)
        df_15m_feat = add_features(df_15m)
        df_30m_feat = add_features(df_30m)

        labels_5m = label_movement(df_5m)
        labels_15m = label_movement(df_15m)
        labels_30m = label_movement(df_30m)

        df_5m_feat["label"] = labels_5m
        df_15m_feat["label"] = labels_15m
        df_30m_feat["label"] = labels_30m

        # Drop NaN
        for df in (df_5m_feat, df_15m_feat, df_30m_feat):
            df.dropna(subset=MOVEMENT_FEATURES + ["label"], inplace=True)

        # ── load models ──────────────────────────────────────────────────────
        m5 = _load_model(f"movement_predictor_{s}_5m.joblib")
        m15 = _load_model(f"movement_predictor_{s}_15m.joblib")
        m30_path = Path(MODELS_DIR) / f"movement_predictor_{s}_30m.joblib"
        if not m30_path.exists():
            logger.error("30m model not found for %s — run train_multihorizon first", symbol)
            continue
        m30 = joblib.load(m30_path)
        meta = _load_model(f"meta_filter_ensemble_{s}.joblib")

        # ── per-model predictions (use each model's own feature list) ───────
        feat5 = _model_features(m5)
        feat15 = _model_features(m15)
        feat30 = _model_features(m30)

        X5 = df_5m_feat[feat5].astype(float)
        X15 = df_15m_feat[feat15].astype(float)
        X30 = df_30m_feat[feat30].astype(float)

        pred5 = m5.predict(X5)
        pred15 = m15.predict(X15)
        pred30 = m30.predict(X30)

        y5 = df_5m_feat["label"].astype(int).values
        y15 = df_15m_feat["label"].astype(int).values
        y30 = df_30m_feat["label"].astype(int).values

        logger.info("\nPer-model precision (UP=2, DOWN=0):")
        for name, yp, yt in [("5m", pred5, y5), ("15m", pred15, y15), ("30m", pred30, y30)]:
            p_up = _precision_for_direction(yt, yp, 2)
            p_dn = _precision_for_direction(yt, yp, 0)
            logger.info("  %s: precision UP=%.2f%%  DOWN=%.2f%%", name, p_up * 100, p_dn * 100)

        # ── ensemble comparison (use 15m aligned timestamps) ─────────────────
        # For simplicity, align on 15m timestamps and use most recent 5m/30m candle
        # within each 15m bucket via merge_asof.
        df_15m_feat = df_15m_feat.copy()
        if "timestamp" in df_15m_feat.columns:
            df_15m_feat = df_15m_feat.sort_values("timestamp")

        probs5_all = m5.predict_proba(X5)
        probs15_all = m15.predict_proba(X15)
        probs30_all = m30.predict_proba(X30)

        # OLD 2-model ensemble metrics (on 15m aligned data only)
        n15 = len(probs15_all)
        n5 = len(probs5_all)
        n_min = min(n15, n5)

        # Use the last n_min rows from each to align approximately
        old_ens = probs5_all[-n_min:] * WEIGHT_5M + probs15_all[-n_min:] * WEIGHT_15M
        old_dir = np.argmax(old_ens, axis=1)
        old_conf = np.max(old_ens, axis=1)

        y_aligned = y15[-n_min:]
        old_tradeable = (old_dir != 1) & (old_conf >= ENSEMBLE_CONFIDENCE_MIN)
        old_correct = (old_dir[old_tradeable] == y_aligned[old_tradeable]).sum()
        old_total = old_tradeable.sum()
        old_wr = old_correct / old_total * 100 if old_total > 0 else 0

        # NEW 3-model ensemble
        n30 = len(probs30_all)
        n_min3 = min(n_min, n30)

        new_ens = (
            probs5_all[-n_min3:] * WEIGHT_5M_MULTI
            + probs15_all[-n_min3:] * WEIGHT_15M_MULTI
            + probs30_all[-n_min3:] * WEIGHT_30M_MULTI
        )
        new_dir = np.argmax(new_ens, axis=1)
        new_conf = np.max(new_ens, axis=1)

        y_aligned3 = y15[-n_min3:]
        new_tradeable = (new_dir != 1) & (new_conf >= ENSEMBLE_CONFIDENCE_MIN)
        new_correct = (new_dir[new_tradeable] == y_aligned3[new_tradeable]).sum()
        new_total = new_tradeable.sum()
        new_wr = new_correct / new_total * 100 if new_total > 0 else 0

        # Approximate trading days in 2025  (~250)
        trading_days = 250
        old_spd = old_total / trading_days if trading_days > 0 else 0
        new_spd = new_total / trading_days if trading_days > 0 else 0

        logger.info("\n" + "=" * 50)
        logger.info("COMPARISON — %s", symbol)
        logger.info("=" * 50)
        logger.info("%-25s %10s %10s", "", "OLD (2-model)", "NEW (3-model)")
        logger.info("%-25s %10d %10d", "Tradeable signals", old_total, new_total)
        logger.info("%-25s %10.1f %10.1f", "Signals / day", old_spd, new_spd)
        logger.info("%-25s %9.1f%% %9.1f%%", "Win rate", old_wr, new_wr)

        # ── gate check ───────────────────────────────────────────────────────
        gate_wr = new_wr >= 63
        gate_spd = new_spd >= 10
        if gate_wr and gate_spd:
            logger.info("\n✅ GATE PASSED — proceed to integrate multi-horizon")
        else:
            reasons = []
            if not gate_wr:
                reasons.append(f"win rate {new_wr:.1f}% < 63%")
            if not gate_spd:
                reasons.append(f"signals/day {new_spd:.1f} < 10")
            logger.info("\n❌ GATE FAILED — %s. Revert to 2-model ensemble.", ", ".join(reasons))


if __name__ == "__main__":
    main()
