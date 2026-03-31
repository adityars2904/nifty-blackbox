#!/usr/bin/env python3
"""
Train 30-minute timeframe movement predictors for NIFTY and BANKNIFTY.

Architecture identical to the existing 5m/15m models:
  - XGBoost 3-class classifier (UP / DOWN / NEUTRAL)
  - Same 17 features from add_features()
  - Same labeling from labels.py
  - Training period: April 2022 – April 2024 (matching existing models)

Usage:
    cd backend && python -m training.train_multihorizon
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

# Ensure backend/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import IST, MODELS_DIR
from adapters.questdb_adapter import init_pool, fetch_candles_range, resample_to_30m
from ml.feature_engineering import add_features, MOVEMENT_FEATURES
from ml.labels import label_movement

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ============================================================================
# HYPERPARAMETERS (matching existing 5m/15m models)
# ============================================================================
XGB_PARAMS = dict(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    objective="multi:softprob",
    num_class=3,
    eval_metric="mlogloss",
    use_label_encoder=False,
    random_state=42,
    n_jobs=-1,
)

# Training period
TRAIN_START = datetime(2022, 4, 1, 9, 15, tzinfo=IST)
TRAIN_END = datetime(2024, 4, 30, 15, 30, tzinfo=IST)

# Walk-forward CV
N_SPLITS = 5


def _fetch_and_prepare(symbol: str) -> pd.DataFrame:
    """Fetch 1m candles, resample to 30m, add features and labels."""
    logger.info("Fetching %s candles from %s to %s …",
                symbol, TRAIN_START.date(), TRAIN_END.date())

    df_1m = fetch_candles_range(symbol, TRAIN_START, TRAIN_END)
    if df_1m.empty:
        raise RuntimeError(f"No candles returned for {symbol}")
    logger.info("  Fetched %d 1m candles", len(df_1m))

    df_30m = resample_to_30m(df_1m)
    logger.info("  Resampled to %d 30m candles", len(df_30m))

    df_feat = add_features(df_30m)
    labels = label_movement(df_30m)
    df_feat["label"] = labels

    # Drop NaN rows (from rolling features and forward-looking labels)
    df_feat = df_feat.dropna(subset=MOVEMENT_FEATURES + ["label"])
    logger.info("  After dropna: %d rows", len(df_feat))

    return df_feat


def _train_model(df: pd.DataFrame, symbol: str) -> XGBClassifier:
    """Train XGBoost with walk-forward cross-validation."""
    X = df[MOVEMENT_FEATURES].astype(float).values
    y = df["label"].astype(int).values

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    fold_accuracies = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        clf = XGBClassifier(**XGB_PARAMS)
        clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        val_acc = clf.score(X_val, np.asarray(y_val))
        fold_accuracies.append(val_acc)
        logger.info("  %s fold %d/%d: val accuracy = %.4f",
                     symbol, fold + 1, N_SPLITS, val_acc)

    mean_acc = np.mean(fold_accuracies)
    logger.info("  %s mean CV accuracy: %.4f", symbol, mean_acc)

    # Final model trained on all data
    final_clf = XGBClassifier(**XGB_PARAMS)
    final_clf.fit(X, y, verbose=False)

    return final_clf


def main() -> None:
    init_pool()

    for symbol in ("NIFTY", "BANKNIFTY"):
        logger.info("=" * 60)
        logger.info("Training 30m model for %s", symbol)
        logger.info("=" * 60)

        df = _fetch_and_prepare(symbol)
        model = _train_model(df, symbol)

        out_path = Path(MODELS_DIR) / f"movement_predictor_{symbol.lower()}_30m.joblib"
        joblib.dump(model, out_path)
        logger.info("Saved: %s", out_path)

    logger.info("Done — both 30m models trained.")


if __name__ == "__main__":
    main()
