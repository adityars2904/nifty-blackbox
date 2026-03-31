"""
Meta-filter service — wraps the meta-filter joblib models and
computes meta-features needed to score each signal.

Feature count is determined at runtime by reading the loaded model's own
feature_names via model.get_booster().feature_names. This means the service
works correctly whether a 19-feature or 21-feature model is loaded — no
code change is needed when models are retrained with additional features.

Simplified for chart-only / research mode: all database-dependent features
(recent_win_rate, consecutive_wins/losses, signals_today, drawdown, etc.)
use neutral defaults. The model loading, feature dict construction, and
predict() function are kept intact for research script imports.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from scipy.stats import entropy

from config import (
    IST,
    MARKET_OPEN,
    META_FILTER_PROB_MIN,
    MODELS_DIR,
    STARTING_CAPITAL,
    META_FILTER_MODE,
)

logger = logging.getLogger(__name__)


# ============================================================================
# MODEL CACHE
# ============================================================================

_meta_models: dict[str, Any]    = {}   # pooled: NIFTY, BANKNIFTY
_meta_models_3m: dict[str, Any] = {}   # 3-model variants
_meta_dir_models: dict[str, Any] = {}  # directional: NIFTY_UP, NIFTY_DOWN etc.
_active_mode: str = META_FILTER_MODE   # "2model" or "3model"


def _load_one(path: Path) -> Any:
    """Load a joblib file and extract the model object from the payload dict."""
    raw = joblib.load(path)
    return raw["model"] if isinstance(raw, dict) and "model" in raw else raw


def load_meta_models() -> dict[str, Any]:
    """
    Load all meta-filter models at startup.

    Loading order:
      1. Pooled 2-model (required)      → meta_filter_ensemble_{symbol}.joblib
      2. Pooled 3-model (optional)      → meta_filter_3model_{symbol}.joblib
      3. Directional models (optional)  → meta_filter_{symbol}_{up|down}.joblib

    Directional models are loaded into _meta_dir_models keyed as
    "NIFTY_UP", "NIFTY_DOWN", "BANKNIFTY_UP", "BANKNIFTY_DOWN".
    If all four directional models are present, predict() will route to them
    automatically. If any are missing, predict() falls back to the pooled model.
    """
    global _meta_models, _meta_models_3m, _meta_dir_models

    model_dir = Path(MODELS_DIR)

    # ── Pooled 2-model (required) ─────────────────────────────────────────────
    for symbol in ("nifty", "banknifty"):
        path = model_dir / f"meta_filter_ensemble_{symbol}.joblib"
        if not path.exists():
            raise FileNotFoundError(f"Required meta-filter not found: {path}")
        _meta_models[symbol.upper()] = _load_one(path)
        n_feat = len(_meta_models[symbol.upper()].get_booster().feature_names)
        logger.info("Loaded pooled 2-model meta-filter: %s (%d features)", path.name, n_feat)

    # ── Pooled 3-model (optional) ─────────────────────────────────────────────
    for symbol in ("nifty", "banknifty"):
        path = model_dir / f"meta_filter_3model_{symbol}.joblib"
        if path.exists():
            _meta_models_3m[symbol.upper()] = _load_one(path)
            n_feat = len(_meta_models_3m[symbol.upper()].get_booster().feature_names)
            logger.info("Loaded 3-model meta-filter: %s (%d features)", path.name, n_feat)

    # ── Directional models (optional) ─────────────────────────────────────────
    dir_paths = {
        "NIFTY_UP":      model_dir / "meta_filter_nifty_up.joblib",
        "NIFTY_DOWN":    model_dir / "meta_filter_nifty_down.joblib",
        "BANKNIFTY_UP":  model_dir / "meta_filter_banknifty_up.joblib",
        "BANKNIFTY_DOWN": model_dir / "meta_filter_banknifty_down.joblib",
    }
    loaded_dir = []
    for key, path in dir_paths.items():
        if path.exists():
            _meta_dir_models[key] = _load_one(path)
            n_feat = len(_meta_dir_models[key].get_booster().feature_names)
            logger.info("Loaded directional meta-filter: %s (%d features)", path.name, n_feat)
            loaded_dir.append(key)

    if len(loaded_dir) == 4:
        logger.info("All 4 directional models loaded — predict() will route directionally.")
    elif loaded_dir:
        logger.info(
            "Partial directional models loaded (%s) — falling back to pooled for missing.",
            loaded_dir,
        )

    return _meta_models


def get_active_mode() -> str:
    return _active_mode


def set_active_mode(mode: str) -> str:
    global _active_mode
    if mode not in ("2model", "3model"):
        raise ValueError(f"Invalid mode '{mode}'. Must be '2model' or '3model'.")
    if mode == "3model" and not _meta_models_3m:
        raise ValueError(
            "3-model meta-filter not trained. Run: "
            "python scripts/train_meta_filter.py --symbol NIFTY --models 3"
        )
    _active_mode = mode
    logger.info("Meta-filter mode switched to: %s", mode)
    return _active_mode


def has_3model() -> bool:
    return bool(_meta_models_3m)


def has_directional_models() -> bool:
    """Return True if all four direction-specific models are loaded."""
    return len(_meta_dir_models) == 4


# ============================================================================
# META-FEATURE COMPUTATION
# ============================================================================

def _session_phase(now: datetime) -> int:
    t = now.hour * 60 + now.minute
    if t < 11 * 60:
        return 0
    if t < 13 * 60 + 30:
        return 1
    return 2


def _minutes_in_session(now: datetime) -> int:
    mkt = MARKET_OPEN.hour * 60 + MARKET_OPEN.minute
    cur = now.hour * 60 + now.minute
    return max(0, cur - mkt)


def _compute_jsd(probs_5m: Optional[tuple], probs_15m: Optional[tuple]) -> float:
    """JSD between 5m and 15m distributions, normalised to [0,1] by log(2)."""
    if probs_5m is None or probs_15m is None:
        return 0.0
    p = np.clip(np.array(probs_5m,  dtype=float), 1e-10, None)
    q = np.clip(np.array(probs_15m, dtype=float), 1e-10, None)
    p /= p.sum(); q /= q.sum()
    m = 0.5 * (p + q)
    return float((0.5 * entropy(p, m) + 0.5 * entropy(q, m)) / np.log(2))


def _build_feature_dict(
    symbol: str,
    ensemble_probs: tuple[float, float, float],
    ensemble_direction: str,
    ensemble_confidence: float,
    atr_current: float,
    atr_avg_20: float,
    probs_5m: Optional[tuple],
    probs_15m: Optional[tuple],
    bn_nifty_ratio_zscore: float,
    *,
    recent_win_rate: float = 0.5,
    consecutive_losses: int = 0,
    consecutive_wins: int = 0,
    signals_today: int = 0,
    time_since_last_signal: float = 30.0,
    current_drawdown: float = 0.0,
    daily_pnl_points: float = 0.0,
    open_position_flag: int = 0,
    avg_meta_prob: float = 0.5,
) -> dict[str, float]:
    """
    Compute all possible meta-features and return them as a named dict.

    The predict() function then selects only the features the loaded model
    expects (via model.get_booster().feature_names). This means new features
    are computed but silently ignored by older 19-feature models, and are
    used automatically once 21-feature models are loaded.

    Database-dependent features (recent_win_rate, consecutive_wins/losses,
    signals_today, drawdown, etc.) accept keyword arguments with neutral
    defaults. Research scripts can override these when simulating trades.
    """
    now = datetime.now(IST)
    prob_down, prob_neutral, prob_up = ensemble_probs

    # atr_regime
    atr_regime = (atr_current / atr_avg_20) if atr_avg_20 > 0 else 1.0

    # Feature #20: JSD
    jsd = _compute_jsd(probs_5m, probs_15m)

    return {
        # ── Original 19 ──────────────────────────────────────────────────────
        "ens_confidence":          ensemble_confidence,
        "conf_5m":                 float(max(probs_5m))  if probs_5m  else ensemble_confidence,
        "conf_15m":                float(max(probs_15m)) if probs_15m else ensemble_confidence,
        "prob_gap":                float(sorted([prob_down, prob_neutral, prob_up])[-1]
                                         - sorted([prob_down, prob_neutral, prob_up])[-2]),
        "p_neutral":               prob_neutral,
        "vol_ratio":               1.0,
        "vol_expansion":           1.0,
        "regime_trend":            0.0,
        "z_score_20":              0.0,
        "close_position":          0.5,
        "z_score_distance_from_vwap": 0.0,
        "time_sin":                float(np.sin(2 * np.pi * _minutes_in_session(now) / 375)),
        "regime_time_of_day":      float(_session_phase(now)),
        "recent_win_rate_5":       recent_win_rate,
        "recent_win_rate_10":      recent_win_rate,
        "consecutive_losses":      float(consecutive_losses),
        "consecutive_wins":        float(consecutive_wins),
        "signals_today":           float(signals_today),
        "bars_since_last_signal":  time_since_last_signal,
        # ── Feature #20 ───────────────────────────────────────────────────────
        "jsd":                     jsd,
        # ── Feature #21 ───────────────────────────────────────────────────────
        "bn_nifty_ratio_zscore":   float(bn_nifty_ratio_zscore),
    }


def compute_meta_features(
    symbol: str,
    ensemble_probs: tuple[float, float, float],
    ensemble_direction: str,
    ensemble_confidence: float,
    atr_current: float,
    atr_avg_20: float,
    probs_5m: Optional[tuple],
    probs_15m: Optional[tuple],
    bn_nifty_ratio_zscore: float,
    feature_names: Optional[list[str]] = None,
    **kwargs,
) -> np.ndarray:
    """
    Build the feature matrix for the meta-filter.

    If feature_names is provided (from model.get_booster().feature_names),
    the returned array contains exactly those features in that order.
    Otherwise returns all features in META_FEATURES order (legacy path).

    Extra kwargs are forwarded to _build_feature_dict() to allow research
    scripts to override database-dependent features.

    Returns shape (1, n_features).
    """
    feat_dict = _build_feature_dict(
        symbol, ensemble_probs, ensemble_direction, ensemble_confidence,
        atr_current, atr_avg_20, probs_5m, probs_15m, bn_nifty_ratio_zscore,
        **kwargs,
    )

    if feature_names is None:
        # Fallback: use all keys in insertion order
        feature_names = list(feat_dict.keys())

    row = np.array([[feat_dict.get(f, 0.0) for f in feature_names]], dtype=float)
    return row


def predict(
    symbol: str,
    ensemble_probs: tuple[float, float, float],
    ensemble_direction: str,
    ensemble_confidence: float,
    atr_current: float,
    atr_avg_20: float,
    probs_5m: Optional[tuple] = None,
    probs_15m: Optional[tuple] = None,
    bn_nifty_ratio_zscore: float = 0.0,
    threshold_override: Optional[float] = None,
    **kwargs,
) -> tuple[float, bool]:
    """
    Run the meta-filter and return (win_probability, approved).

    Model selection priority:
      1. Directional models (NIFTY_UP / NIFTY_DOWN etc.) if all four loaded
      2. 3-model pooled if _active_mode == "3model"
      3. 2-model pooled (default fallback)

    Feature vector is built using model.get_booster().feature_names so the
    correct subset and order is always used regardless of model version.

    Extra kwargs are forwarded to compute_meta_features() → _build_feature_dict()
    to allow research scripts to override database-dependent features.
    """
    # ── Select model ──────────────────────────────────────────────────────────
    dir_key = f"{symbol}_{ensemble_direction}"
    if has_directional_models() and dir_key in _meta_dir_models:
        model      = _meta_dir_models[dir_key]
        mode_label = f"directional_{ensemble_direction.lower()}"
    elif _active_mode == "3model" and symbol in _meta_models_3m:
        model      = _meta_models_3m[symbol]
        mode_label = "3model_pooled"
    elif symbol in _meta_models:
        model      = _meta_models[symbol]
        mode_label = "2model_pooled"
    else:
        raise KeyError(f"No meta-filter model loaded for {symbol}")

    # ── Build feature vector using model's own feature names ──────────────────
    feature_names = list(model.get_booster().feature_names)
    features      = compute_meta_features(
        symbol, ensemble_probs, ensemble_direction, ensemble_confidence,
        atr_current, atr_avg_20, probs_5m, probs_15m, bn_nifty_ratio_zscore,
        feature_names=feature_names,
        **kwargs,
    )

    # ── Predict ───────────────────────────────────────────────────────────────
    proba    = model.predict_proba(features)
    win_prob = float(proba[0][1])

    # ── Apply threshold ───────────────────────────────────────────────────────
    if threshold_override is not None:
        threshold = threshold_override
    else:
        threshold = META_FILTER_PROB_MIN

    approved = win_prob >= threshold

    logger.info(
        "%s meta-filter [%s, %d-feat]: WIN_prob=%.4f threshold=%.2f → %s",
        symbol, mode_label, len(feature_names), win_prob, threshold,
        "APPROVED" if approved else "REJECTED",
    )

    return win_prob, approved