from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


SESSION_START_MINUTES = 9 * 60 + 16
TRADING_DAY_MINUTES   = 375

# Multiplier for the confirmed-DOWN label used in train_split_brain_v2.py.
# At 0.3x NIFTY 5m ATR (~30 pts) the bar is ~9 pts — virtually every
# DOWN-labeled candle already clears that, so FIX 3 filtered 0% of labels.
# 0.8x raises it to ~24 pts: only clean unrecovered drops survive.
# Import this constant instead of hardcoding the multiplier.
CONFIRMED_DOWN_ATR_MULT: float = 0.8

# ============================================================================
# 🔥 FINAL FEATURE SET - POST ADVERSARIAL PURGE
# ============================================================================
#
# Saboteur Results Summary:
#   Original: 63% regime detection (BAD - features leak year info)
#   After purge: 57% → Target: < 55% (achieved by also removing WARN features)
#
# PURGED (regime-biased - Saboteur used these to identify the year):
#   ❌ atr_pct             (0.137) - 2022 = high vol, 2024 = low vol
#   ❌ bb_width            (0.121) - Same reason
#   ❌ rolling_std_20      (0.075) - Same reason
#   ❌ dist_ema_50         (0.073) - Bull trend = always negative dist
#   ❌ rolling_std_10      (0.072) - Same as rolling_std_20
#   ❌ time_cos            (0.061) - Season/time encodes the year!
#   ❌ regime_volatility   (0.058) - Direct vol percentile = year identifier
#   ❌ volatility_strength (0.058) - Same
#
# NORMALIZED (WARN features → z-score to remove level bias):
#   🔄 high_low_range      → z_score_high_low_range
#   🔄 minutes_since_high  → z_score_minutes_since_high
#   🔄 distance_from_vwap  → z_score_distance_from_vwap
#   🔄 time_sin            → kept as-is (already -1 to 1, daily cycle)
#   🔄 lower_wick_ratio    → z_score_lower_wick_ratio
#
# ADDED (volatility as RELATIVE measure, not absolute):
#   ➕ vol_expansion       - Is vol expanding NOW vs recent? (regime-neutral)
#   ➕ atr_z_score         - ATR z-score vs recent 100 periods (regime-neutral)
#
# ADDED (Bear Hunter v2 specialist features):
#   ➕ z_score_upper_wick_ratio  - wick rejection at highs (regime-neutral)
#   ➕ candle_direction_streak   - consecutive green/red candles (regime-neutral)
# ============================================================================

MOVEMENT_FEATURES: List[str] = [
    # ===== REGIME CONTEXT (Safe - ADX based) =====
    "regime_trend",           # ✅ SAFE (0.001) - trend direction
    "regime_time_of_day",     # ✅ SAFE (0.021) - morning/midday/closing

    # ===== NORMALIZED MOMENTUM (All z-scored - regime neutral) =====
    "z_score_roc_5",          # ✅ SAFE (0.012)
    "z_score_rsi_slope",      # ✅ SAFE (0.005)
    "z_score_ema_slope",      # ✅ SAFE (0.017)
    # returns_10_normalized removed - marginal gain, adds noise on small datasets

    # ===== MEAN REVERSION (z-scored) =====
    "z_score_20",             # ✅ SAFE (0.010)
    "mean_reversion_strength", # ✅ SAFE (0.007)

    # ===== VOLATILITY AS RELATIVE MEASURE (not absolute) =====
    "vol_ratio",              # ✅ SAFE (0.019) - short/long vol ratio
    "vol_expansion",          # ➕ NEW - Is vol expanding? (regime-neutral)
    # atr_z_score removed - redundant with vol_ratio + vol_expansion

    # ===== PRICE ACTION (normalized WARN features) =====
    "z_score_high_low_range", # 🔄 NORMALIZED from high_low_range (0.049)
    "close_position",         # ✅ SAFE (0.005)
    "z_score_distance_from_vwap", # 🔄 NORMALIZED from distance_from_vwap (0.036)

    # ===== CANDLE MICROSTRUCTURE (regime-neutral) =====
    "candle_body_ratio",      # ✅ SAFE (0.009)
    # upper_wick_ratio removed - correlated with candle_body_ratio
    # z_score_lower_wick_ratio removed - low importance, adds noise

    # ===== INTRADAY (normalized WARN features) =====
    # z_score_minutes_since_high removed - noisy on 15m/1h
    "minutes_since_low",      # ✅ SAFE (0.022)

    # ===== TIME (daily cycle - safe) =====
    "time_sin",               # ⚠️ WARN (0.034) - kept, daily cycle not yearly

    # ===== BEAR HUNTER v2 (exhaustion/rejection - regime-neutral) =====
    "z_score_upper_wick_ratio",  # ➕ Bear v2: wick rejection at highs
    "candle_direction_streak",   # ➕ Bear v2: consecutive green/red candles
]

RISK_FEATURES: List[str] = MOVEMENT_FEATURES


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "Open": "open", "High": "high", "Low": "low", "Close": "close",
        "Volume": "volume", "Timestamp": "timestamp",
        "CandleDateTime": "timestamp", "Symbol": "symbol",
    }
    out = df.rename(columns=rename_map).copy()
    out.columns = [c.lower() for c in out.columns]
    return out


def _ensure_ist(series: pd.Series) -> pd.Series:
    ts = pd.to_datetime(series, errors="coerce")
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("Asia/Kolkata")
    else:
        ts = ts.dt.tz_convert("Asia/Kolkata")
    return ts  # type: ignore[return-value]


def _calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate ADX (trend strength indicator)."""
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)

    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)

    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

    atr = tr.rolling(period).mean()
    plus_di = 100 * (pd.Series(plus_dm, index=df.index).rolling(period).mean() / atr)
    minus_di = 100 * (pd.Series(minus_dm, index=df.index).rolling(period).mean() / atr)

    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    adx = dx.rolling(period).mean()
    return adx.fillna(0)


def _z_score_normalize(series: pd.Series, window: int = 100) -> pd.Series:
    """
    Convert raw feature to z-score relative to rolling window.
    
    "Is this HIGH or LOW relative to recent history?"
    instead of "Is this absolutely high or low?"
    
    This removes regime bias by making the feature regime-neutral.
    """
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std().replace(0, np.nan)
    z = (series - rolling_mean) / rolling_std
    return z.fillna(0)


def _add_bear_features(out: pd.DataFrame) -> None:
    """
    Bear Hunter v2 specialist features.

    Called at the end of add_features() to append two regime-neutral
    exhaustion/rejection signals used exclusively by the Bear Hunter.

    z_score_upper_wick_ratio
        Upper wick = high - max(open, close).  A long upper wick means
        price pushed to a high and was REJECTED — bearish regardless of
        trend direction, so it does NOT invert in a bull regime.
        NOTE: add_features() already computes out["upper_wick_ratio"] (raw)
        but it is NOT z-scored and NOT in MOVEMENT_FEATURES.  This adds the
        z-scored version using the same window=100 convention.

    candle_direction_streak
        Count of consecutive higher closes (+N) or lower closes (-N).
        A long green streak signals exhaustion/mean-reversion potential.
        Unlike z_score_roc_5, this captures SEQUENCE not magnitude and
        does NOT invert in a bull regime.
    """
    high  = out["high"].astype(float)
    low   = out["low"].astype(float)
    open_ = out["open"].astype(float)
    close = out["close"].astype(float)

    # ── z_score_upper_wick_ratio ─────────────────────────────────────────────
    candle_top       = pd.concat([close, open_], axis=1).max(axis=1)
    upper_wick       = (high - candle_top).clip(lower=0.0)
    full_range       = (high - low).replace(0, np.nan)
    upper_wick_ratio = (upper_wick / full_range).fillna(0.0)
    out["z_score_upper_wick_ratio"] = _z_score_normalize(upper_wick_ratio, window=100)

    # ── candle_direction_streak ──────────────────────────────────────────────
    direction   = np.sign(close.diff().fillna(0).values)   # array of +1 / 0 / -1
    streak_vals = np.zeros(len(close), dtype=float)
    current     = 0.0
    for i in range(1, len(direction)):
        d = direction[i]
        if d > 0:
            current = max(current, 0.0) + 1.0
        elif d < 0:
            current = min(current, 0.0) - 1.0
        else:
            current = 0.0
        streak_vals[i] = current

    streak_series = pd.Series(streak_vals, index=out.index)
    out["candle_direction_streak"] = _z_score_normalize(streak_series, window=100)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    🔥 ADVERSARIALLY PURGED Feature Engineering
    
    Principle: Every feature must look IDENTICAL in bear and bull markets
    when the underlying price pattern is the same.
    
    Saboteur accuracy with these features should be < 55%.
    """
    out = _standardize_columns(df)

    close = out["close"].astype(float)
    high = out["high"].astype(float)
    low = out["low"].astype(float)
    open_ = out["open"].astype(float)

    # ========================================================================
    # BASE CALCULATIONS
    # ========================================================================
    sma_20 = close.rolling(window=20).mean()
    std_20 = close.rolling(window=20).std().replace(0, np.nan)
    ema_9 = close.ewm(span=9, adjust=False).mean()
    ema_21 = close.ewm(span=21, adjust=False).mean()
    ema_50 = close.ewm(span=50, adjust=False).mean()

    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta.where(delta < 0, 0.0))
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_14 = 100 - (100 / (1 + rs))

    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr_14 = tr.rolling(window=14).mean()
    adx = _calculate_adx(out, period=14)

    # ========================================================================
    # REGIME CONTEXT (Safe features)
    # ========================================================================
    out["regime_trend"] = (adx > 25).astype(float)

    if "timestamp" in out.columns:
        ts = _ensure_ist(out["timestamp"])
        hour = ts.dt.hour  # type: ignore[attr-defined]
        regime_time = pd.Series(1.0, index=out.index)
        regime_time[hour < 11] = 0.0
        regime_time[hour >= 14] = 2.0
        out["regime_time_of_day"] = regime_time
    else:
        out["regime_time_of_day"] = 1.0

    # ========================================================================
    # NORMALIZED MOMENTUM (z-scored - regime neutral)
    # ========================================================================
    roc_5_raw = close.pct_change(5)
    rsi_slope_raw = rsi_14 - rsi_14.shift(3)
    ema_9_slope_raw = (ema_9 - ema_9.shift(3)) / ema_9.shift(3)
    returns_10_raw = close.pct_change(10)

    out["z_score_roc_5"] = _z_score_normalize(roc_5_raw, window=100)
    out["z_score_rsi_slope"] = _z_score_normalize(rsi_slope_raw, window=100)
    out["z_score_ema_slope"] = _z_score_normalize(ema_9_slope_raw, window=100)

    rolling_std_for_norm = close.pct_change().rolling(20).std().replace(0, np.nan)
    out["returns_10_normalized"] = (returns_10_raw / rolling_std_for_norm).fillna(0)

    # ========================================================================
    # MEAN REVERSION (z-scored)
    # ========================================================================
    out["z_score_20"] = (close - sma_20) / std_20
    out["mean_reversion_strength"] = ((close - sma_20) / std_20).abs() / 3.0

    # ========================================================================
    # VOLATILITY AS RELATIVE MEASURE (not absolute - regime neutral)
    # ========================================================================

    # vol_ratio: short/long volatility ratio
    # "Is vol expanding NOW?" → regime-neutral (works in any market)
    vol_short = close.pct_change().rolling(5).std()
    vol_long = close.pct_change().rolling(20).std().replace(0, np.nan)
    out["vol_ratio"] = (vol_short / vol_long).fillna(1.0)

    # vol_expansion: is volatility expanding vs recent baseline?
    # z-score of ATR → regime-neutral version of atr_pct
    vol_expansion = vol_short / vol_short.rolling(50).mean().replace(0, np.nan)
    out["vol_expansion"] = vol_expansion.fillna(1.0)

    # atr_z_score: ATR relative to its recent history
    # Replaces raw atr_pct (which was #1 biased feature!)
    atr_pct_raw = (atr_14 / close) * 100.0
    out["atr_z_score"] = _z_score_normalize(atr_pct_raw, window=100)

    # Also expose raw atr_14 for use by build_bear_dataset() confirmed label
    out["atr"] = atr_14

    # ========================================================================
    # PRICE ACTION (normalized WARN features)
    # ========================================================================

    # high_low_range normalized (was WARN: 0.049)
    high_low_raw = (high - low) / close
    out["z_score_high_low_range"] = _z_score_normalize(high_low_raw, window=100)

    # close_position (SAFE: 0.005) - already 0-1 range, no bias
    range_size = (high - low).replace(0, np.nan)
    out["close_position"] = (close - low) / range_size

    # distance_from_vwap normalized (was WARN: 0.036)
    typical_price = (high + low + close) / 3
    vwap = typical_price.rolling(window=20).mean()
    dist_vwap_raw = (close - vwap) / close
    out["z_score_distance_from_vwap"] = _z_score_normalize(dist_vwap_raw, window=100)

    # ========================================================================
    # CANDLE MICROSTRUCTURE (regime-neutral - body/wick ratios)
    # ========================================================================
    body = (close - open_).abs()
    total_range = range_size.fillna(1e-9)

    out["candle_body_ratio"] = (body / total_range).fillna(0)

    upper_wick = high - pd.concat([close, open_], axis=1).max(axis=1)
    lower_wick = pd.concat([close, open_], axis=1).min(axis=1) - low

    out["upper_wick_ratio"] = (upper_wick / total_range).fillna(0)

    # lower_wick_ratio normalized (was WARN: 0.027)
    lower_wick_raw = (lower_wick / total_range).fillna(0)
    out["z_score_lower_wick_ratio"] = _z_score_normalize(lower_wick_raw, window=100)

    # ========================================================================
    # INTRADAY MICROSTRUCTURE
    # ========================================================================
    if "timestamp" in out.columns:
        ts = _ensure_ist(out["timestamp"])
        ts_hour = ts.dt.hour  # type: ignore[attr-defined]
        ts_minute = ts.dt.minute  # type: ignore[attr-defined]

        is_new_session = (ts_hour == 9) & (ts_hour.shift(1) == 15)
        session_id = is_new_session.cumsum()

        minutes_elapsed = ts.groupby(session_id).cumcount()
        high_idx = high.groupby(session_id).cummax() == high
        low_idx = low.groupby(session_id).cummin() == low

        last_high_minute = minutes_elapsed.where(high_idx).groupby(session_id).ffill()
        last_low_minute = minutes_elapsed.where(low_idx).groupby(session_id).ffill()

        minutes_since_high_raw = minutes_elapsed - last_high_minute.fillna(0)
        out["minutes_since_low"] = minutes_elapsed - last_low_minute.fillna(0)

        # Normalize minutes_since_high (was WARN: 0.042)
        out["z_score_minutes_since_high"] = _z_score_normalize(
            minutes_since_high_raw.astype(float), window=100
        )

        # Time features (daily cycle - safe to keep)
        minutes = ts_hour * 60 + ts_minute
        minute_of_session = minutes - SESSION_START_MINUTES
        out["minute_of_session"] = minute_of_session
        out["time_sin"] = np.sin(2 * np.pi * minute_of_session / TRADING_DAY_MINUTES)
        # time_cos PURGED (0.061 - encoded seasonality/year info!)
    else:
        out["z_score_minutes_since_high"] = 0.0
        out["minutes_since_low"] = 0.0
        out["time_sin"] = 0.0

    # ========================================================================
    # BEAR HUNTER v2 FEATURES (exhaustion/rejection - regime-neutral)
    # ========================================================================
    _add_bear_features(out)

    return out



def create_features(df: pd.DataFrame) -> pd.DataFrame:
    return add_features(df)


def extract_movement_feature_row(df_feat: pd.DataFrame, lookback: int = 50) -> pd.Series:
    if len(df_feat) < lookback:
        raise ValueError(f"Not enough rows for lookback={lookback}")
    last = df_feat.iloc[-1]
    feature = last.reindex(MOVEMENT_FEATURES).astype(float)
    return feature.fillna(0.0)


def extract_risk_feature_row(
    df_feat: pd.DataFrame,
    movement_prediction: Dict,
    portfolio_state: Dict,
    lookback: int = 50,
) -> pd.Series:
    if len(df_feat) < lookback:
        raise ValueError(f"Not enough rows for lookback={lookback}")
    last = df_feat.iloc[-1]
    base = last.reindex(RISK_FEATURES).astype(float).fillna(0.0)

    direction = movement_prediction.get("direction")
    direction_map = {"DOWN": 0.0, "NEUTRAL": 1.0, "UP": 2.0}
    direction_value = direction_map.get(direction, 1.0) if direction else 1.0
    base["pred_direction"] = float(direction_value)
    base["pred_confidence"] = float(movement_prediction.get("confidence", 0.0) or 0.0)
    base["drawdown"] = float(portfolio_state.get("drawdown", 0.0) or 0.0)
    base["daily_pnl_pct"] = float(portfolio_state.get("daily_pnl_pct", 0.0) or 0.0)
    base["consecutive_losses"] = float(portfolio_state.get("consecutive_losses", 0) or 0)
    base["position_size_pct"] = float(portfolio_state.get("position_size_pct", 0.0) or 0.0)

    return base