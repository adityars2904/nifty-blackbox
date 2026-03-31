from __future__ import annotations

from typing import Optional
import numpy as np
import pandas as pd

SLIPPAGE_PCT = 0.0005


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names to lowercase."""
    rename_map = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Timestamp": "timestamp",
        "CandleDateTime": "timestamp",
        "Symbol": "symbol",
    }
    out = df.rename(columns=rename_map).copy()
    out.columns = [str(c).lower() for c in out.columns]
    return out


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range (ATR)."""
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)

    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window=period).mean()


def label_movement(
    df: pd.DataFrame,
    horizon: int = 6,
    atr_period: int = 14,
    atr_mult: float = 0.5,
) -> pd.Series:
    """
    Generate movement labels based on forward price movement vs ATR threshold.
    
    Labels:
        0 = DOWN (future_close < close - threshold)
        1 = NEUTRAL (within threshold)
        2 = UP (future_close > close + threshold)
    """
    out = _standardize_columns(df)
    close = out["close"].astype(float)
    atr = _atr(out, period=atr_period)
    
    # Look ahead 'horizon' candles
    future_close = close.shift(-horizon)

    # Threshold = 0.5 * ATR (volatility-adjusted)
    threshold = atr * atr_mult
    
    # Initialize labels as float
    labels = pd.Series(np.nan, index=out.index, dtype=float)

    # Assign labels based on movement vs threshold
    labels[future_close > close + threshold] = 2.0  # UP
    labels[future_close < close - threshold] = 0.0  # DOWN
    labels[(future_close <= close + threshold) & (future_close >= close - threshold)] = 1.0  # NEUTRAL

    return labels


# ============================================================================
# 🔥 NEW: MAE-BASED RISK LABELING (REPLACES SHARPE)
# ============================================================================

def label_risk_score(
    df: pd.DataFrame,
    direction_labels: pd.Series,
    atr_period: int = 14,
    time_horizon: int = 12,
    atr_mult: float = 1.5,
) -> pd.Series:
    """
    Generate risk scores based on MAE (Max Adverse Excursion).
    
    🔥 WHY MAE IS BETTER THAN SHARPE:
    - Sharpe: Ratio of mean/std → Very noisy for short trades
    - MAE: "Worst price against position" → Stable and predictable
    
    Higher score = Lower MAE = Safer trade
    
    Formula:
        MAE = max(price movement against position) / entry_price
        Risk Score = 100 - (MAE% × multiplier)
    
    Args:
        df: OHLC DataFrame
        direction_labels: Movement labels (0=DOWN, 1=NEUTRAL, 2=UP)
        atr_period: ATR calculation period
        time_horizon: Look-ahead period (default: 12 candles = 60 min)
        atr_mult: Multiplier for MAE threshold
    
    Returns:
        Series of risk scores (0-100, higher = safer trade)
    """
    out = _standardize_columns(df)
    atr = _atr(out, period=atr_period)
    close = out["close"].astype(float)
    high = out["high"].astype(float)
    low = out["low"].astype(float)

    scores: list[float] = []
    
    for idx in range(len(out)):
        # Get direction label for this candle
        label = direction_labels.iloc[idx] if idx < len(direction_labels) else np.nan
        
        # Skip if no valid label or NEUTRAL
        if not np.isfinite(label) or label == 1:
            scores.append(50.0)  # Neutral = medium risk
            continue

        # Determine trade side
        side = "LONG" if int(label) == 2 else "SHORT"
        
        # Get entry price and ATR
        entry_price = close.iloc[idx]
        atr_value = atr.iloc[idx]
        
        if not np.isfinite(entry_price) or not np.isfinite(atr_value) or atr_value <= 0:
            scores.append(50.0)
            continue
        
        # Look ahead to measure MAE
        end_idx = min(idx + time_horizon, len(out))
        
        if end_idx <= idx + 1:
            scores.append(50.0)
            continue
        
        try:
            if side == "LONG":
                # For LONG: MAE = worst low price
                worst_price = float(low.iloc[idx+1:end_idx].min())
                mae_pct = (entry_price - worst_price) / entry_price
            else:
                # For SHORT: MAE = worst high price
                worst_price = float(high.iloc[idx+1:end_idx].max())
                mae_pct = (worst_price - entry_price) / entry_price
            
            # Normalize MAE by ATR (volatility-adjusted)
            mae_atr_ratio = (mae_pct * entry_price) / atr_value
            
            # Convert to 0-100 score
            # MAE of 0 ATRs → score 100 (perfect)
            # MAE of 2 ATRs → score 0 (terrible)
            # Linear scale between
            score = 100.0 - (mae_atr_ratio / 2.0 * 100.0)
            score = float(np.clip(score, 0.0, 100.0))
            
            scores.append(score)
            
        except Exception:
            scores.append(50.0)

    return pd.Series(scores, index=out.index, dtype=float)


# ============================================================================
# LEGACY: Keep old Sharpe-based function for reference
# ============================================================================

def label_risk_score_sharpe(
    df: pd.DataFrame,
    direction_labels: pd.Series,
    atr_period: int = 14,
    stop_mult: float = 1.5,
    take_mult: float = 3.0,
    time_stop: int = 12,
    slippage_pct: float = SLIPPAGE_PCT,
) -> pd.Series:
    """
    LEGACY: Sharpe-based risk scoring (kept for comparison).
    
    ⚠️ THIS DOESN'T WORK WELL - use label_risk_score() instead.
    """
    # [Keep old implementation for reference if needed]
    # For now, just call the new MAE version
    return label_risk_score(df, direction_labels, atr_period=atr_period)