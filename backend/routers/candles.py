"""
Candles router — fetch OHLCV candle data with VWAP computation.

Primary: QuestDB. Fallback: parquet files (when QuestDB is empty).
Supports raw 1m candles and resampled timeframes (5m, 15m, 1h, 1d).
Returns TradingView Lightweight Charts–compatible JSON (Unix epoch seconds).

Two routes:
  - /api/candles — API-prefixed route
  - /candles     — non-prefixed route (used by chart JS)
"""

from __future__ import annotations

import logging
from datetime import datetime, time
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from adapters.questdb_adapter import fetch_candles_raw, fetch_candles, _resample, QuestDBConnectionError
from config import IST

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Market Data"])

# Allowed timeframes
VALID_TIMEFRAMES = {"1m", "5m", "15m", "1h", "1d"}

# Mapping timeframe → pandas resample rule
_RESAMPLE_RULES = {
    "5m": "5min",
    "15m": "15min",
    "1h": "1h",
    "1d": "1D",
}

# Parquet fallback paths
_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"
_PARQUET_FILES = {
    "NIFTY": _DATA_DIR / "nifty_cleaned.parquet",
    "BANKNIFTY": _DATA_DIR / "banknifty_cleaned.parquet",
}

# In-memory parquet cache (loaded once)
_parquet_cache: dict[str, pd.DataFrame] = {}


def _load_parquet(symbol: str) -> pd.DataFrame:
    """Load parquet into memory (cached)."""
    if symbol not in _parquet_cache:
        fpath = _PARQUET_FILES.get(symbol)
        if fpath is None or not fpath.exists():
            raise HTTPException(status_code=404, detail=f"No data for {symbol}")
        df = pd.read_parquet(fpath)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        _parquet_cache[symbol] = df
    return _parquet_cache[symbol]


def _parquet_candles(symbol: str, timeframe: str, n: int) -> list[dict]:
    """Build TradingView-format candles from parquet data."""
    df = _load_parquet(symbol)

    if timeframe != "1m":
        rule = _RESAMPLE_RULES[timeframe]
        df_ts = df.set_index("timestamp")
        from typing import Hashable, Callable, Mapping, Any
        import pandas as pd
        agg: Mapping[Hashable, Callable | str | list[Any]] = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        df_rs = df_ts.resample(rule, closed="left", label="left").agg(agg).dropna(subset=["close"])
        df_rs = df_rs.tail(n)
        candles = []
        for ts, row in df_rs.iterrows():
            ts_val = ts
            # Only convert to pd.Timestamp if ts_val is a valid type
            if not isinstance(ts_val, pd.Timestamp):
                if isinstance(ts_val, (str, int, float)):
                    try:
                        ts_val = pd.Timestamp(ts_val)
                    except Exception:
                        ts_val = pd.Timestamp(0, unit="s")
                else:
                    ts_val = pd.Timestamp(0, unit="s")
            time_val = int(ts_val.timestamp()) if hasattr(ts_val, "timestamp") else 0
            candles.append({
                "time": time_val,
                "open": round(float(row["open"]), 2),
                "high": round(float(row["high"]), 2),
                "low": round(float(row["low"]), 2),
                "close": round(float(row["close"]), 2),
                "volume": int(row["volume"]) if row["volume"] else 0,
            })
        return candles
    else:
        df_tail = df.tail(n)
        candles = []
        for _, row in df_tail.iterrows():
            import pandas as pd
            ts_val = row["timestamp"]
            ts_val = ts_val if hasattr(ts_val, "timestamp") else pd.Timestamp(ts_val)
            candles.append({
                "time": int(ts_val.timestamp()),
                "open": round(float(row["open"]), 2),
                "high": round(float(row["high"]), 2),
                "low": round(float(row["low"]), 2),
                "close": round(float(row["close"]), 2),
                "volume": int(row["volume"]) if pd.notna(row["volume"]) else 0,
            })
        return candles


def _add_vwap(candles: list[dict]) -> list[dict]:
    """Add session-anchored VWAP field to each candle dict.

    VWAP resets at each session open (09:15 IST daily).
    """
    if not candles:
        return candles

    cum_tp_vol = 0.0
    cum_vol = 0.0
    prev_date = None

    session_start_time = time(9, 15)

    for c in candles:
        # Convert epoch seconds to IST datetime for session boundary detection
        ist_dt = datetime.fromtimestamp(c["time"], tz=IST)
        current_date = ist_dt.date()
        current_time = ist_dt.time()

        # Detect session boundary
        new_session = False
        if prev_date is None:
            new_session = True
        elif current_date != prev_date and current_time >= session_start_time:
            new_session = True
        elif current_time == session_start_time:
            new_session = True

        if new_session:
            cum_tp_vol = 0.0
            cum_vol = 0.0

        prev_date = current_date

        # VWAP computation
        typical_price = (c["high"] + c["low"] + c["close"]) / 3
        volume = c.get("volume", 0) or 0

        cum_tp_vol += typical_price * volume
        cum_vol += volume

        if cum_vol > 0:
            c["vwap"] = round(cum_tp_vol / cum_vol, 2)
        else:
            c["vwap"] = round(c["close"], 2)

    return candles


def _get_candles_impl(
    symbol: str = "NIFTY",
    timeframe: str = "1m",
    n: int = 200,
) -> list[dict]:
    """Core implementation shared by both routes."""
    if symbol not in ("NIFTY", "BANKNIFTY"):
        raise HTTPException(status_code=400, detail="symbol must be NIFTY or BANKNIFTY")
    if timeframe not in VALID_TIMEFRAMES:
        raise HTTPException(status_code=400, detail="Invalid timeframe")

    n = min(max(n, 1), 100000)

    try:
        if timeframe == "1m":
            try:
                result = fetch_candles_raw(symbol, n)
            except QuestDBConnectionError:
                result = []
            if result and len(result) >= n:
                return _add_vwap(result)
            # QuestDB empty or insufficient — supplement with parquet
            parquet = _parquet_candles(symbol, timeframe, n)
            if not result:
                return _add_vwap(parquet)
            # Merge: parquet + questdb, dedup by time, keep questdb values
            merged = {c["time"]: c for c in parquet}
            merged.update({c["time"]: c for c in result})  # QuestDB wins
            combined = sorted(merged.values(), key=lambda c: c["time"])
            return _add_vwap(combined[-n:])
        else:
            multiplier = {"5m": 5, "15m": 15, "1h": 60, "1d": 375}
            raw_n = n * multiplier.get(timeframe, 15) + 50
            raw_n = min(raw_n, 350_000)

            try:
                df = fetch_candles(symbol, n_candles=raw_n)
            except Exception:
                import pandas as pd
                df = pd.DataFrame()

            if df.empty:
                return _add_vwap(_parquet_candles(symbol, timeframe, n))

            rule = _RESAMPLE_RULES[timeframe]
            resampled = _resample(df, rule)

            resampled = resampled.tail(n)
            candles = []
            for _, row in resampled.iterrows():
                ts_val = row.get("timestamp") or row.get("ts")
                import pandas as pd
                if ts_val is not None:
                    ts_val = ts_val if hasattr(ts_val, "timestamp") else pd.Timestamp(ts_val)
                    time_val = int(ts_val.timestamp())
                else:
                    time_val = 0
                candles.append({
                    "time": time_val,
                    "open": round(float(row["open"]), 2),
                    "high": round(float(row["high"]), 2),
                    "low": round(float(row["low"]), 2),
                    "close": round(float(row["close"]), 2),
                    "volume": int(row["volume"]) if row["volume"] else 0,
                })

            if len(candles) < n:
                parquet = _parquet_candles(symbol, timeframe, n)
                merged = {c["time"]: c for c in parquet}
                merged.update({c["time"]: c for c in candles})
                combined = sorted(merged.values(), key=lambda c: c["time"])
                return _add_vwap(combined[-n:])

            return _add_vwap(candles)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error fetching candles: %s", e)
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


# ── Route 1: /api/candles (used by API consumers) ───────────────────────────
@router.get("/api/candles")
def get_candles_api(
    symbol: str = Query("NIFTY", description="NIFTY or BANKNIFTY"),
    timeframe: str = Query("1m", description="1m | 5m | 15m | 1h | 1d"),
    n: int = Query(200, ge=1, le=100000, description="Number of candles"),
):
    """OHLCV candles with VWAP (API-prefixed route)."""
    return _get_candles_impl(symbol, timeframe, n)


# ── Route 2: /candles (used by chart JS — no /api prefix) ──────────────────
@router.get("/candles")
def get_candles_chart(
    symbol: str = Query("NIFTY", description="NIFTY or BANKNIFTY"),
    timeframe: str = Query("5m", description="1m | 5m | 15m | 1h | 1d"),
    n: int = Query(200, ge=1, le=100000, description="Number of candles"),
):
    """OHLCV candles with VWAP (non-prefixed route for chart JS)."""
    return _get_candles_impl(symbol, timeframe, n)
