"""
QuestDB adapter — all QuestDB communication lives here.

Uses the HTTP REST API on port 9000.
No other file may query QuestDB directly.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional
from urllib.parse import urlencode

import pandas as pd
import requests

from config import (
    IST,
    QUESTDB_HOST,
    QUESTDB_PORT,
)

logger = logging.getLogger(__name__)

# Date floor to skip corrupt partitions (candles~10/2024-01-20.2/low.d)
_PARTITION_SAFE_FLOOR = "2025-01-01"


# ============================================================================
# CUSTOM EXCEPTION
# ============================================================================

class QuestDBConnectionError(Exception):
    """Raised when QuestDB is unreachable or query fails."""


# ============================================================================
# HTTP SESSION (module-level singleton)
# ============================================================================

_session: Optional[requests.Session] = None
_base_url: Optional[str] = None


def init_pool(minconn: int = 1, maxconn: int = 5) -> None:
    """Initialise the HTTP session. Safe to call multiple times.
    Signature kept for backward-compatibility with callers passing minconn/maxconn."""
    global _session, _base_url
    if _session is not None:
        return
    _base_url = f"http://{QUESTDB_HOST}:{QUESTDB_PORT}"
    _session = requests.Session()
    _session.headers.update({"Accept": "application/json"})
    logger.info("QuestDB HTTP session initialised (%s)", _base_url)


def close_pool() -> None:
    """Tear down the session (called on shutdown)."""
    global _session, _base_url
    if _session is not None:
        _session.close()
        _session = None
        _base_url = None
        logger.info("QuestDB HTTP session closed")


# ============================================================================
# HELPERS
# ============================================================================

def _localize_ts(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the 'ts' column is a tz-aware IST datetime."""
    if df.empty:
        return df
    df = df.copy()
    ts = pd.to_datetime(df["ts"], errors="coerce")
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("UTC").dt.tz_convert(IST)
    else:
        ts = ts.dt.tz_convert(IST)
    df["ts"] = ts
    return df


def _query_df(sql: str, params: tuple = ()) -> pd.DataFrame:
    """Execute *sql* via the QuestDB REST API and return the result as a DataFrame.

    For parameterised queries the adapter does safe string interpolation
    (QuestDB REST API does not support bind-parameters).
    """
    global _session, _base_url
    if _session is None:
        init_pool()
    assert _session is not None and _base_url is not None

    # Inline params: replace %s placeholders with properly quoted values
    if params:
        sql = _interpolate(sql, params)

    url = f"{_base_url}/exec"
    try:
        resp = _session.get(url, params={"query": sql}, timeout=30)
        resp.raise_for_status()
        body = resp.json()
    except requests.RequestException as exc:
        raise QuestDBConnectionError(f"QuestDB query failed: {exc}") from exc

    if "error" in body:
        raise QuestDBConnectionError(f"QuestDB query failed: {body['error']}")

    columns = [c["name"] for c in body.get("columns", [])]
    rows = body.get("dataset", [])
    return pd.DataFrame(rows, columns=columns)


def _interpolate(sql: str, params: tuple) -> str:
    """Replace %s placeholders with QuestDB-safe literals."""
    parts: list[str] = []
    idx = 0
    for p in params:
        pos = sql.index("%s", idx)
        parts.append(sql[idx:pos])
        if isinstance(p, str):
            parts.append(f"'{p}'")
        elif isinstance(p, datetime):
            parts.append(f"'{p.strftime('%Y-%m-%dT%H:%M:%S.%fZ')}'")
        elif p is None:
            parts.append("NULL")
        else:
            parts.append(str(p))
        idx = pos + 2
    parts.append(sql[idx:])
    return "".join(parts)


# ============================================================================
# PUBLIC API
# ============================================================================

def fetch_candles(symbol: str, n_candles: int = 200) -> pd.DataFrame:
    """
    Fetch the last *n_candles* 1-minute candles for *symbol*.

    Returns DataFrame with columns: symbol, ts, open, high, low, close, volume.
    ``ts`` is tz-aware IST.
    """
    sql = f"""
        SELECT symbol, ts, open, high, low, close, volume
        FROM candles
        WHERE symbol = %s
          AND ts > '{_PARTITION_SAFE_FLOOR}'
        ORDER BY ts DESC
        LIMIT %s
    """
    df = _query_df(sql, (symbol, n_candles))
    if df.empty:
        raise QuestDBConnectionError(f"No candles returned for {symbol}")
    df = df.sort_values("ts").reset_index(drop=True)
    df = _localize_ts(df)
    return df


def fetch_candles_range(
    symbol: str,
    start_dt: datetime,
    end_dt: datetime,
) -> pd.DataFrame:
    """Fetch all candles between two IST datetimes (inclusive)."""
    sql = """
        SELECT symbol, ts, open, high, low, close, volume
        FROM candles
        WHERE symbol = %s
          AND ts >= %s
          AND ts <= %s
        ORDER BY ts ASC
    """
    df = _query_df(sql, (symbol, start_dt, end_dt))
    if df.empty:
        return df
    df = _localize_ts(df)
    return df


def get_latest_candle_time(symbol: str) -> Optional[datetime]:
    """Return the timestamp of the most recent candle for *symbol* (IST)."""
    sql = f"""
        SELECT ts
        FROM candles
        WHERE symbol = %s
          AND ts > '{_PARTITION_SAFE_FLOOR}'
        ORDER BY ts DESC
        LIMIT 1
    """
    df = _query_df(sql, (symbol,))
    if df.empty:
        return None
    ts = pd.to_datetime(df.iloc[0]["ts"])
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC").tz_convert(IST)
    else:
        ts = ts.tz_convert(IST)
    return ts


def fetch_candles_raw(symbol: str, n: int = 200) -> list[dict]:
    """
    Fetch last *n* 1-minute candles as a list of dicts with Unix-epoch time
    (seconds).  Used by the frontend chart router.
    """
    sql = f"""
        SELECT
            extract(epoch FROM ts) AS time,
            open, high, low, close, volume
        FROM candles
        WHERE symbol = %s
          AND ts > '{_PARTITION_SAFE_FLOOR}'
        ORDER BY ts DESC
        LIMIT %s
    """
    df = _query_df(sql, (symbol, n))
    if df.empty:
        return []
    # Return chronological order (oldest first)
    result = [
        {
            "time": int(float(row["time"])),
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": int(row["volume"]) if row["volume"] else 0,
        }
        for _, row in df.iloc[::-1].iterrows()
    ]
    return result


def check_connection() -> bool:
    """Return True if QuestDB REST API is reachable."""
    global _session, _base_url
    if _session is None:
        try:
            init_pool()
        except Exception:
            return False
    assert _session is not None and _base_url is not None
    try:
        resp = _session.get(f"{_base_url}/exec", params={"query": "SELECT 1"}, timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


# ============================================================================
# RESAMPLING
# ============================================================================

def _resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """
    Resample 1-minute OHLCV to the given rule.

    Uses closed='left', label='left' to avoid look-ahead bias.
    Drops the final incomplete candle.
    """
    if df.empty:
        return df

    df = df.copy()
    df = df.set_index("ts")

    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }

    resampled = df.resample(rule, closed="left", label="left").agg(agg)  # type: ignore[arg-type]
    resampled = resampled.dropna(subset=["open"])  # remove empty buckets

    # Drop last (likely incomplete) candle
    if len(resampled) > 1:
        resampled = resampled.iloc[:-1]

    resampled = resampled.reset_index()
    resampled = resampled.rename(columns={"ts": "ts"})

    # Carry symbol forward if present
    if "symbol" in df.columns:
        resampled["symbol"] = df["symbol"].iloc[0]

    # Rename the index back to 'ts' (it's already named ts from reset_index)
    # Ensure timestamp column is named 'timestamp' for feature_engineering compat
    resampled = resampled.rename(columns={"ts": "timestamp"})

    return resampled


def resample_to_5m(df: pd.DataFrame) -> pd.DataFrame:
    """Resample 1-min DataFrame → 5-min OHLCV. Drops final incomplete candle."""
    return _resample(df, "5min")


def resample_to_15m(df: pd.DataFrame) -> pd.DataFrame:
    """Resample 1-min DataFrame → 15-min OHLCV. Drops final incomplete candle."""
    return _resample(df, "15min")


def resample_to_30m(df: pd.DataFrame) -> pd.DataFrame:
    """Resample 1-min DataFrame → 30-min OHLCV (Stage 1 only)."""
    return _resample(df, "30min")
