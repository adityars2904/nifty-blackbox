from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import Optional, Any, cast
import pandas as pd
import requests


def db_query(sql: str) -> dict:
    """Execute SQL via QuestDB REST API and return JSON response."""
    host = os.getenv("QUESTDB_HOST", "localhost")
    port = os.getenv("QUESTDB_PORT", "9000")
    url = f"http://{host}:{port}/exec"
    try:
        resp = requests.get(url, params={"query": sql}, timeout=30,
                            headers={"Accept": "application/json"})
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        raise RuntimeError(f"QuestDB query failed: {exc}") from exc


@dataclass
class SessionWindow:
    start: str = "09:15"
    end: str = "15:30"


TIMEFRAME_SQL = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
    "1d": "1d",
}


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names to lowercase."""
    rename_map = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
        "Timestamp": "timestamp",
        "CandleDateTime": "timestamp",
        "Candle_Date": "candle_date",
        "Candle_Time": "candle_time",
        "Symbol": "symbol",
    }
    out = df.rename(columns=rename_map).copy()
    out.columns = [str(c).lower() for c in out.columns]
    return out


def _ensure_ist(series: pd.Series) -> pd.Series:
    """Ensure timestamp series is in IST timezone."""
    # Force type to Any to avoid Pylance issues with .dt accessor later
    ts: Any = pd.to_datetime(series, errors="coerce")
    
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("Asia/Kolkata")
    else:
        ts = ts.dt.tz_convert("Asia/Kolkata")
    return ts


def filter_session(df: pd.DataFrame, session: SessionWindow) -> pd.DataFrame:
    """
    Filter DataFrame to only include session hours (09:15-15:15 IST).
    Optimized to use integer comparison instead of string formatting.
    """
    if "timestamp" not in df.columns:
        return df

    # FIX 1: Type hint 'ts' as Any so Pylance allows .dt access
    ts: Any = _ensure_ist(df["timestamp"])
    out = df.copy()
    out["timestamp"] = ts

    # Convert "09:15" -> 915 for faster integer comparison
    start_int = int(session.start.replace(":", ""))
    end_int = int(session.end.replace(":", ""))

    # Create integer time representation (HHMM)
    # Using Any type for ts allows access to .dt without errors
    series_time = ts.dt.hour * 100 + ts.dt.minute

    mask = (series_time >= start_int) & (series_time <= end_int)
    return out.loc[mask].reset_index(drop=True)


def resample_ohlc(df: pd.DataFrame, rule: str = "5min") -> pd.DataFrame:
    """Resample OHLC data to specified timeframe."""
    if "timestamp" not in df.columns:
        raise ValueError("timestamp column required for resampling")

    df = df.sort_values("timestamp")
    df_indexed = df.set_index("timestamp")

    agg_dict: dict[str, str] = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
    }

    # FIX 2: Cast agg_dict to Any to suppress dictionary type mismatch
    resampled = df_indexed.resample(rule, label="right", closed="right").agg(cast(Any, agg_dict))

    if "symbol" in df_indexed.columns:
        # Cast to Any to suppress resample type warnings
        symbol_col: Any = df_indexed["symbol"]
        symbol_resampled = symbol_col.resample(rule, label="right", closed="right").last()
        resampled["symbol"] = symbol_resampled

    resampled = resampled.dropna(subset=["open", "high", "low", "close"]).reset_index()
    return resampled


def load_questdb_candles(
    symbol: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    table: str = "candles",
) -> pd.DataFrame:
    """Load raw 1-minute candles from QuestDB."""
    query = f"""
        SELECT ts as timestamp, open, high, low, close, volume, symbol
        FROM {table}
        WHERE symbol = '{symbol}'
    """
    
    if start:
        query += f" AND ts >= '{start}'"
    if end:
        query += f" AND ts <= '{end}'"
    query += " ORDER BY ts"

    result = db_query(query)
    columns = [c["name"] for c in result.get("columns", [])]
    rows = result.get("dataset", [])
    df = pd.DataFrame(rows, columns=columns)

    return standardize_columns(df)


def fetch_candles(
    symbol: str,
    timeframe: str = "5m",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: Optional[int] = None,
    table: str = "candles",
) -> pd.DataFrame:
    """
    Fetch candles with strict date boundaries and optional resampling.
    
    🚨 CRITICAL FIX: WHERE clause MUST come BEFORE SAMPLE BY in QuestDB SQL!
    
    Args:
        symbol: Trading symbol (NIFTY, BANKNIFTY)
        timeframe: 1m, 5m, 15m, 1h, 1d
        start_date: Start date (YYYY-MM-DD HH:MM:SS)
        end_date: End date (YYYY-MM-DD HH:MM:SS)
        limit: Maximum number of rows
        table: QuestDB table name
    
    Returns:
        DataFrame with OHLC data
    """
    if timeframe not in TIMEFRAME_SQL:
        raise ValueError(f"Invalid timeframe: {timeframe}")

    # Build WHERE clause conditions (must come BEFORE SAMPLE BY)
    where_conditions = [f"symbol = '{symbol}'"]
    
    if start_date:
        where_conditions.append(f"ts >= '{start_date}'")
    if end_date:
        where_conditions.append(f"ts <= '{end_date}'")
    
    where_clause = " AND ".join(where_conditions)

    # Build query based on timeframe
    if timeframe == "1m":
        query = f"""
            SELECT ts as timestamp, open, high, low, close, volume, symbol
            FROM {table}
            WHERE {where_clause}
        """
    else:
        interval = TIMEFRAME_SQL[timeframe]
        query = f"""
            SELECT
                ts as timestamp,
                first(open)  AS open,
                max(high)    AS high,
                min(low)     AS low,
                last(close)  AS close,
                sum(volume)  AS volume,
                last(symbol) AS symbol
            FROM {table}
            WHERE {where_clause}
            SAMPLE BY {interval}
        """

    # Add ORDER BY and LIMIT
    query += " ORDER BY ts ASC"
    
    if limit:
        query += f" LIMIT {limit}"

    # Execute query via QuestDB REST API
    result = db_query(query)
    columns = [c["name"] for c in result.get("columns", [])]
    rows = result.get("dataset", [])
    df = pd.DataFrame(rows, columns=columns)

    # Standardize and clean
    df = standardize_columns(df)
    if "volume" in df.columns:
        df = df.drop(columns=["volume"])
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    return df


def process_candles(
    df: pd.DataFrame,
    session: SessionWindow = SessionWindow(),
    resample_rule: str = "5min",
) -> pd.DataFrame:
    """Process raw candles: filter session + resample."""
    out = standardize_columns(df)
    if "volume" in out.columns:
        out = out.drop(columns=["volume"])

    out = filter_session(out, session=session)
    out = resample_ohlc(out, rule=resample_rule)
    return out


def load_and_process_questdb(
    symbol: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    table: str = "candles",
    session: SessionWindow = SessionWindow(),
    resample_rule: str = "5min",
) -> pd.DataFrame:
    """Load from QuestDB, filter session, and resample in one call."""
    df = load_questdb_candles(symbol=symbol, start=start, end=end, table=table)
    return process_candles(df, session=session, resample_rule=resample_rule)


def save_parquet(df: pd.DataFrame, path: str) -> None:
    """Save DataFrame to parquet file."""
    df.to_parquet(path, index=False)


# ============================================================================
# 🚨 CRITICAL: TIME MACHINE FIX - PROPER DATE BOUNDARIES
# ============================================================================

def load_training_data(
    symbol: str,
    timeframe: str = "5m",
    table: str = "candles",
) -> pd.DataFrame:
    """Load training data (2022-04-11 to 2024-04-30)."""
    return fetch_candles(
        symbol=symbol,
        timeframe=timeframe,
        start_date="2022-04-11 09:16:00",  # ✅ CORRECT
        end_date="2024-04-30 15:30:00",     # ✅ CORRECT
        table=table,
    )

def load_validation_data(
    symbol: str,
    timeframe: str = "5m",
    table: str = "candles",
) -> pd.DataFrame:
    """Load validation data (2024-05-01 to 2024-12-31)."""
    return fetch_candles(
        symbol=symbol,
        timeframe=timeframe,
        start_date="2024-05-01 09:15:00",   # ✅ CORRECT
        end_date="2024-12-31 15:30:00",     # ✅ CORRECT
        table=table,
    )


def load_test_data(
    symbol: str,
    timeframe: str = "5m",
    table: str = "candles",
) -> pd.DataFrame:
    """Load test data (2025 onwards)."""
    return fetch_candles(
        symbol=symbol,
        timeframe=timeframe,
        start_date="2025-01-01 09:15:00",
        table=table,
    )