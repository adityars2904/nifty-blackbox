#!/usr/bin/env python3
"""
07_streak_decomposition.py
===========================
Decompose what `consecutive_wins` is actually encoding, and identify
direct trend features that should replace it.

Three-part analysis:

  PART 1 — CORRELATION AUDIT
  Measure the correlation between `consecutive_wins` and every price-structure
  feature in the meta-filter feature set. If consecutive_wins is a trend proxy,
  it will correlate most strongly with regime_trend, z_score_ema_slope, and
  z_score_roc_5 (directional momentum features).

  PART 2 — CONDITIONAL IC DECOMPOSITION
  Split vault signals into streak buckets: [0], [1-2], [3+].
  Within each bucket, compute the Newey-West IC of every other feature
  against continuous pnl_pts.
  Key question: do price-structure features (regime_trend, vol_ratio,
  z_score_20, etc.) retain their IC in the high-streak bucket, or do they
  collapse to zero? If they collapse, consecutive_wins is absorbing their
  information — they become redundant once you're on a roll.

  PART 3 — REPLACEMENT CANDIDATE RANKING
  Test five candidate trend features as replacements for consecutive_wins.
  Each is measured by:
    (a) Pearson IC against pnl_pts (Newey-West corrected)
    (b) Correlation with consecutive_wins (how much does it overlap?)
    (c) Incremental IC: IC after consecutive_wins is already included
  Rank by: high IC, low overlap, high incremental IC.

  Candidates:
    C1. multi_tf_trend_alignment  — Are 5m, 15m, and daily EMAs all pointing
        the same direction? (0=none, 1=partial, 2=full alignment)
    C2. ema_stack_strength        — Normalized distance between EMA(9),
        EMA(21), EMA(50). Larger gaps = stronger trend.
    C3. price_momentum_persistence — Fraction of last 20 bars that closed
        higher than open in the signal direction. [0, 1]
    C4. session_trend_commitment  — Distance of current price from session
        open, normalized by session ATR. Measures whether the day has
        committed to a direction.
    C5. adx_slope                 — Rate of change of ADX over last 5 bars,
        z-scored. Captures whether a trend is strengthening or exhausting.

Output:
    research/outputs/streak_decomposition_correlations.csv
    research/outputs/streak_decomposition_conditional_ic.csv
    research/outputs/streak_decomposition_candidates.csv

Usage:
    cd research && python 07_streak_decomposition.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr, t as t_dist

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.ml.data_loader import fetch_candles
from backend.ml.feature_engineering import add_features, MOVEMENT_FEATURES
from backend.ml.labels import label_movement
from backend.ml.ensemble_predictor import EnsemblePredictor
from scripts.validate_meta_filter_2025 import (
    generate_ensemble_signals,
    build_meta_features,
    label_signal_outcomes,
)
from research.core_execution import simulate_portfolio, THRESHOLDS

MODEL_DIR   = str(project_root / "backend" / "models")
VAULT_START = "2025-01-01 09:15:00"
VAULT_END   = "2025-12-10 15:30:00"
OUTPUT_DIR  = Path(__file__).parent / "outputs"


# ════════════════════════════════════════════════════════════════════════════
# NEWEY-WEST IC UTILITY
# ════════════════════════════════════════════════════════════════════════════

def nw_ic(x: np.ndarray, y: np.ndarray, max_lag: int = 10) -> tuple[float, float]:
    """Pearson IC with Newey-West variance correction. Returns (ic, p_value)."""
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 20:
        return float("nan"), float("nan")
    x_v, y_v = x[valid], y[valid]
    N = len(x_v)

    x_s = (x_v - x_v.mean()) / (x_v.std() + 1e-12)
    y_s = (y_v - y_v.mean()) / (y_v.std() + 1e-12)
    beta = float(np.dot(x_s, y_s) / (np.dot(x_s, x_s) + 1e-12))
    residuals = y_s - beta * x_s

    Q = float(np.dot(residuals ** 2, x_s ** 2))
    for lag in range(1, max_lag + 1):
        w = 1 - lag / (max_lag + 1)
        Q += w * 2 * float(
            np.dot(residuals[lag:] * x_s[lag:], residuals[:-lag] * x_s[:-lag])
        )
    Q /= N
    se = float(np.sqrt(max(Q, 1e-20) / (np.dot(x_s, x_s) / N) ** 2 / N))
    t_stat = beta / (se + 1e-12)
    p_val = float(t_dist.sf(abs(t_stat), df=N - 2) * 2)
    return round(beta, 4), round(p_val, 4)


# ════════════════════════════════════════════════════════════════════════════
# CANDIDATE FEATURE COMPUTATION
# ════════════════════════════════════════════════════════════════════════════

def compute_candidates(feat_5m: pd.DataFrame) -> pd.DataFrame:
    """
    Compute five candidate trend features on the 5m feature DataFrame.
    Returns the input DataFrame with new columns attached.
    """
    df = feat_5m.copy()
    close = df["close"].astype(float)
    high  = df["high"].astype(float)
    low   = df["low"].astype(float)
    open_ = df["open"].astype(float)

    # ── C1: multi_tf_trend_alignment ──────────────────────────────────────
    # Are EMA(9), EMA(21), EMA(50) all stacked in the same direction?
    ema9  = close.ewm(span=9,  adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()

    # Bullish stack: ema9 > ema21 > ema50. Bearish: ema9 < ema21 < ema50.
    bull_stack = ((ema9 > ema21) & (ema21 > ema50)).astype(float)
    bear_stack = ((ema9 < ema21) & (ema21 < ema50)).astype(float)
    partial    = (
        ((ema9 > ema21) | (ema21 > ema50)) &
        ~((ema9 > ema21) & (ema21 > ema50)) &
        ~((ema9 < ema21) & (ema21 < ema50))
    ).astype(float) * 0.5

    # Score: +1 = full bull, -1 = full bear, ±0.5 = partial
    raw_alignment = np.where(
        bull_stack == 1, 1.0,
        np.where(bear_stack == 1, -1.0, np.where(partial == 0.5, 0.5, 0.0))
    )
    df["c1_multi_tf_trend_alignment"] = pd.Series(raw_alignment, index=df.index)

    # ── C2: ema_stack_strength ────────────────────────────────────────────
    # Normalized distance between EMA(9) and EMA(50) relative to ATR.
    # Sign: positive = bullish stack, negative = bearish.
    atr14 = df["atr"].astype(float) if "atr" in df.columns else \
        pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low  - close.shift(1)).abs(),
        ], axis=1).max(axis=1).rolling(14).mean()

    raw_stack_dist = (ema9 - ema50) / (atr14.replace(0, np.nan))
    # Z-score over 100 bars (same convention as MOVEMENT_FEATURES)
    rolling_mean = raw_stack_dist.rolling(100).mean()
    rolling_std  = raw_stack_dist.rolling(100).std().replace(0, np.nan)
    df["c2_ema_stack_strength"] = ((raw_stack_dist - rolling_mean) / rolling_std).fillna(0)

    # ── C3: price_momentum_persistence ───────────────────────────────────
    # Fraction of last 20 bars where close > open (bullish candles).
    # Gives a continuous measure of directional conviction.
    candle_dir = (close > open_).astype(float)  # 1 = bull, 0 = bear
    df["c3_price_momentum_persistence"] = candle_dir.rolling(20).mean().fillna(0.5)

    # ── C4: session_trend_commitment ─────────────────────────────────────
    # (current_price - session_open) / session_ATR_so_far
    # Measures how far the session has committed to a direction.
    if "timestamp" in df.columns:
        from zoneinfo import ZoneInfo
        IST = ZoneInfo("Asia/Kolkata")
        ts = pd.to_datetime(df["timestamp"])
        if ts.dt.tz is None:
            ts = ts.dt.tz_localize(IST)
        else:
            ts = ts.dt.tz_convert(IST)

        is_session_start = (ts.dt.hour == 9) & (ts.dt.minute == 15)
        session_id = is_session_start.cumsum()

        session_open = open_.groupby(session_id).transform("first")
        session_high = high.groupby(session_id).transform("cummax")
        session_low  = low.groupby(session_id).transform("cummin")
        session_range = (session_high - session_low).replace(0, np.nan)

        raw_commit = (close - session_open) / session_range
        rm  = raw_commit.rolling(100).mean()
        rs  = raw_commit.rolling(100).std().replace(0, np.nan)
        df["c4_session_trend_commitment"] = ((raw_commit - rm) / rs).fillna(0)
    else:
        df["c4_session_trend_commitment"] = 0.0

    # ── C5: adx_slope ─────────────────────────────────────────────────────
    # Rate of change of ADX over last 5 bars, z-scored.
    # Positive = trend strengthening. Negative = trend exhausting.
    if "adx_raw" not in df.columns:
        plus_dm  = high.diff()
        minus_dm = -low.diff()
        plus_dm  = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low  - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr_adx = tr.rolling(14).mean()
        plus_di  = 100 * (pd.Series(plus_dm, index=df.index).rolling(14).mean() / atr_adx)
        minus_di = 100 * (pd.Series(minus_dm, index=df.index).rolling(14).mean() / atr_adx)
        dx  = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
        adx_raw = dx.rolling(14).mean()
    else:
        adx_raw = df["adx_raw"]

    adx_slope_raw = adx_raw - adx_raw.shift(5)
    rm5 = adx_slope_raw.rolling(100).mean()
    rs5 = adx_slope_raw.rolling(100).std().replace(0, np.nan)
    df["c5_adx_slope"] = ((adx_slope_raw - rm5) / rs5).fillna(0)

    return df


# ════════════════════════════════════════════════════════════════════════════
# PART 1 — CORRELATION AUDIT
# ════════════════════════════════════════════════════════════════════════════

def part1_correlation_audit(signals: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """
    Correlate consecutive_wins with every other feature in the signal set.
    High correlation with price/momentum features = trend proxy confirmed.
    """
    if "consecutive_wins" not in signals.columns:
        return pd.DataFrame()

    cw = signals["consecutive_wins"].values.astype(float)
    rows = []

    for feat in feature_names:
        if feat == "consecutive_wins" or feat not in signals.columns:
            continue
        x = signals[feat].values.astype(float)
        valid = np.isfinite(x) & np.isfinite(cw)
        if valid.sum() < 20:
            continue

        pearson_r, pearson_p   = pearsonr(cw[valid], x[valid])
        spearman_r, spearman_p = spearmanr(cw[valid], x[valid])

        rows.append({
            "feature":      feat,
            "pearson_r":    round(float(pearson_r),  3),
            "pearson_p":    round(float(pearson_p),  4),
            "spearman_r":   round(float(spearman_r), 3),
            "spearman_p":   round(float(spearman_p), 4),
            "abs_corr":     round(abs(float(pearson_r)), 3),
        })

    return pd.DataFrame(rows).sort_values("abs_corr", ascending=False)


# ════════════════════════════════════════════════════════════════════════════
# PART 2 — CONDITIONAL IC DECOMPOSITION
# ════════════════════════════════════════════════════════════════════════════

def part2_conditional_ic(signals: pd.DataFrame,
                         feature_names: list[str],
                         pnl_col: str = "pnl_pts") -> pd.DataFrame:
    """
    Within each streak bucket, compute NW IC of each feature vs pnl_pts.
    If a feature's IC collapses inside high-streak bucket, consecutive_wins
    is absorbing its information.
    """
    if "consecutive_wins" not in signals.columns or pnl_col not in signals.columns:
        return pd.DataFrame()

    buckets = {
        "streak_0":    signals["consecutive_wins"] == 0,
        "streak_1_2":  (signals["consecutive_wins"] >= 1) & (signals["consecutive_wins"] <= 2),
        "streak_3plus": signals["consecutive_wins"] >= 3,
        "all":         pd.Series(True, index=signals.index),
    }

    rows = []
    y_all = signals[pnl_col].values.astype(float)

    for feat in feature_names:
        if feat == "consecutive_wins" or feat not in signals.columns:
            continue
        x_all = signals[feat].values.astype(float)

        row = {"feature": feat}
        for bucket_name, mask in buckets.items():
            idx = mask[mask].index
            if len(idx) < 15:
                row[f"ic_{bucket_name}"]  = float("nan")
                row[f"p_{bucket_name}"]   = float("nan")
                row[f"n_{bucket_name}"]   = len(idx)
                continue

            x_sub = x_all[signals.index.isin(idx)]
            y_sub = y_all[signals.index.isin(idx)]
            ic, p  = nw_ic(x_sub, y_sub)
            row[f"ic_{bucket_name}"] = ic
            row[f"p_{bucket_name}"]  = p
            row[f"n_{bucket_name}"]  = len(idx)

        # IC collapse: feature IC drops >50% in high-streak vs zero-streak bucket
        ic0 = row.get("ic_streak_0", float("nan"))
        ic3 = row.get("ic_streak_3plus", float("nan"))
        if not np.isnan(ic0) and not np.isnan(ic3) and abs(ic0) > 0.02:
            collapse_pct = (abs(ic0) - abs(ic3)) / abs(ic0) * 100
        else:
            collapse_pct = float("nan")
        row["ic_collapse_pct"] = round(float(collapse_pct), 1) \
            if not np.isnan(collapse_pct) else float("nan")

        rows.append(row)

    df = pd.DataFrame(rows)
    if "ic_collapse_pct" in df.columns:
        df = df.sort_values("ic_collapse_pct", ascending=False)
    return df


# ════════════════════════════════════════════════════════════════════════════
# PART 3 — REPLACEMENT CANDIDATE RANKING
# ════════════════════════════════════════════════════════════════════════════

def part3_candidate_ranking(signals: pd.DataFrame,
                            pnl_col: str = "pnl_pts") -> pd.DataFrame:
    """
    Rank the five candidate features by IC, overlap with consecutive_wins,
    and incremental IC after consecutive_wins is accounted for.
    """
    candidates = [
        "c1_multi_tf_trend_alignment",
        "c2_ema_stack_strength",
        "c3_price_momentum_persistence",
        "c4_session_trend_commitment",
        "c5_adx_slope",
    ]

    y = signals[pnl_col].values.astype(float)
    cw = signals["consecutive_wins"].values.astype(float) \
        if "consecutive_wins" in signals.columns else np.zeros(len(signals))

    rows = []
    for cand in candidates:
        if cand not in signals.columns:
            continue

        x = signals[cand].values.astype(float)

        # Raw IC
        ic, p = nw_ic(x, y)

        # Overlap with consecutive_wins
        valid_cw = np.isfinite(x) & np.isfinite(cw)
        overlap_r, _ = pearsonr(x[valid_cw], cw[valid_cw]) \
            if valid_cw.sum() > 10 else (0.0, 1.0)

        # Incremental IC: partial out consecutive_wins first
        # Residualize x on cw, then compute IC of residual vs y
        if valid_cw.sum() > 20 and np.std(cw[valid_cw]) > 0:
            cw_s = (cw - np.nanmean(cw)) / (np.nanstd(cw) + 1e-12)
            x_s  = (x  - np.nanmean(x))  / (np.nanstd(x)  + 1e-12)
            beta_cw = float(np.dot(x_s[valid_cw], cw_s[valid_cw]) /
                            (np.dot(cw_s[valid_cw], cw_s[valid_cw]) + 1e-12))
            x_resid = x_s - beta_cw * cw_s
            incr_ic, incr_p = nw_ic(x_resid, y)
        else:
            incr_ic, incr_p = ic, p

        rows.append({
            "candidate":        cand.replace("c1_", "").replace("c2_", "").replace(
                                    "c3_", "").replace("c4_", "").replace("c5_", ""),
            "raw_ic":           ic,
            "raw_p":            p,
            "overlap_with_cw":  round(float(overlap_r), 3),
            "incremental_ic":   incr_ic,
            "incremental_p":    incr_p,
            "significant":      "✅" if (p is not None and not np.isnan(p) and p < 0.05) else "—",
            "score": (
                abs(ic if ic == ic else 0) * 0.4 +
                (1 - abs(overlap_r)) * 0.3 +
                abs(incr_ic if incr_ic == incr_ic else 0) * 0.3
            ),
        })

    return pd.DataFrame(rows).sort_values("score", ascending=False)


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 80)
    print("STREAK DECOMPOSITION — Is consecutive_wins a trend proxy?")
    print(f"Vault period: {VAULT_START} → {VAULT_END}")
    print("=" * 80)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_corr = []
    all_cond_ic = []
    all_cand = []

    for symbol in ["NIFTY", "BANKNIFTY"]:
        print(f"\n{'─'*60}")
        print(f"Loading {symbol}…")

        raw_5m  = fetch_candles(symbol, timeframe="5m",  start_date=VAULT_START, end_date=VAULT_END)
        raw_15m = fetch_candles(symbol, timeframe="15m", start_date=VAULT_START, end_date=VAULT_END)

        if len(raw_5m) == 0:
            print(f"  ❌ No data for {symbol}")
            continue

        feat_5m  = add_features(raw_5m).copy()
        feat_5m["label"] = label_movement(feat_5m)
        feat_5m  = feat_5m.dropna(
            subset=MOVEMENT_FEATURES + ["label", "atr"]
        ).reset_index(drop=True)
        feat_15m = add_features(raw_15m).dropna(
            subset=MOVEMENT_FEATURES
        ).reset_index(drop=True)

        # Attach candidate features to the 5m data
        feat_5m_cands = compute_candidates(feat_5m)
        print(f"  Candidate features computed ✓")

        for direction in ["UP", "DOWN"]:
            print(f"\n  ── {symbol} {direction} ──")

            # Get approved signals
            ensemble = EnsemblePredictor.load(MODEL_DIR, symbol=symbol)
            signals  = generate_ensemble_signals(feat_5m, feat_15m, ensemble)
            dir_sigs = signals[signals["direction"] == direction].reset_index(drop=True)

            if len(dir_sigs) < 30:
                print(f"    ⚠  Only {len(dir_sigs)} signals — skipping")
                continue

            dir_sigs["win"] = label_signal_outcomes(dir_sigs, feat_5m)
            dir_sigs = build_meta_features(dir_sigs, feat_5m, symbol)

            # Load meta-filter to get feature names and run approval
            threshold = THRESHOLDS.get((symbol, direction), 0.55)
            meta_path = project_root / "backend" / "models" / \
                f"meta_filter_{symbol.lower()}_{direction.lower()}.joblib"
            if not meta_path.exists():
                meta_path = project_root / "backend" / "models" / \
                    f"meta_filter_ensemble_{symbol.lower()}.joblib"

            loaded     = joblib.load(meta_path)
            meta_model = loaded["model"] if isinstance(loaded, dict) else loaded
            feat_names = list(meta_model.get_booster().feature_names)

            probs    = meta_model.predict_proba(dir_sigs[feat_names].astype(float))[:, 1]
            approved = dir_sigs[probs >= threshold].copy()

            if len(approved) < 20:
                print(f"    ⚠  Only {len(approved)} approved signals — skipping")
                continue

            # Simulate to get continuous pnl_pts
            t_df, _ = simulate_portfolio(approved, feat_5m_cands)
            if len(t_df) == 0:
                print(f"    ⚠  No trades resolved — skipping")
                continue

            # Merge pnl_pts back onto approved signals
            t_df["timestamp_merge"] = pd.to_datetime(
                t_df["entry_ts"]
            ).dt.tz_localize(None)
            approved["timestamp_merge"] = pd.to_datetime(
                approved["timestamp"]
            ).dt.tz_localize(None)
            approved = approved.merge(
                t_df[["timestamp_merge", "pnl_pts"]],
                on="timestamp_merge", how="inner",
            )

            # Merge candidate features onto approved signals
            cand_cols = [c for c in feat_5m_cands.columns if c.startswith("c")]
            if cand_cols:
                cand_df = feat_5m_cands[["timestamp"] + cand_cols].copy()
                cand_df["timestamp"] = pd.to_datetime(
                    cand_df["timestamp"]
                ).dt.tz_localize(None)
                approved["timestamp_merge"] = pd.to_datetime(
                    approved["timestamp"]
                ).dt.tz_localize(None)
                approved = pd.merge_asof(
                    approved.sort_values("timestamp_merge"),
                    cand_df.rename(columns={"timestamp": "timestamp_merge"}).sort_values("timestamp_merge"),
                    on="timestamp_merge",
                    direction="backward",
                )

            n_sigs = len(approved)
            streak_dist = approved["consecutive_wins"].value_counts().sort_index()
            print(f"    Approved signals with pnl: {n_sigs}")
            print(f"    consecutive_wins distribution: "
                  f"0={streak_dist.get(0,0)}  "
                  f"1-2={(streak_dist.get(1,0)+streak_dist.get(2,0))}  "
                  f"3+={streak_dist[streak_dist.index>=3].sum()}")

            # ── PART 1 ─────────────────────────────────────────────────────
            print(f"\n    [Part 1] Correlation audit…")
            corr_df = part1_correlation_audit(approved, feat_names)
            if len(corr_df) > 0:
                corr_df.insert(0, "symbol",    symbol)
                corr_df.insert(1, "direction", direction)
                all_corr.append(corr_df)

                top5 = corr_df.head(5)
                print(f"    Top 5 features correlated with consecutive_wins:")
                for _, row in top5.iterrows():
                    print(f"      {row['feature']:35s} r={row['pearson_r']:+.3f}  "
                          f"p={row['pearson_p']:.4f}")

                # Verdict
                price_features = {
                    "regime_trend", "z_score_ema_slope", "z_score_roc_5",
                    "vol_ratio", "z_score_20", "regime_time_of_day",
                }
                top3_names = set(corr_df.head(3)["feature"].tolist())
                price_overlap = top3_names & price_features
                if price_overlap:
                    print(f"    ⚠  consecutive_wins correlates most with "
                          f"price features {price_overlap} → TREND PROXY CONFIRMED")
                else:
                    print(f"    ✓  consecutive_wins not primarily correlated "
                          f"with price features → may be genuine pipeline signal")

            # ── PART 2 ─────────────────────────────────────────────────────
            print(f"\n    [Part 2] Conditional IC decomposition…")
            cond_ic_df = part2_conditional_ic(approved, feat_names)
            if len(cond_ic_df) > 0:
                cond_ic_df.insert(0, "symbol",    symbol)
                cond_ic_df.insert(1, "direction", direction)
                all_cond_ic.append(cond_ic_df)

                # Show features with >50% IC collapse in high-streak bucket
                collapsed = cond_ic_df[
                    cond_ic_df["ic_collapse_pct"].notna() &
                    (cond_ic_df["ic_collapse_pct"] > 50)
                ].head(5)

                if len(collapsed) > 0:
                    print(f"    Features whose IC collapses in high-streak bucket "
                          f"(absorbed by consecutive_wins):")
                    for _, row in collapsed.iterrows():
                        print(f"      {row['feature']:35s}  "
                              f"IC(streak=0)={row['ic_streak_0']:+.3f}  "
                              f"IC(streak=3+)={row['ic_streak_3plus']:+.3f}  "
                              f"collapse={row['ic_collapse_pct']:.0f}%")
                else:
                    print(f"    No major IC collapse detected — "
                          f"price features retain IC in high-streak bucket")

            # ── PART 3 ─────────────────────────────────────────────────────
            print(f"\n    [Part 3] Replacement candidate ranking…")
            cand_df_res = part3_candidate_ranking(approved)
            if len(cand_df_res) > 0:
                cand_df_res.insert(0, "symbol",    symbol)
                cand_df_res.insert(1, "direction", direction)
                all_cand.append(cand_df_res)

                print(f"    Candidates ranked (score = 0.4×IC + 0.3×low_overlap "
                      f"+ 0.3×incremental_IC):")
                for _, row in cand_df_res.iterrows():
                    print(f"      {row['candidate']:30s}  "
                          f"IC={row['raw_ic']:+.3f} {row['significant']}  "
                          f"overlap_cw={row['overlap_with_cw']:+.3f}  "
                          f"incr_IC={row['incremental_ic']:+.3f}  "
                          f"score={row['score']:.3f}")

    # ── Save all outputs ────────────────────────────────────────────────────
    if all_corr:
        pd.concat(all_corr, ignore_index=True).to_csv(
            OUTPUT_DIR / "streak_decomposition_correlations.csv", index=False
        )
    if all_cond_ic:
        pd.concat(all_cond_ic, ignore_index=True).to_csv(
            OUTPUT_DIR / "streak_decomposition_conditional_ic.csv", index=False
        )
    if all_cand:
        cand_out = pd.concat(all_cand, ignore_index=True)
        cand_out.to_csv(
            OUTPUT_DIR / "streak_decomposition_candidates.csv", index=False
        )

    # ── Consolidated recommendation ─────────────────────────────────────────
    print("\n" + "=" * 80)
    print("CONSOLIDATED RECOMMENDATION")
    print("=" * 80)

    if all_cand:
        cand_all = pd.concat(all_cand, ignore_index=True)

        # Average score across all models per candidate
        avg_scores = (
            cand_all.groupby("candidate")["score"]
            .mean()
            .sort_values(ascending=False)
        )
        avg_ic = cand_all.groupby("candidate")["raw_ic"].mean()
        avg_sig = cand_all.groupby("candidate")["significant"].apply(
            lambda x: (x == "✅").sum()
        )

        print("\n  Candidates averaged across all symbol/direction combinations:")
        for cand, score in avg_scores.items():
            n_sig = avg_sig.get(cand, 0)
            n_total = (cand_all["candidate"] == cand).sum()
            print(f"    {cand:30s}  "
                  f"avg_score={score:.3f}  "
                  f"avg_IC={avg_ic.get(cand, float('nan')):+.3f}  "
                  f"significant in {n_sig}/{n_total} models")

        # Top candidate
        top_cand = avg_scores.index[0]
        top_score = avg_scores.iloc[0]
        top_sig = avg_sig.get(top_cand, 0)
        n_total = (cand_all["candidate"] == top_cand).sum()

        print(f"\n  ── RECOMMENDATION ──")
        if top_sig >= 3 and top_score > 0.05:
            print(f"  Replace `consecutive_wins` with `{top_cand}`")
            print(f"  Significant in {top_sig}/{n_total} model combinations, "
                  f"avg score {top_score:.3f}")
            print(f"  Next step: add this feature to META_FEATURES in "
                  f"train_meta_filter.py, remove JSD and consecutive_wins, "
                  f"retrain all four directional models, vault-validate.")
        elif top_sig >= 2:
            print(f"  `{top_cand}` shows marginal evidence ({top_sig}/{n_total} significant)")
            print(f"  Consider keeping `consecutive_wins` but at reduced weight — "
                  f"run with `min_child_weight=20` to reduce its dominance.")
            print(f"  Alternatively: keep consecutive_wins, add top candidate as "
                  f"supplementary feature by removing JSD slot.")
        else:
            print(f"  No candidate consistently outperforms on IC metric.")
            print(f"  `consecutive_wins` dominance may be genuine pipeline autocorrelation,")
            print(f"  not pure trend proxy. Do not remove without further evidence.")
            print(f"  Next step: run the 2-day lag test described in the project context.")

    print(f"\nOutputs saved to {OUTPUT_DIR}/streak_decomposition_*.csv")
    print("=" * 80)


if __name__ == "__main__":
    main()


# =============================================================================
# INTERPRETATION GUIDE
# =============================================================================
#
# Part 1 — Correlation audit:
#   If consecutive_wins correlates most strongly with regime_trend,
#   z_score_ema_slope, z_score_roc_5 → trend proxy confirmed.
#   r > 0.3 with any of these is a strong signal.
#
# Part 2 — Conditional IC:
#   Features where IC drops >50% in streak_3plus vs streak_0 bucket:
#   These features are being made redundant by consecutive_wins.
#   If many price features collapse → consecutive_wins is a compressed
#   representation of the trend state. Replacing it with a direct trend
#   feature unlocks those features again.
#
# Part 3 — Candidate ranking:
#   score = 0.4×|IC| + 0.3×(1-|overlap_with_cw|) + 0.3×|incremental_IC|
#   High IC: predicts P&L magnitude, not just direction
#   Low overlap: brings new information, doesn't duplicate consecutive_wins
#   High incremental IC: adds value even if consecutive_wins stays
#
#   Candidates that score well AND have low overlap make the strongest
#   replacement case. Candidates with high IC but high overlap are only
#   useful if you're removing consecutive_wins entirely.
#
# =============================================================================
