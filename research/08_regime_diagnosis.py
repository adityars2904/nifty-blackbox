#!/usr/bin/env python3
"""
08_regime_diagnosis.py
======================
Autopsy: Why does the regime classifier hurt BANKNIFTY (-2% to -3.6%) while
helping NIFTY UP (+1.2%)?

Three hypotheses tested in order:

  H1 — WRONG THRESHOLDS
       The ADX 25/20 and vol_ratio 1.3x thresholds are calibrated for NIFTY.
       BANKNIFTY has higher baseline volatility and faster trend cycles.
       Test: Sweep thresholds and find the optimal values per symbol.
       Verdict: If optimal thresholds differ substantially between symbols,
                H1 is confirmed.

  H2 — WRONG ARCHITECTURE
       Using regime as a position-sizer (halving size in VOLATILE/CHOPPY)
       is the wrong lever. BANKNIFTY makes large moves in volatile regimes.
       The edge doesn't shrink in volatile regimes — it grows.
       Test: Measure per-regime win rate and expectancy *before* any sizing
             adjustment. If win rate in VOLATILE > win rate in TRENDING,
             H2 is confirmed — the classifier is penalising the best regime.

  H3 — NO STABLE REGIMES
       BANKNIFTY regime transitions are too fast for 75-bar trailing averages
       to classify reliably. By the time the label stabilises, the regime
       has already changed.
       Test: Measure regime persistence (autocorrelation of regime label),
             and measure the accuracy of T-1 regime label at predicting
             T+1 signal quality. If persistence < 0.5 or predictive accuracy
             is near random, H3 is confirmed.

Output: research/outputs/regime_diagnosis.csv + printed verdict per hypothesis.

Usage:
    cd research && python 08_regime_diagnosis.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from itertools import product as iterproduct

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

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

# ── Regime classifier (mirrors 04_regime_analysis.py exactly) ──────────────

def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high  = df["high"].astype(float)
    low   = df["low"].astype(float)
    close = df["close"].astype(float)
    plus_dm  = high.diff()
    minus_dm = -low.diff()
    plus_dm  = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr      = tr.rolling(period).mean()
    plus_di  = 100 * (plus_dm.rolling(period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
    dx  = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    return dx.rolling(period).mean()


def classify_regime(adx_val: float, vol_ratio: float,
                    adx_trend: float, adx_choppy: float,
                    vol_thresh: float) -> str:
    if pd.isna(adx_val) or pd.isna(vol_ratio):
        return "MIXED"
    if vol_ratio > vol_thresh:
        return "VOLATILE"
    if adx_val > adx_trend:
        return "TRENDING"
    if adx_val < adx_choppy:
        return "CHOPPY"
    return "MIXED"


def attach_regime(feat_5m: pd.DataFrame,
                  adx_trend: float = 25, adx_choppy: float = 20,
                  vol_thresh: float = 1.3,
                  trailing_bars: int = 75) -> pd.DataFrame:
    """Attach trailing regime label to a feature DataFrame."""
    df = feat_5m.copy()
    df["adx"] = compute_adx(df)
    df["atr_75"] = df["atr"].rolling(trailing_bars).mean()
    df["vol_ratio_reg"] = df["atr"] / df["atr_75"].replace(0, np.nan)
    df["trailing_adx"] = df["adx"].rolling(trailing_bars).mean().shift(1)
    df["trailing_vol"]  = df["vol_ratio_reg"].rolling(trailing_bars).mean().shift(1)
    df["regime"] = df.apply(
        lambda r: classify_regime(
            r["trailing_adx"], r["trailing_vol"],
            adx_trend, adx_choppy, vol_thresh,
        ), axis=1
    )
    return df


def get_approved_signals(symbol, feat_5m, feat_15m, direction):
    """Return meta-filter-approved signals for one symbol/direction."""
    ensemble = EnsemblePredictor.load(MODEL_DIR, symbol=symbol)
    signals  = generate_ensemble_signals(feat_5m, feat_15m, ensemble)
    dir_sigs = signals[signals["direction"] == direction].reset_index(drop=True)
    if len(dir_sigs) == 0:
        return pd.DataFrame()

    dir_sigs["win"] = label_signal_outcomes(dir_sigs, feat_5m)
    dir_sigs = build_meta_features(dir_sigs, feat_5m, symbol)

    threshold = THRESHOLDS.get((symbol, direction), 0.55)
    meta_path = project_root / "backend" / "models" / \
        f"meta_filter_{symbol.lower()}_{direction.lower()}.joblib"
    if not meta_path.exists():
        meta_path = project_root / "backend" / "models" / \
            f"meta_filter_ensemble_{symbol.lower()}.joblib"

    loaded     = joblib.load(meta_path)
    meta_model = loaded["model"] if isinstance(loaded, dict) else loaded
    feat_names = list(meta_model.get_booster().feature_names)
    probs      = meta_model.predict_proba(dir_sigs[feat_names].astype(float))[:, 1]
    return dir_sigs[probs >= threshold].copy()


# ════════════════════════════════════════════════════════════════════════════
# H1 — THRESHOLD SWEEP
# ════════════════════════════════════════════════════════════════════════════

def h1_threshold_sweep(approved: pd.DataFrame, feat_5m_with_regime_cols: pd.DataFrame,
                       symbol: str, direction: str) -> pd.DataFrame:
    """
    Sweep ADX and vol thresholds.
    For each combination, measure regime-weighted vs baseline expectancy.
    Returns a DataFrame of results sorted by regime-adjusted improvement.
    """
    adx_trend_vals  = [18, 22, 25, 28, 32]
    adx_choppy_vals = [14, 18, 20, 22]
    vol_thresh_vals = [1.15, 1.25, 1.35, 1.50]

    base_trades, _ = simulate_portfolio(approved, feat_5m_with_regime_cols)
    base_exp = base_trades["pnl_pts"].mean() if len(base_trades) > 0 else 0.0

    rows = []
    for adx_t, adx_c, vol_t in iterproduct(adx_trend_vals, adx_choppy_vals, vol_thresh_vals):
        if adx_c >= adx_t:
            continue  # invalid combination

        # Re-classify with these thresholds
        df_reg = attach_regime(
            feat_5m_with_regime_cols,
            adx_trend=adx_t, adx_choppy=adx_c, vol_thresh=vol_t,
        )

        # Merge regime onto approved signals
        sigs = approved.copy()
        if "regime" in sigs.columns:
            sigs = sigs.drop(columns=["regime"])
        sigs = sigs.merge(
            df_reg[["timestamp", "regime"]].drop_duplicates("timestamp"),
            on="timestamp", how="left",
        )
        sigs["regime"] = sigs["regime"].fillna("MIXED")

        # Regime scalars (same as 04_regime_analysis.py)
        SCALARS = {"TRENDING": 1.0, "VOLATILE": 0.5, "CHOPPY": 0.5, "MIXED": 0.8}
        sigs["risk_scalar"] = sigs["regime"].map(SCALARS).fillna(0.8)

        t_df, _ = simulate_portfolio(sigs, feat_5m_with_regime_cols)
        adj_exp = t_df["pnl_pts"].mean() if len(t_df) > 0 else 0.0
        improvement = adj_exp - base_exp

        # Distribution across regimes for these signals
        regime_counts = sigs["regime"].value_counts()

        rows.append({
            "symbol":     symbol,
            "direction":  direction,
            "adx_trend":  adx_t,
            "adx_choppy": adx_c,
            "vol_thresh": vol_t,
            "base_exp":   round(base_exp, 2),
            "adj_exp":    round(adj_exp, 2),
            "improvement": round(improvement, 2),
            "pct_trending": round(regime_counts.get("TRENDING", 0) / max(len(sigs), 1) * 100, 1),
            "pct_volatile": round(regime_counts.get("VOLATILE", 0) / max(len(sigs), 1) * 100, 1),
            "pct_choppy":   round(regime_counts.get("CHOPPY",   0) / max(len(sigs), 1) * 100, 1),
        })

    return pd.DataFrame(rows).sort_values("improvement", ascending=False)


# ════════════════════════════════════════════════════════════════════════════
# H2 — ARCHITECTURE: measure per-regime expectancy BEFORE sizing
# ════════════════════════════════════════════════════════════════════════════

def h2_per_regime_expectancy(approved: pd.DataFrame,
                              feat_5m_with_regime: pd.DataFrame,
                              symbol: str, direction: str) -> pd.DataFrame:
    """
    For each regime bucket, compute raw (unsized) expectancy.
    If VOLATILE regime has higher expectancy than TRENDING, the position-sizer
    is penalising the best setup — H2 confirmed.
    """
    sigs = approved.copy()
    if "regime" in sigs.columns:
        sigs = sigs.drop(columns=["regime"])
    sigs = sigs.merge(
        feat_5m_with_regime[["timestamp", "regime"]].drop_duplicates("timestamp"),
        on="timestamp", how="left",
    )
    sigs["regime"] = sigs["regime"].fillna("MIXED")
    sigs["risk_scalar"] = 1.0  # Unsized — equal weight across all regimes

    rows = []
    # Overall baseline (all regimes together, no scaling)
    t_all, _ = simulate_portfolio(sigs, feat_5m_with_regime)
    if len(t_all) > 0:
        rows.append({
            "regime":        "ALL",
            "n_signals":     len(sigs),
            "win_rate_pct":  round((t_all["pnl_pts"] > 0).mean() * 100, 1),
            "avg_win_pts":   round(t_all[t_all["pnl_pts"] > 0]["pnl_pts"].mean(), 1),
            "avg_loss_pts":  round(abs(t_all[t_all["pnl_pts"] <= 0]["pnl_pts"].mean()), 1),
            "expectancy_pts": round(t_all["pnl_pts"].mean(), 2),
            "regime_scalar_applied": "1.0 (none)",
        })

    for regime in ["TRENDING", "VOLATILE", "CHOPPY", "MIXED"]:
        regime_sigs = sigs[sigs["regime"] == regime].copy()
        if len(regime_sigs) < 5:
            continue
        t_df, _ = simulate_portfolio(regime_sigs, feat_5m_with_regime)
        if len(t_df) == 0:
            continue

        SCALARS = {"TRENDING": 1.0, "VOLATILE": 0.5, "CHOPPY": 0.5, "MIXED": 0.8}
        rows.append({
            "regime":        regime,
            "n_signals":     len(regime_sigs),
            "win_rate_pct":  round((t_df["pnl_pts"] > 0).mean() * 100, 1),
            "avg_win_pts":   round(t_df[t_df["pnl_pts"] > 0]["pnl_pts"].mean(), 1),
            "avg_loss_pts":  round(abs(t_df[t_df["pnl_pts"] <= 0]["pnl_pts"].mean()), 1),
            "expectancy_pts": round(t_df["pnl_pts"].mean(), 2),
            "regime_scalar_applied": str(SCALARS[regime]),
        })

    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════════════
# H3 — STABILITY: how persistent and predictive are regime labels?
# ════════════════════════════════════════════════════════════════════════════

def h3_regime_stability(feat_5m_with_regime: pd.DataFrame,
                        approved: pd.DataFrame,
                        symbol: str, direction: str) -> dict:
    """
    Two tests:
    1. Regime persistence: autocorrelation of regime label over 5-bar horizon
       (encoded as int: TRENDING=0, VOLATILE=1, CHOPPY=2, MIXED=3)
    2. Predictive accuracy: does T-1 regime label predict T signal win/loss
       better than chance?
    """
    reg_map = {"TRENDING": 0, "VOLATILE": 1, "CHOPPY": 2, "MIXED": 3}
    reg_int = feat_5m_with_regime["regime"].map(reg_map).fillna(3)

    # Persistence: correlation between label[t] and label[t+5]
    lag5 = reg_int.shift(-5)
    valid = ~(reg_int.isna() | lag5.isna())
    if valid.sum() > 100:
        corr_5bar, _ = spearmanr(reg_int[valid], lag5[valid])
    else:
        corr_5bar = float("nan")

    lag20 = reg_int.shift(-20)  # ~100 minutes
    valid20 = ~(reg_int.isna() | lag20.isna())
    if valid20.sum() > 100:
        corr_20bar, _ = spearmanr(reg_int[valid20], lag20[valid20])
    else:
        corr_20bar = float("nan")

    # Predictive accuracy: for each approved signal, did the T-1 regime
    # correctly predict whether the trade won?
    sigs = approved.copy()
    if "regime" in sigs.columns:
        sigs = sigs.drop(columns=["regime"])
    sigs = sigs.merge(
        feat_5m_with_regime[["timestamp", "regime"]].drop_duplicates("timestamp"),
        on="timestamp", how="left",
    )
    sigs["regime"] = sigs["regime"].fillna("MIXED")

    # Naive prediction: TRENDING=win, VOLATILE/CHOPPY=loss, MIXED=neutral
    sigs["regime_predicts_win"] = sigs["regime"].map(
        {"TRENDING": 1, "VOLATILE": 0, "CHOPPY": 0, "MIXED": None}
    )
    predictable = sigs.dropna(subset=["regime_predicts_win"])
    if len(predictable) >= 20:
        naive_acc = (
            (predictable["regime_predicts_win"] == predictable["win"])
            .mean()
        )
    else:
        naive_acc = float("nan")

    # Per-regime win rate (how much does regime actually discriminate?)
    per_regime_wr = sigs.groupby("regime")["win"].mean().to_dict()

    # Max discrimination = difference between best and worst regime win rate
    wrs = [v for v in per_regime_wr.values() if not np.isnan(v)]
    discrimination = max(wrs) - min(wrs) if len(wrs) >= 2 else 0.0

    return {
        "symbol":              symbol,
        "direction":           direction,
        "regime_autocorr_5bar":  round(float(corr_5bar),  3),
        "regime_autocorr_20bar": round(float(corr_20bar), 3),
        "naive_predictive_acc":  round(float(naive_acc),  3),
        "regime_discrimination": round(discrimination, 3),
        "per_regime_wr":         {k: round(v, 3) for k, v in per_regime_wr.items()},
        "n_signals":             len(sigs),
    }


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 80)
    print("REGIME DIAGNOSIS — H1/H2/H3 Autopsy")
    print(f"Vault period: {VAULT_START} → {VAULT_END}")
    print("=" * 80)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_h1 = []
    all_h2 = []
    all_h3 = []

    for symbol in ["NIFTY", "BANKNIFTY"]:
        print(f"\n{'─'*60}")
        print(f"Loading {symbol} vault data…")

        raw_5m  = fetch_candles(symbol, timeframe="5m",  start_date=VAULT_START, end_date=VAULT_END)
        raw_15m = fetch_candles(symbol, timeframe="15m", start_date=VAULT_START, end_date=VAULT_END)

        if len(raw_5m) == 0:
            print(f"  ❌ No data for {symbol}, skipping")
            continue

        feat_5m  = add_features(raw_5m).copy()
        feat_5m["label"] = label_movement(feat_5m)
        feat_5m  = feat_5m.dropna(
            subset=MOVEMENT_FEATURES + ["label", "atr"]
        ).reset_index(drop=True)
        feat_15m = add_features(raw_15m).dropna(
            subset=MOVEMENT_FEATURES
        ).reset_index(drop=True)

        # Attach default regime labels (ADX 25/20, vol 1.3)
        feat_5m_reg = attach_regime(feat_5m)

        # Print baseline regime distribution
        dist = feat_5m_reg["regime"].value_counts()
        print(f"\n  Baseline regime distribution ({symbol}):")
        for r, n in dist.items():
            print(f"    {r:10s}: {n:5d} bars  ({n/len(feat_5m_reg)*100:.1f}%)")

        for direction in (["UP", "DOWN"] if symbol == "NIFTY" else ["UP", "DOWN"]):
            print(f"\n  ── {symbol} {direction} ──")
            approved = get_approved_signals(symbol, feat_5m, feat_15m, direction)

            if len(approved) < 10:
                print(f"    ⚠  Only {len(approved)} approved signals — skipping")
                continue

            print(f"    Approved signals: {len(approved)}")

            # ── H2 first (cheapest, most informative) ──────────────────────
            print(f"\n  [H2] Per-regime expectancy (unsized)…")
            h2 = h2_per_regime_expectancy(approved, feat_5m_reg, symbol, direction)
            h2.insert(0, "symbol",    symbol)
            h2.insert(1, "direction", direction)
            all_h2.append(h2)
            print(h2[["regime", "n_signals", "win_rate_pct",
                        "expectancy_pts", "regime_scalar_applied"]].to_string(index=False))

            # Detect H2 verdict
            if len(h2) >= 2:
                volatile_row = h2[h2["regime"] == "VOLATILE"]
                trending_row = h2[h2["regime"] == "TRENDING"]
                if len(volatile_row) > 0 and len(trending_row) > 0:
                    v_exp = volatile_row["expectancy_pts"].values[0]
                    t_exp = trending_row["expectancy_pts"].values[0]
                    if v_exp > t_exp:
                        print(f"\n  ⚠  H2 SIGNAL: VOLATILE exp ({v_exp:.1f}) "
                              f"> TRENDING exp ({t_exp:.1f}) "
                              f"— position-sizer is penalising the best regime")

            # ── H3 (fast) ──────────────────────────────────────────────────
            print(f"\n  [H3] Regime stability / predictive power…")
            h3 = h3_regime_stability(feat_5m_reg, approved, symbol, direction)
            all_h3.append(h3)
            print(f"    Regime autocorr (5-bar):  {h3['regime_autocorr_5bar']:.3f}")
            print(f"    Regime autocorr (20-bar): {h3['regime_autocorr_20bar']:.3f}")
            print(f"    Naive predictive accuracy: {h3['naive_predictive_acc']:.3f} (0.5 = chance)")
            print(f"    Regime discrimination:     {h3['regime_discrimination']:.3f} (win rate spread)")
            print(f"    Per-regime win rates:      {h3['per_regime_wr']}")

            if h3["regime_autocorr_5bar"] < 0.5:
                print(f"    ⚠  H3 SIGNAL: Low regime persistence — labels flip too fast to be reliable")
            if h3["naive_predictive_acc"] < 0.55:
                print(f"    ⚠  H3 SIGNAL: Near-random predictive accuracy")
            if h3["regime_discrimination"] < 0.05:
                print(f"    ⚠  H3 SIGNAL: Win rate spread across regimes < 5pp — no meaningful structure")

            # ── H1 (expensive — runs last) ─────────────────────────────────
            print(f"\n  [H1] Threshold sweep (this takes ~30s)…")
            h1 = h1_threshold_sweep(approved, feat_5m_reg, symbol, direction)
            all_h1.append(h1)

            best = h1.iloc[0]
            current = h1[(h1["adx_trend"] == 25) &
                         (h1["adx_choppy"] == 20) &
                         (h1["vol_thresh"] == 1.3)]
            current_imp = current["improvement"].values[0] if len(current) > 0 else float("nan")

            print(f"    Current thresholds (ADX 25/20, vol 1.3):  "
                  f"improvement = {current_imp:+.2f} pts")
            print(f"    Best thresholds found: ADX {best['adx_trend']:.0f}/{best['adx_choppy']:.0f}, "
                  f"vol {best['vol_thresh']:.2f}")
            print(f"    Best improvement: {best['improvement']:+.2f} pts  "
                  f"(regime mix: {best['pct_trending']:.0f}% TREND / "
                  f"{best['pct_volatile']:.0f}% VOL / "
                  f"{best['pct_choppy']:.0f}% CHOP)")

            h1_diff = best["improvement"] - current_imp
            if h1_diff > 2.0:
                print(f"    ⚠  H1 SIGNAL: Re-tuning thresholds would gain {h1_diff:+.2f} pts")
            else:
                print(f"    ✓  H1: Threshold re-tuning gain is marginal ({h1_diff:+.2f} pts)")

    # ── Save outputs ────────────────────────────────────────────────────────
    if all_h2:
        h2_df = pd.concat(all_h2, ignore_index=True)
        h2_df.to_csv(OUTPUT_DIR / "regime_diagnosis_h2_per_regime.csv", index=False)

    if all_h1:
        h1_df = pd.concat(all_h1, ignore_index=True)
        h1_df.to_csv(OUTPUT_DIR / "regime_diagnosis_h1_sweep.csv", index=False)

    if all_h3:
        h3_df = pd.DataFrame(all_h3).drop(columns=["per_regime_wr"])
        h3_df.to_csv(OUTPUT_DIR / "regime_diagnosis_h3_stability.csv", index=False)

    # ── Consolidated verdict ─────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("CONSOLIDATED VERDICT")
    print("=" * 80)

    if all_h2 and all_h3:
        for h3 in all_h3:
            sym = h3["symbol"]
            dirn = h3["direction"]
            h2_match = [
                df for df in all_h2
                if (df["symbol"] == sym).all() and (df["direction"] == dirn).all()
            ]

            signals = []

            # H2 check
            if h2_match:
                h2_df_single = h2_match[0]
                v_row = h2_df_single[h2_df_single["regime"] == "VOLATILE"]
                t_row = h2_df_single[h2_df_single["regime"] == "TRENDING"]
                if len(v_row) > 0 and len(t_row) > 0:
                    if v_row["expectancy_pts"].values[0] > t_row["expectancy_pts"].values[0]:
                        signals.append("H2: position-sizer penalises best regime")

            # H3 checks
            if h3["regime_autocorr_5bar"] < 0.5:
                signals.append("H3: regime labels flip too fast (low persistence)")
            if h3["naive_predictive_acc"] < 0.55:
                signals.append("H3: near-random predictive accuracy")
            if h3["regime_discrimination"] < 0.05:
                signals.append("H3: <5pp win rate spread — no useful structure")

            if signals:
                print(f"\n  {sym} {dirn}: REGIME CLASSIFIER IS HARMFUL")
                for s in signals:
                    print(f"    → {s}")
                print(f"    Recommendation: Remove regime position-sizer. "
                      f"Add regime as raw feature input to meta-filter instead.")
            else:
                print(f"\n  {sym} {dirn}: Regime classifier is working as intended")
                print(f"    Recommendation: Keep, but consider H1 threshold tuning if gain > 2pts")

    print(f"\nOutputs saved to {OUTPUT_DIR}/regime_diagnosis_*.csv")
    print("=" * 80)


if __name__ == "__main__":
    main()


# =============================================================================
# INTERPRETATION GUIDE
# =============================================================================
#
# H1 confirmed (gain > 2pts from re-tuning):
#   → Run per-symbol threshold optimisation. Use the best (adx_trend, adx_choppy,
#     vol_thresh) values per symbol in 04_regime_analysis.py.
#
# H2 confirmed (VOLATILE expectancy > TRENDING expectancy):
#   → The position-sizer architecture is wrong for this model.
#   → Immediate fix: set VOLATILE scalar to 1.0 (same as TRENDING).
#   → Better fix (Track C): promote regime label to meta-filter feature input.
#     Add `intraday_vol_percentile` and `session_range_percentile` as continuous
#     features and let XGBoost weight them. Remove the hard gate entirely.
#
# H3 confirmed (autocorr < 0.5 OR discrimination < 0.05):
#   → The regime classifier is producing noise, not signal.
#   → Remove it entirely. Do not replace with a different classifier.
#   → Instead, add the raw underlying metrics (ADX value, vol_ratio) as
#     continuous meta-filter features — the model will find the right
#     non-linear threshold itself.
#
# Multiple hypotheses confirmed simultaneously:
#   → H2 + H3 together = the regime gate is both architecturally wrong AND
#     operating on noisy labels. Full removal is warranted.
#   → H1 + H2 = wrong thresholds AND wrong architecture. Fix architecture first.
#
# =============================================================================
