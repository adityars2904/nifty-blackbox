#!/usr/bin/env python3
"""
Smoke Test — validates the entire ML pipeline end-to-end.

Run manually:
    cd backend && python smoke_test.py

Does NOT write to any database, start any server, or import services/routers.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Ensure backend/ is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> None:
    import joblib
    import numpy as np
    import pandas as pd

    from config import MODELS_DIR, META_FILTER_PROB_MIN, IST
    from adapters.questdb_adapter import init_pool, fetch_candles, resample_to_5m, resample_to_15m
    from ml.feature_engineering import add_features, MOVEMENT_FEATURES
    from ml.ensemble_predictor import EnsemblePredictor

    t0 = time.perf_counter()

    # ── Step 1: Load all 6 models ────────────────────────────────────────────
    print("=" * 60)
    print("STEP 1 — Loading models")
    print("=" * 60)

    model_files = [
        "movement_predictor_nifty_5m.joblib",
        "movement_predictor_nifty_15m.joblib",
        "movement_predictor_banknifty_5m.joblib",
        "movement_predictor_banknifty_15m.joblib",
        "meta_filter_ensemble_nifty.joblib",
        "meta_filter_ensemble_banknifty.joblib",
    ]

    models: dict = {}
    for fname in model_files:
        fpath = Path(MODELS_DIR) / fname
        if not fpath.exists():
            print(f"  ❌ MISSING: {fpath}")
            sys.exit(1)
        models[fname] = joblib.load(fpath)
        print(f"  ✅ Loaded: {fname}")

    # ── Step 2: Connect to QuestDB ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2 — Connecting to QuestDB")
    print("=" * 60)

    init_pool()
    print("  ✅ QuestDB connection pool OK")

    # ── Step 3: Fetch candles ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3 — Fetching candles")
    print("=" * 60)

    for symbol in ("NIFTY", "BANKNIFTY"):
        # Need at least 50 bars per timeframe after resampling.
        # 1000 1m candles → ~200 5m, ~66 15m — comfortably above the 50-bar minimum.
        df_1m = fetch_candles(symbol, 1000)
        print(f"  {symbol}: {len(df_1m)} 1-min candles (latest: {df_1m['ts'].iloc[-1]})")

        # ── Step 4: Feature engineering ──────────────────────────────────────
        df_5m = resample_to_5m(df_1m)
        df_15m = resample_to_15m(df_1m)

        print(f"  {symbol}: Resampled → {len(df_5m)} 5m, {len(df_15m)} 15m candles")

        # Verify features can be computed (validation only — ensemble computes its own)
        df_5m_check = add_features(df_5m.copy())
        df_15m_check = add_features(df_15m.copy())

        feat_cols_5m = [c for c in df_5m_check.columns if c in MOVEMENT_FEATURES]
        feat_cols_15m = [c for c in df_15m_check.columns if c in MOVEMENT_FEATURES]

        if len(feat_cols_5m) != 17:
            print(f"  ❌ 5m feature count = {len(feat_cols_5m)}, expected 17")
            print(f"     Present: {feat_cols_5m}")
            missing = set(MOVEMENT_FEATURES) - set(feat_cols_5m)
            if missing:
                print(f"     Missing: {missing}")
            sys.exit(1)

        if len(feat_cols_15m) != 17:
            print(f"  ❌ 15m feature count = {len(feat_cols_15m)}, expected 17")
            print(f"     Present: {feat_cols_15m}")
            missing = set(MOVEMENT_FEATURES) - set(feat_cols_15m)
            if missing:
                print(f"     Missing: {missing}")
            sys.exit(1)

        print(f"  ✅ {symbol}: Feature engineering OK (17 features)")

        # ── Step 5: Ensemble prediction ──────────────────────────────────────
        # Pass raw resampled data — EnsemblePredictor._extract_features() adds features internally
        ensemble = EnsemblePredictor.load(model_dir=MODELS_DIR, symbol=symbol)
        pred = ensemble.predict(df_5m, df_15m)

        print(f"\n  --- {symbol} Ensemble Prediction ---")
        print(f"  Direction:  {pred.direction}")
        print(f"  Confidence: {pred.confidence:.4f}")
        print(f"  P(UP):      {pred.prob_up:.4f}")
        print(f"  P(DOWN):    {pred.prob_down:.4f}")
        print(f"  P(NEUTRAL): {pred.prob_neutral:.4f}")
        print(f"  Should trade: {pred.should_trade}")
        if pred.veto_reason:
            print(f"  Veto reason:  {pred.veto_reason}")

        # ── Step 6: Meta-filter ──────────────────────────────────────────────
        meta_model_key = f"meta_filter_ensemble_{symbol.lower()}.joblib"
        meta_model_obj = models[meta_model_key]
        # The joblib file stores a dict with keys: model, metrics, symbol
        meta_model = meta_model_obj["model"] if isinstance(meta_model_obj, dict) else meta_model_obj

        # Build a minimal 19-feature vector for the meta-filter
        meta_features = np.array([[
            pred.prob_up,                           # 1  ensemble_up_prob
            pred.prob_down,                         # 2  ensemble_down_prob
            pred.prob_neutral,                      # 3  ensemble_neutral_prob
            pred.confidence,                        # 4  ensemble_confidence
            {"UP": 1, "DOWN": -1, "NEUTRAL": 0}.get(pred.direction, 0),  # 5
            pd.Timestamp.now(tz=IST).hour,          # 6  hour_of_day
            0,                                      # 7  minute_of_day (placeholder)
            pd.Timestamp.now(tz=IST).weekday(),     # 8  day_of_week
            0.5,                                    # 9  recent_win_rate (neutral)
            0,                                      # 10 recent_signal_count
            0,                                      # 11 consecutive_losses
            0,                                      # 12 consecutive_wins
            0.5,                                    # 13 avg_meta_prob_recent
            30,                                     # 14 time_since_last_signal_minutes
            1,                                      # 15 session_phase (midday)
            0.0,                                    # 16 current_drawdown
            0.0,                                    # 17 daily_pnl_points
            0,                                      # 18 open_position_flag
            1.0,                                    # 19 atr_regime
        ]])

        meta_proba = meta_model.predict_proba(meta_features)
        win_prob = float(meta_proba[0][1])
        approved = win_prob >= META_FILTER_PROB_MIN

        print(f"\n  --- {symbol} Meta-Filter ---")
        print(f"  WIN probability: {win_prob:.4f}")
        print(f"  Decision:        {'✅ APPROVED' if approved else '❌ REJECTED'} (threshold {META_FILTER_PROB_MIN})")
        print()

    elapsed = time.perf_counter() - t0
    print("=" * 60)
    print(f"SMOKE TEST COMPLETE — {elapsed:.2f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
