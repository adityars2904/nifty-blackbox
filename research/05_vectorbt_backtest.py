#!/usr/bin/env python3
"""
05_vectorbt_backtest.py
========================
Institutional-Grade Multi-Asset Portfolio Backtest (VectorBT Hybrid).

Architecture:
  1. resolve_trade_paths() → finds exact entry/exit timestamps + slippage-adjusted prices
  2. Converts trades into VectorBT order arrays (entries, exits, sizes as % of portfolio)
  3. vbt.Portfolio.from_orders() handles shared capital pooling, margin contention, and compounding

Usage:
    cd research && python 05_vectorbt_backtest.py
"""

from __future__ import annotations
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import vectorbt as vbt
import warnings
warnings.filterwarnings('ignore')

from backend.ml.data_loader import fetch_candles
from backend.ml.feature_engineering import add_features, MOVEMENT_FEATURES
from backend.ml.labels import label_movement
from backend.ml.ensemble_predictor import EnsemblePredictor
from scripts.validate_meta_filter_2025 import generate_ensemble_signals, label_signal_outcomes, build_meta_features
from research.core_execution import resolve_trade_paths, THRESHOLDS, RISK_PER_TRADE, STOP_LOSS_ATR, DEFAULT_CAPITAL, MAX_RISK_RUPEES

MODEL_DIR = str(project_root / "backend" / "models")
OUTPUT_DIR = Path(__file__).parent / "outputs"

VAULT_START = "2025-01-01 09:15:00"
VAULT_END   = "2025-12-10 15:30:00"


def get_approved_signals(symbol, feat_5m, feat_15m):
    """Generates and filters signals through meta-model for a single symbol."""
    ensemble = EnsemblePredictor.load(MODEL_DIR, symbol=symbol)
    signals = generate_ensemble_signals(feat_5m, feat_15m, ensemble)

    all_approved = []
    for direction in ["UP", "DOWN"]:
        dir_signals = signals[signals["direction"] == direction].reset_index(drop=True)
        if len(dir_signals) == 0:
            continue

        dir_signals["win"] = label_signal_outcomes(dir_signals, feat_5m)
        dir_signals = build_meta_features(dir_signals, feat_5m, symbol)

        threshold = THRESHOLDS.get((symbol, direction), 0.55)
        meta_path = project_root / "backend" / "models" / f"meta_filter_{symbol.lower()}_{direction.lower()}.joblib"
        if not meta_path.exists():
            meta_path = project_root / "backend" / "models" / f"meta_filter_ensemble_{symbol.lower()}.joblib"

        loaded = joblib.load(meta_path)
        meta_model = loaded["model"] if isinstance(loaded, dict) and "model" in loaded else loaded
        feature_names = list(meta_model.get_booster().feature_names)

        if len(dir_signals) > 0:
            X = dir_signals[feature_names].astype(float)
            probs = meta_model.predict_proba(X)[:, 1]
            approved = dir_signals[probs >= threshold].copy()
            if len(approved) > 0:
                all_approved.append(approved)

    if not all_approved:
        return pd.DataFrame()
    return pd.concat(all_approved).sort_values("timestamp").reset_index(drop=True)


def main():
    print("=" * 80)
    print("INSTITUTIONAL BACKTEST — VectorBT Multi-Asset Portfolio")
    print(f"Period: {VAULT_START} to {VAULT_END}")
    print(f"Shared Capital Pool: ₹{DEFAULT_CAPITAL:,.0f}")
    print("=" * 80)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    symbols = ["NIFTY", "BANKNIFTY"]
    all_resolved_trades = []
    feat_data = {}
    
    # ── STEP 1: Resolve trade paths for each symbol ──
    for symbol in symbols:
        print(f"\nResolving {symbol} trade paths...")
        raw_5m = fetch_candles(symbol, timeframe="5m", start_date=VAULT_START, end_date=VAULT_END)
        raw_15m = fetch_candles(symbol, timeframe="15m", start_date=VAULT_START, end_date=VAULT_END)

        if len(raw_5m) == 0:
            print(f"  ❌ No data for {symbol}")
            continue

        feat_5m = add_features(raw_5m).copy()
        feat_5m["label"] = label_movement(feat_5m)
        feat_5m = feat_5m.dropna(subset=MOVEMENT_FEATURES + ["label", "atr"]).reset_index(drop=True)
        feat_15m = add_features(raw_15m).dropna(subset=MOVEMENT_FEATURES).reset_index(drop=True)
        feat_data[symbol] = feat_5m

        approved = get_approved_signals(symbol, feat_5m, feat_15m)
        if approved.empty:
            print(f"  ❌ No approved signals for {symbol}")
            continue

        trades = resolve_trade_paths(approved, feat_5m, slippage_frac=0.05)
        if not trades.empty:
            all_resolved_trades.append(trades)
            print(f"  ✅ {len(trades)} trades resolved ({symbol})")

    if not all_resolved_trades:
        print("\n❌ Backtest yielded no trades.")
        return

    # ── STEP 2: Merge into unified trade timeline ──
    all_trades = pd.concat(all_resolved_trades).sort_values("entry_ts").reset_index(drop=True)
    all_trades["entry_ts"] = pd.to_datetime(all_trades["entry_ts"])
    all_trades["exit_ts"] = pd.to_datetime(all_trades["exit_ts"])
    
    print(f"\n📊 Total resolved trades across all symbols: {len(all_trades)}")

    # ── STEP 3: Build VectorBT order arrays ──
    # We need a unified 5m price index across both symbols
    # Build multi-column price DataFrame aligned to a common 5m grid
    all_timestamps = set()
    price_series = {}
    
    for symbol in symbols:
        if symbol in feat_data:
            df = feat_data[symbol]
            ts = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
            prices = df["close"].values
            s = pd.Series(prices, index=ts, name=symbol)
            s = s[~s.index.duplicated(keep='first')]
            price_series[symbol] = s
            all_timestamps.update(ts.tolist())
    
    common_idx = pd.DatetimeIndex(sorted(all_timestamps))
    price_df = pd.DataFrame(index=common_idx)
    for symbol in symbols:
        if symbol in price_series:
            price_df[symbol] = price_series[symbol].reindex(common_idx).ffill()
    price_df = price_df.dropna()

    # ── STEP 4: Shared-Capital Portfolio Simulation ──
    # This is the critical fix: both NIFTY and BANKNIFTY compete for the SAME ₹500k pool.
    # When capital is locked in an open position, it is unavailable for new trades.
    
    print("\n🏦 Running shared-capital multi-asset portfolio simulation...")
    
    # Build event timeline: every entry and exit is an event
    events = []
    for trade_id, trade in all_trades.iterrows():
        events.append({"time": trade["entry_ts"], "type": "ENTRY", "trade_id": trade_id})
        events.append({"time": trade["exit_ts"],  "type": "EXIT",  "trade_id": trade_id})
    events = sorted(events, key=lambda x: (x["time"], 0 if x["type"] == "EXIT" else 1))
    
    current_capital = float(DEFAULT_CAPITAL)
    open_positions = {}   # trade_id -> {units, locked_capital, direction, symbol}
    equity_log = []
    trade_results = []
    
    for evt in events:
        tid = evt["trade_id"]
        trade = all_trades.iloc[tid]
        
        if evt["type"] == "ENTRY":
            # Size from AVAILABLE capital (not total capital)
            available = current_capital - sum(p["locked_capital"] for p in open_positions.values())
            if available <= 0:
                continue  # No capital available — skip this trade (margin rejection)
            
            risk_rupees = available * RISK_PER_TRADE * trade["risk_scalar"]
            if risk_rupees > MAX_RISK_RUPEES:
                risk_rupees = MAX_RISK_RUPEES
            
            stop_pts = trade["stop_pts"]
            if stop_pts <= 0:
                continue
                
            units = risk_rupees / stop_pts
            locked = units * trade["entry_price"]
            
            # Cannot lock more capital than available
            if locked > available:
                units = available / trade["entry_price"]
                locked = available
            
            open_positions[tid] = {
                "units": units,
                "locked_capital": locked,
                "direction": trade["direction"],
                "symbol": trade["symbol"],
                "entry_price": trade["entry_price"]
            }
            
        elif evt["type"] == "EXIT":
            if tid not in open_positions:
                continue  # Was skipped at entry due to capital exhaustion
                
            pos = open_positions.pop(tid)
            
            if pos["direction"] == "UP":
                pnl_pts = trade["exit_price"] - pos["entry_price"]
            else:
                pnl_pts = pos["entry_price"] - trade["exit_price"]
            
            pnl_rupees = pnl_pts * pos["units"]
            current_capital += pnl_rupees
            
            trade_results.append({
                "symbol": pos["symbol"],
                "direction": pos["direction"],
                "entry_ts": trade["entry_ts"],
                "exit_ts": trade["exit_ts"],
                "entry_price": pos["entry_price"],
                "exit_price": trade["exit_price"],
                "units": pos["units"],
                "pnl_pts": pnl_pts,
                "pnl_rupees": pnl_rupees,
                "capital_after": current_capital
            })
            
            equity_log.append({
                "timestamp": trade["exit_ts"],
                "capital": current_capital,
                "trade_pnl": pnl_rupees,
                "symbol": pos["symbol"]
            })
    
    if not trade_results:
        print("❌ No trades executed (all rejected by capital constraints).")
        return
    
    results_df = pd.DataFrame(trade_results)
    equity_df = pd.DataFrame(equity_log).set_index("timestamp").sort_index()
    
    # ── STEP 5: Compute institutional metrics ──
    print("\n" + "=" * 80)
    print("METRICS SUMMARY (Shared-Capital Multi-Asset Portfolio)")
    print(f"Initial Capital: ₹{DEFAULT_CAPITAL:,.0f}")
    print("=" * 80)
    
    metrics_list = []
    for symbol in symbols:
        sym_trades = results_df[results_df["symbol"] == symbol]
        if len(sym_trades) == 0:
            continue
        
        wins = sym_trades[sym_trades["pnl_rupees"] > 0]
        losses = sym_trades[sym_trades["pnl_rupees"] <= 0]
        win_rate = len(wins) / len(sym_trades)
        pf_ratio = wins["pnl_rupees"].sum() / abs(losses["pnl_rupees"].sum()) if abs(losses["pnl_rupees"].sum()) > 0 else float("inf")
        total_pnl = sym_trades["pnl_rupees"].sum()
        
        # Per-symbol equity for drawdown
        sym_equity = equity_df[equity_df["symbol"] == symbol]["capital"]
        if len(sym_equity) > 0:
            peak = sym_equity.cummax()
            dd = (peak - sym_equity) / peak
            max_dd = dd.max()
        else:
            max_dd = 0.0
        
        metrics_list.append({
            "Symbol": symbol,
            "Trades": len(sym_trades),
            "Win Rate": win_rate,
            "Profit Factor": pf_ratio,
            "Total PnL": total_pnl,
            "Max DD": max_dd,
        })
        
        print(f"\n  {symbol}:")
        print(f"    Trades: {len(sym_trades)} | Win%: {win_rate:.1%} | PF: {pf_ratio:.2f}")
        print(f"    Net PnL: ₹{total_pnl:,.0f} | Max DD: {max_dd:.1%}")
    
    # Global portfolio metrics
    global_equity = equity_df.groupby(equity_df.index).agg({"capital": "last", "trade_pnl": "sum"})
    global_peak = global_equity["capital"].cummax()
    global_dd = (global_peak - global_equity["capital"]) / global_peak
    global_max_dd = global_dd.max()
    
    # Sharpe from daily PnL (Standardized 0-Fill)
    daily_pnl = global_equity["trade_pnl"].resample("D").sum()
    all_bcols = pd.date_range(start=daily_pnl.index.min(), end=daily_pnl.index.max(), freq="B")
    daily_pnl = daily_pnl.reindex(all_bcols).fillna(0)
    
    if len(daily_pnl) > 1:
        daily_ret = daily_pnl / DEFAULT_CAPITAL
        sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(252) if daily_ret.std() > 0 else 0
        downside = daily_ret[daily_ret < 0]
        sortino = (daily_ret.mean() / downside.std()) * np.sqrt(252) if len(downside) > 0 and downside.std() > 0 else 0
    else:
        sharpe = 0.0
        sortino = 0.0
    
    total_return = (current_capital - DEFAULT_CAPITAL) / DEFAULT_CAPITAL
    final_pnl = current_capital - DEFAULT_CAPITAL
    
    print(f"\n  {'─' * 50}")
    print(f"  🏦 COMBINED PORTFOLIO (Shared ₹{DEFAULT_CAPITAL:,.0f} Capital):")
    print(f"    Final Capital: ₹{current_capital:,.0f}")
    print(f"    Net PnL: ₹{final_pnl:,.0f} ({total_return:.1%})")
    print(f"    Total Trades Executed: {len(results_df)}")
    print(f"    Trades Rejected (Capital Exhaustion): {len(all_trades) - len(results_df)}")
    print(f"    Max Drawdown: {global_max_dd:.1%}")
    print(f"    Sharpe: {sharpe:.2f} | Sortino: {sortino:.2f}")
    
    # Save outputs
    if metrics_list:
        import json
        
        # Build JSON dictionary before stringifying the DataFrame
        json_payload = {
            "portfolio": {
                "initial_capital": DEFAULT_CAPITAL,
                "final_capital": current_capital,
                "net_pnl": final_pnl,
                "return_pct": total_return * 100,
                "total_trades": len(results_df),
                "rejected_trades": len(all_trades) - len(results_df),
                "max_dd": global_max_dd * 100,
                "sharpe": sharpe,
                "sortino": sortino
            },
            "assets": metrics_list
        }
        with open(OUTPUT_DIR / "backtest_metrics.json", "w") as f:
            json.dump(json_payload, f, indent=4)
            
        df_metrics = pd.DataFrame(metrics_list)
        for c in ["Win Rate", "Max DD"]:
            df_metrics[c] = df_metrics[c].apply(lambda x: f"{x:.1%}")
        df_metrics["Profit Factor"] = df_metrics["Profit Factor"].apply(lambda x: f"{x:.2f}")
        df_metrics["Total PnL"] = df_metrics["Total PnL"].apply(lambda x: f"₹{x:,.0f}")
        df_metrics.to_csv(OUTPUT_DIR / "backtest_metrics.csv", index=False)
    
    results_df.to_csv(OUTPUT_DIR / "backtest_trades.csv", index=False)
    
    # ── STEP 6: Plot equity curve ──
    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [3, 1]})
    
    ax1 = axes[0]
    ax1.plot(global_equity.index, global_equity["capital"], label="Combined Equity", linewidth=1.5, color="#00ff88")
    
    for symbol, color in zip(symbols, ["#4488ff", "#ff6644"]):
        sym_eq = equity_df[equity_df["symbol"] == symbol]
        if len(sym_eq) > 0:
            cum_pnl = sym_eq["trade_pnl"].cumsum() + DEFAULT_CAPITAL
            ax1.plot(sym_eq.index, cum_pnl.values, label=f"{symbol} (Cumulative)", linewidth=0.8, alpha=0.6, color=color)
    
    ax1.axhline(y=DEFAULT_CAPITAL, color='white', linestyle='--', alpha=0.3, label='Initial Capital')
    ax1.set_title("Shared-Capital Multi-Asset Portfolio (Vault 2025)", fontsize=14, pad=15)
    ax1.set_ylabel("Portfolio Value (₹)", fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.15)
    
    ax2 = axes[1]
    ax2.fill_between(global_dd.index, global_dd.values, alpha=0.4, color="#ff4444", label="Portfolio Drawdown")
    ax2.set_title("Drawdown Profile", fontsize=12)
    ax2.set_ylabel("Drawdown", fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.15)
    
    img_path = OUTPUT_DIR / "equity_curve.png"
    plt.tight_layout()
    plt.savefig(img_path, dpi=300)
    print(f"\n📊 Equity curve saved to {img_path}")
    print(f"💾 Trade log saved to {OUTPUT_DIR / 'backtest_trades.csv'}")


if __name__ == "__main__":
    main()
