import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from backend.ml.data_loader import fetch_candles
from backend.ml.feature_engineering import add_features, MOVEMENT_FEATURES
from backend.ml.labels import label_movement
import importlib
vectorbt_module = importlib.import_module("research.05_vectorbt_backtest")
get_approved_signals = vectorbt_module.get_approved_signals
from research.core_execution import resolve_trade_paths, DEFAULT_CAPITAL

VAULT_START = "2025-01-01 09:15:00"
VAULT_END   = "2025-12-10 15:30:00"

def run_leverage_sweep():
    print("=" * 60)
    print("INSTITUTIONAL LEVERAGE SWEEP ANALYSIS")
    print("=" * 60)
    
    symbols = ["NIFTY", "BANKNIFTY"]
    all_resolved_trades = []
    
    for symbol in symbols:
        print(f"Resolving {symbol} trade paths...")
        raw_5m = fetch_candles(symbol, timeframe="5m", start_date=VAULT_START, end_date=VAULT_END)
        raw_15m = fetch_candles(symbol, timeframe="15m", start_date=VAULT_START, end_date=VAULT_END)
        
        if len(raw_5m) == 0: continue
        feat_5m = add_features(raw_5m).copy()
        feat_5m["label"] = label_movement(feat_5m)
        feat_5m = feat_5m.dropna(subset=MOVEMENT_FEATURES + ["label", "atr"]).reset_index(drop=True)
        feat_15m = add_features(raw_15m).dropna(subset=MOVEMENT_FEATURES).reset_index(drop=True)
        
        approved = get_approved_signals(symbol, feat_5m, feat_15m)
        if approved.empty: continue
        
        trades = resolve_trade_paths(approved, feat_5m, slippage_frac=0.05)
        all_resolved_trades.append(trades)
        
    if not all_resolved_trades:
        print("No trades resolved.")
        return
        
    all_trades = pd.concat(all_resolved_trades).sort_values("entry_ts").reset_index(drop=True)
    
    # Pre-build chronological event list
    events = []
    for trade_id, trade in all_trades.iterrows():
        events.append({"time": trade["entry_ts"], "type": "ENTRY", "trade_id": trade_id})
        events.append({"time": trade["exit_ts"],  "type": "EXIT",  "trade_id": trade_id})
    events = sorted(events, key=lambda x: (x["time"], 0 if x["type"] == "EXIT" else 1))
    
    # Risk tiers to test
    risk_tiers = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.10]
    
    results = []
    
    for risk_pct in risk_tiers:
        # Scale the max risk cap proportionally so it doesn't bottleneck high leverage
        # 0.5% -> 10k cap. So cap multiplier is 4x the risk amount (2.0% of default capital)
        max_risk_cap = DEFAULT_CAPITAL * (risk_pct * 4) 
        
        current_capital = float(DEFAULT_CAPITAL)
        open_positions = {}
        equity_log = []
        
        for evt in events:
            tid = evt["trade_id"]
            trade = all_trades.iloc[tid]
            
            if evt["type"] == "ENTRY":
                available = current_capital - sum(p["locked_capital"] for p in open_positions.values())
                if available <= 0: continue
                
                risk_rupees = available * risk_pct * trade["risk_scalar"]
                if risk_rupees > max_risk_cap:
                    risk_rupees = max_risk_cap
                    
                stop_pts = trade["stop_pts"]
                if stop_pts <= 0: continue
                
                units = risk_rupees / stop_pts
                
                # In Indian derivative markets, we don't block 100% of notional. 
                # We block ~15% Span Margin.
                locked = units * trade["entry_price"] * 0.15
                
                if locked > available:
                    # Max out at available margin
                    units = available / (trade["entry_price"] * 0.15)
                    locked = available
                    
                open_positions[tid] = {
                    "units": units,
                    "locked_capital": locked,
                    "direction": trade["direction"],
                    "symbol": trade["symbol"],
                    "entry_price": trade["entry_price"]
                }
                
            elif evt["type"] == "EXIT":
                if tid not in open_positions: continue
                pos = open_positions.pop(tid)
                
                if pos["direction"] == "UP":
                    pnl_pts = trade["exit_price"] - pos["entry_price"]
                else:
                    pnl_pts = pos["entry_price"] - trade["exit_price"]
                    
                pnl_rupees = pnl_pts * pos["units"]
                current_capital += pnl_rupees
                
                equity_log.append({
                    "timestamp": trade["exit_ts"],
                    "capital": current_capital,
                    "trade_pnl": pnl_rupees
                })
                
        # Calculate metrics for this tier
        if not equity_log: continue
        equity_df = pd.DataFrame(equity_log).set_index("timestamp").sort_index()
        global_equity = equity_df.groupby(equity_df.index).agg({"capital": "last", "trade_pnl": "sum"})
        
        # Max Drawdown
        peak = global_equity["capital"].cummax()
        dd = (peak - global_equity["capital"]) / peak
        max_dd = dd.max()
        
        # Standardized 0-fill Sharpe
        daily_pnl = global_equity["trade_pnl"].resample("D").sum()
        all_bcols = pd.date_range(start=daily_pnl.index.min(), end=daily_pnl.index.max(), freq="B")
        daily_pnl = daily_pnl.reindex(all_bcols).fillna(0)
        
        if len(daily_pnl) > 1:
            daily_ret = daily_pnl / DEFAULT_CAPITAL
            sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(252) if daily_ret.std() > 0 else 0
        else:
            sharpe = 0.0
            
        total_pnl = current_capital - DEFAULT_CAPITAL
        total_ret = total_pnl / DEFAULT_CAPITAL
        
        # Calmar Ratio (Ann Return / Max DD)
        days = len(all_bcols)
        ann_ret = (1 + total_ret) ** (252 / days) - 1
        calmar = ann_ret / max_dd if max_dd > 0 else float('inf')
        
        results.append({
            "Risk Per Trade": f"{risk_pct * 100:.1f}%",
            "Max Drawdown": max_dd,
            "Total Return": total_ret,
            "Ann Return": ann_ret,
            "Sharpe": sharpe,
            "Calmar": calmar,
            "Final Capital": current_capital
        })

    # Print Report
    print("\n" + "-" * 75)
    print(f"{'Risk %':<10} | {'Max DD':<10} | {'Tot Return':<12} | {'Ann Ret':<10} | {'Sharpe':<8} | {'Calmar'}")
    print("-" * 75)
    for r in results:
        print(f"{r['Risk Per Trade']:<10} | {r['Max Drawdown']:<10.1%} | {r['Total Return']:<12.1%} | {r['Ann Return']:<10.1%} | {r['Sharpe']:<8.2f} | {r['Calmar']:.2f}")
    
    print("\nIdeally, an institutional portfolio targets a Calmar Ratio > 3.0 and Max Drawdown between 10-15%.")

if __name__ == "__main__":
    run_leverage_sweep()
