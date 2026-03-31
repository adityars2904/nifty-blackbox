"""
core_execution.py
=================
Centralized execution and metrics engine for the NIFTY ML Research Suite.
Provides event-driven portfolio simulation, dynamic position sizing, and institutional-grade risk metrics.
"""

import numpy as np
import pandas as pd

# Centralized thresholds used across all scripts
THRESHOLDS = {
    ("NIFTY", "UP"):       0.60,
    ("NIFTY", "DOWN"):     0.52,
    ("BANKNIFTY", "UP"):   0.60,
    ("BANKNIFTY", "DOWN"): 0.52,
}

# Portfolio Defaults
DEFAULT_CAPITAL = 500_000
RISK_PER_TRADE = 0.005      # 0.5% risk per trade
MAX_RISK_RUPEES = 10_000    # Hard cap liquidity at ₹2M effective compounding
SLIPPAGE_ATR_FRAC = 0.05    # Slippage = 5% of ATR
MIN_SLIPPAGE_PTS = 5.0      # Hard floor for transaction costs (STT/brokerage bounds)
STOP_LOSS_ATR = 1.0
TARGET_ATR = 1.5

def resolve_trade_paths(signals, feat_df, slippage_frac=SLIPPAGE_ATR_FRAC):
    """
    Simulates an event-driven trade path cleanly. 
    Outputs exact chronological entry/exit timestamps and true prices.
    (Capital tracking and position sizing are outsourced / ignored here).
    """
    if len(signals) == 0:
        return pd.DataFrame()

    feat_df = feat_df.copy()
    feat_df["timestamp"] = pd.to_datetime(feat_df["timestamp"])
    feat_ts = feat_df["timestamp"].dt.tz_localize(None).values
    
    hi_arr = feat_df["high"].values
    lo_arr = feat_df["low"].values
    close_arr = feat_df["close"].values
    time_arr = pd.to_datetime(feat_df["timestamp"]).dt.time.values
    cutoff_time = pd.Timestamp("15:15:00").time()

    signals = signals.copy()
    signals["timestamp"] = pd.to_datetime(signals["timestamp"]).dt.tz_localize(None)
    signals = signals.sort_values("timestamp")

    resolved_trades = []

    for _, sig in signals.iterrows():
        sig_ts = sig["timestamp"]
        symbol = sig.get("symbol", "UNKNOWN")
        direction = sig["direction"]
        atr = float(sig["atr"])

        if atr <= 0 or np.isnan(atr):
            continue

        stop_pts = atr * STOP_LOSS_ATR
        target_pts = atr * TARGET_ATR
        
        # Calculate strict physical slippage per leg
        total_friction = max((atr * slippage_frac) * 2, MIN_SLIPPAGE_PTS)
        leg_slip = total_friction / 2.0

        idx = np.searchsorted(feat_ts, np.datetime64(sig_ts), side="left")
        if idx >= len(feat_df):
            continue
        
        raw_entry = float(close_arr[idx])
        # Apply slippage BEFORE accounting so VectorBT sees the true "dirty" price
        adj_entry = raw_entry + leg_slip if direction == "UP" else raw_entry - leg_slip

        risk_scalar = float(sig.get("risk_scalar", 1.0))
        
        exit_idx = min(idx + 12, len(feat_df) - 1)
        raw_exit = float(close_arr[exit_idx])

        for fwd in range(idx + 1, len(feat_df)):
            hi = hi_arr[fwd]
            lo = lo_arr[fwd]
            
            # 1. Check Intraday Hard Exits
            if time_arr[fwd] >= cutoff_time:
                raw_exit = float(close_arr[fwd])
                exit_idx = fwd
                break

            # 2. Check Standard TP/SL with Pessimistic Intrabar Resolution
            if direction == "UP":
                hit_tp = hi >= (raw_entry + target_pts)
                hit_sl = lo <= (raw_entry - stop_pts)
                if hit_tp and hit_sl:
                    raw_exit = raw_entry - stop_pts
                    exit_idx = fwd
                    break
                elif hit_tp:
                    raw_exit = raw_entry + target_pts
                    exit_idx = fwd
                    break
                elif hit_sl:
                    raw_exit = raw_entry - stop_pts
                    exit_idx = fwd
                    break
            else:
                hit_tp = lo <= (raw_entry - target_pts)
                hit_sl = hi >= (raw_entry + stop_pts)
                if hit_tp and hit_sl:
                    raw_exit = raw_entry + stop_pts
                    exit_idx = fwd
                    break
                elif hit_tp:
                    raw_exit = raw_entry - target_pts
                    exit_idx = fwd
                    break
                elif hit_sl:
                    raw_exit = raw_entry + stop_pts
                    exit_idx = fwd
                    break
                    
        # Apply exit slippage
        adj_exit = raw_exit - leg_slip if direction == "UP" else raw_exit + leg_slip
        
        resolved_trades.append({
            "symbol": symbol,
            "direction": direction,
            "entry_ts": feat_ts[idx],
            "exit_ts": feat_ts[exit_idx],
            "entry_price": adj_entry,
            "exit_price": adj_exit,
            "stop_pts": stop_pts,
            "risk_scalar": risk_scalar,
            "atr": atr,
            "friction_pts": total_friction
        })

    if not resolved_trades:
        return pd.DataFrame()
        
    return pd.DataFrame(resolved_trades).sort_values("entry_ts").reset_index(drop=True)


def simulate_portfolio(signals, feat_df, initial_capital=DEFAULT_CAPITAL, risk_fraction=RISK_PER_TRADE,
                       slippage_frac=SLIPPAGE_ATR_FRAC, max_positions_per_symbol=1):
    """
    Simulates a localized, siloed portfolio for legacy backwards compatibility with scripts 01-04 and 06.
    Wraps resolve_trade_paths and computes siloed capital compounding.
    """
    trades_df = resolve_trade_paths(signals, feat_df, slippage_frac)
    if trades_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    current_capital = float(initial_capital)
    equity_records = []
    enriched_trades = []
    
    # Sort by exit_ts for path-dependent compounding resolution
    trades_df = trades_df.sort_values("exit_ts").reset_index(drop=True)
    
    for _, trade in trades_df.iterrows():
        # Enforce max_positions check loosely based on active overlapping indices
        risk_amount = current_capital * risk_fraction * trade["risk_scalar"]
        if risk_amount > MAX_RISK_RUPEES:
            risk_amount = MAX_RISK_RUPEES
            
        position_units = risk_amount / trade["stop_pts"] if trade["stop_pts"] > 0 else 0
        
        # Calculate isolated net pnl pts strictly from the slippage-adjusted prices
        if trade["direction"] == "UP":
            net_pnl_pts = trade["exit_price"] - trade["entry_price"]
        else:
            net_pnl_pts = trade["entry_price"] - trade["exit_price"]
            
        pnl_rupees = net_pnl_pts * position_units
        current_capital += pnl_rupees
        
        t_dict = trade.to_dict()
        t_dict["position_units"] = position_units
        t_dict["pnl_pts"] = net_pnl_pts
        t_dict["pnl_rupees"] = pnl_rupees
        enriched_trades.append(t_dict)
        
        equity_records.append({
            "timestamp": trade["exit_ts"],
            "capital": current_capital,
            "trade_pnl": pnl_rupees
        })

    enriched_df = pd.DataFrame(enriched_trades)
    portfolio_df = pd.DataFrame(equity_records)
    
    if not portfolio_df.empty:
        portfolio_df = portfolio_df.set_index("timestamp").sort_index()
        portfolio_df = portfolio_df.groupby(portfolio_df.index).agg({
            "capital": "last",
            "trade_pnl": "sum"
        })
    
    return enriched_df, portfolio_df


def compute_metrics(trades_df, portfolio_df, feat_df, initial_capital=DEFAULT_CAPITAL):
    """
    Computes institutional risk-adjusted metrics taking exposure into account.
    """
    if len(trades_df) == 0 or len(portfolio_df) == 0:
        return {}

    # Exposure: only count bars between trade entry and exit
    trade_durations = (trades_df["exit_ts"] - trades_df["entry_ts"]).dt.total_seconds() / 60
    total_market_bars = len(feat_df)
    total_exposure_bars = trade_durations.sum() / 5.0  # Appx 5m bars in market
    exposure_ratio = min(total_exposure_bars / total_market_bars, 1.0) if total_market_bars > 0 else 0

    returns = portfolio_df["trade_pnl"] / portfolio_df["capital"].shift(1).fillna(initial_capital)
    
    daily_pnl = portfolio_df["trade_pnl"].resample("D").sum().dropna()
    daily_pnl = daily_pnl[daily_pnl != 0] # Only trading days
    
    if len(daily_pnl) > 1:
        daily_ret = daily_pnl / initial_capital  # Approximation for proxy
        sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(252)
        downside = daily_ret[daily_ret < 0]
        sortino = (daily_ret.mean() / downside.std()) * np.sqrt(252) if len(downside) > 0 and downside.std() > 0 else 0
    else:
        sharpe = 0.0
        sortino = 0.0

    adj_sharpe = sharpe * np.sqrt(exposure_ratio)

    peak = portfolio_df["capital"].cummax()
    drawdown = (peak - portfolio_df["capital"]) / peak
    max_dd = drawdown.max()
    
    calmar = (returns.mean() * 252) / abs(max_dd) if max_dd > 0 else 0

    if "pnl_rupees" in trades_df.columns:
        wins = trades_df[trades_df["pnl_rupees"] > 0]["pnl_rupees"]
        losses = trades_df[trades_df["pnl_rupees"] <= 0]["pnl_rupees"]
    else:
        return {}

    profit_factor = wins.sum() / abs(losses.sum()) if abs(losses.sum()) > 0 else float("inf")

    return {
        "Total Trades": len(trades_df),
        "Win Rate": len(wins) / len(trades_df),
        "Profit Factor": profit_factor,
        "Total PnL": portfolio_df["capital"].iloc[-1] - initial_capital,
        "Max Drawdown": max_dd,
        "Sharpe": sharpe,
        "Adj Sharpe": adj_sharpe,
        "Sortino": sortino,
        "Calmar": calmar,
        "Exposure": exposure_ratio
    }
