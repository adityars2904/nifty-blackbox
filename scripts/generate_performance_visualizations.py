#!/usr/bin/env python3
"""
generate_performance_visualizations.py
=============================================================================
Comprehensive visualization suite for the NIFTY/BANKNIFTY ML Trading System.

UPDATES:
- Extracts REAL performance data by running models on Validation/Vault data.
- Generates Equity Curve from actual backtest results.
- Supports dynamic Symbol selection.
- ALL 10 CHARTS included.

Usage:
    python scripts/generate_performance_visualizations.py --symbol NIFTY
    python scripts/generate_performance_visualizations.py --symbol BANKNIFTY

Output:
    All charts saved to outputs/visualizations/ with symbol prefix.
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch, Rectangle
from sklearn.metrics import confusion_matrix

from backend.ml.data_loader import load_training_data, load_validation_data, fetch_candles
from backend.ml.feature_engineering import add_features, MOVEMENT_FEATURES
from backend.ml.labels import label_movement
from backend.ml.ensemble_predictor import EnsemblePredictor

# ============================================================================
# CONFIGURATION
# ============================================================================

OUTPUT_DIR = Path("outputs/visualizations")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_DIR = "backend/models"

# Styling
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

COLORS = {
    'train': '#3498db',
    'validation': '#e74c3c',
    'vault_2025': '#2ecc71',
    'up': '#27ae60',
    'down': '#e74c3c',
    'neutral': '#95a5a6',
    'approved': '#27ae60',
    'rejected': '#e74c3c',
}

META_FEATURES = [
    "ens_confidence", "conf_5m", "conf_15m", "prob_gap", "p_neutral",
    "vol_ratio", "vol_expansion", "regime_trend", "z_score_20",
    "close_position", "z_score_distance_from_vwap", "time_sin",
    "regime_time_of_day", "recent_win_rate_5", "recent_win_rate_10",
    "consecutive_losses", "consecutive_wins", "signals_today",
    "bars_since_last_signal",
]

# ============================================================================
# DATA PREPARATION HELPERS
# ============================================================================

def label_signal_outcomes(signals: pd.DataFrame, feat_5m: pd.DataFrame) -> pd.Series:
    """Replicate training logic to determine WIN/LOSS for backtesting."""
    feat_5m = feat_5m.copy().reset_index(drop=True)
    feat_ts = np.asarray(pd.to_datetime(feat_5m["timestamp"]).values, dtype='datetime64[ns]')
    outcomes = []
    
    ATR_STOP = 1.0
    ATR_TARGET = 1.5
    FWD_CANDLES = 12

    for _, row in signals.iterrows():
        sig_ts = pd.to_datetime(row["timestamp"])
        direction = row["direction"]
        atr = float(row["atr"])
        
        idx = int(np.searchsorted(feat_ts, np.datetime64(sig_ts), side="left"))
        if idx >= len(feat_5m) - 1 or atr <= 0 or np.isnan(atr):
            outcomes.append(0)
            continue
            
        entry = float(feat_5m.iloc[idx]["close"])
        stop_pts = atr * ATR_STOP
        target_pts = atr * ATR_TARGET
        result = 0
        
        for fwd in range(idx + 1, min(idx + 1 + FWD_CANDLES, len(feat_5m))):
            h, l = float(feat_5m.iloc[fwd]["high"]), float(feat_5m.iloc[fwd]["low"])
            if direction == "UP":
                if h >= entry + target_pts: result = 1; break
                if l <= entry - stop_pts: result = 0; break
            else:
                if l <= entry - target_pts: result = 1; break
                if h >= entry + stop_pts: result = 0; break
        outcomes.append(result)
    return pd.Series(outcomes, index=signals.index)

def build_meta_features(signals: pd.DataFrame, feat_5m: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct meta-features for the visualization set."""
    df = signals.copy().reset_index(drop=True)
    probs = df[["prob_down", "prob_neutral", "prob_up"]].values
    sorted_p = np.sort(probs, axis=1)
    df["prob_gap"] = sorted_p[:, 2] - sorted_p[:, 1]
    df["p_neutral"] = df["prob_neutral"]
    
    wins = df["win"].values
    rwr5, rwr10, closs, cwins = [], [], [], []
    
    for i in range(len(df)):
        start5 = max(0, i-5); start10 = max(0, i-10)
        rwr5.append(float(np.mean(np.asarray(wins[start5:i], dtype=float))) if i > 0 else 0.5)
        rwr10.append(float(np.mean(np.asarray(wins[start10:i], dtype=float))) if i > 0 else 0.5)
        
        streak = 0
        if i > 0:
            last_res = wins[i-1]
            for j in range(i-1, -1, -1):
                if wins[j] == last_res: streak += 1
                else: break
            if last_res == 0: closs.append(streak); cwins.append(0)
            else: closs.append(0); cwins.append(streak)
        else:
            closs.append(0); cwins.append(0)
            
    df["recent_win_rate_5"] = rwr5
    df["recent_win_rate_10"] = rwr10
    df["consecutive_losses"] = closs
    df["consecutive_wins"] = cwins
    
    ts = pd.to_datetime(df["timestamp"])
    df["signals_today"] = df.groupby(ts.dt.date).cumcount()
    
    feat_ts = np.asarray(pd.to_datetime(feat_5m["timestamp"]).values, dtype='datetime64[ns]')
    sig_ts = np.asarray(ts.values, dtype='datetime64[ns]')
    idxs = np.searchsorted(feat_ts, sig_ts)
    df["bars_since_last_signal"] = np.concatenate(([0], np.diff(idxs))).astype(float)
    
    return df

def generate_full_results(symbol: str, data_type="validation"):
    """Run full pipeline (Ensemble + Meta) on data to get raw signals."""
    print(f"  ⚡ Running inference on {data_type} data...")
    
    if data_type == "validation":
        df_5m = load_validation_data(symbol, "5m")
        df_15m = load_validation_data(symbol, "15m")
    else: # vault
        df_5m = fetch_candles(symbol, "5m", "2025-01-01 09:15:00", "2025-02-28 15:30:00")
        df_15m = fetch_candles(symbol, "15m", "2025-01-01 09:15:00", "2025-02-28 15:30:00")
    
    if len(df_5m) == 0: return pd.DataFrame(), pd.DataFrame()

    f5 = add_features(df_5m); f5['label'] = label_movement(f5)
    f5 = f5.dropna(subset=MOVEMENT_FEATURES + ['label', 'atr']).reset_index(drop=True)
    f15 = add_features(df_15m).dropna(subset=MOVEMENT_FEATURES).reset_index(drop=True)

    ensemble = EnsemblePredictor.load(MODEL_DIR, symbol)
    meta_path = Path(f"{MODEL_DIR}/meta_filter_ensemble_{symbol.lower()}.joblib")
    if not meta_path.exists(): return pd.DataFrame(), pd.DataFrame()
    
    meta_data = joblib.load(meta_path)
    meta_filter = meta_data['model'] if isinstance(meta_data, dict) else meta_data

    ens_feats_5 = ensemble.model_5m.get_booster().feature_names
    ens_feats_15 = ensemble.model_15m.get_booster().feature_names
    
    p5 = ensemble.model_5m.predict_proba(f5[ens_feats_5])
    p15 = ensemble.model_15m.predict_proba(f15[ens_feats_15])
    
    f5_c = f5.copy(); f15_c = f15.copy()
    f5_c[['p_d_5','p_n_5','p_u_5']] = p5
    f15_c[['p_d_15','p_n_15','p_u_15']] = p15
    
    merged = pd.merge_asof(f5_c.sort_values('timestamp'), 
                           f15_c[['timestamp','p_d_15','p_n_15','p_u_15']], 
                           on='timestamp', direction='backward').dropna()
    
    w5, w15 = 0.3, 0.7
    merged['prob_down'] = merged['p_d_5']*w5 + merged['p_d_15']*w15
    merged['prob_neutral'] = merged['p_n_5']*w5 + merged['p_n_15']*w15
    merged['prob_up'] = merged['p_u_5']*w5 + merged['p_u_15']*w15
    
    merged['ens_conf'] = merged[['prob_down','prob_neutral','prob_up']].max(axis=1)
    merged['ens_dir'] = merged[['prob_down','prob_neutral','prob_up']].idxmax(axis=1).map(
        {'prob_down':'DOWN', 'prob_neutral':'NEUTRAL', 'prob_up':'UP'}
    )
    
    merged['conf_5m'] = p5[merged.index].max(axis=1)
    merged['conf_15m'] = merged[['p_d_15','p_n_15','p_u_15']].max(axis=1)
    
    mask = (merged['ens_dir'] != 'NEUTRAL') & (merged['ens_conf'] >= ensemble.min_confidence)
    signals = merged[mask].copy().rename(columns={'ens_dir':'direction', 'ens_conf':'ens_confidence'})
    
    signals['win'] = label_signal_outcomes(signals, f5)
    signals = build_meta_features(signals, f5)
    
    avail_feats = [f for f in META_FEATURES if f in signals.columns]
    signals['meta_prob'] = meta_filter.predict_proba(signals[avail_feats])[:, 1]
    signals['approved'] = signals['meta_prob'] >= 0.55
    
    return signals, f5

def load_raw_regime_data(symbol):
    """Load raw feature data for Chart 6."""
    t = load_training_data(symbol, "5m")
    v = load_validation_data(symbol, "5m")
    va = fetch_candles(symbol, "5m", "2025-01-01 09:15:00", "2025-02-28 15:30:00")
    
    def process(df, with_label=False):
        if len(df) == 0: return pd.DataFrame()
        f = add_features(df)
        if with_label:
            f['label'] = label_movement(f)
        return f.dropna(subset=MOVEMENT_FEATURES + (['label'] if with_label else []))
        
    return process(t, with_label=True), process(v, with_label=True), process(va, with_label=False)

# ============================================================================
# CHART GENERATION
# ============================================================================

def save_fig(fig, name, symbol):
    path = OUTPUT_DIR / f"{symbol}_{name}.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  ✅ Saved: {path}")
    plt.close(fig)

# --- Chart 1 ---
def chart_model_performance(symbol, val_signals, vault_signals):
    print("\n📊 Chart 1: Model Performance Comparison")
    
    def get_metrics(df):
        if len(df) == 0: return 0, 0, 0, 0
        base_prec = df['win'].mean() * 100
        meta_prec = df[df['approved']]['win'].mean() * 100 if df['approved'].sum() > 0 else 0
        sig_rate = len(df) / 120 
        meta_sig_rate = len(df[df['approved']]) / 120
        return base_prec, meta_prec, sig_rate, meta_sig_rate

    v_base, v_meta, v_rate, vm_rate = get_metrics(val_signals)
    va_base, va_meta, va_rate, vam_rate = get_metrics(vault_signals)
    
    metrics = {
        'Ensemble\n(Validation)': {'Precision': v_base, 'Signals/Day': v_rate},
        'Meta-Filter\n(Validation)': {'Precision': v_meta, 'Signals/Day': vm_rate},
        'Ensemble\n(2025 Vault)': {'Precision': va_base, 'Signals/Day': va_rate},
        'Meta-Filter\n(2025 Vault)': {'Precision': va_meta, 'Signals/Day': vam_rate},
    }
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(f'{symbol} Real Performance (Calculated from Data)', fontsize=16, fontweight='bold')
    
    names = list(metrics.keys())
    x = np.arange(len(names))
    
    ax = axes[0]
    precs = [metrics[n]['Precision'] for n in names]
    colors = [COLORS['validation'], COLORS['approved'], COLORS['validation'], COLORS['vault_2025']]
    bars = ax.bar(x, precs, color=colors)
    ax.axhline(50, color='red', linestyle='--')
    ax.set_title('Precision (Win Rate)')
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=15)
    for bar, v in zip(bars, precs): ax.text(bar.get_x()+bar.get_width()/2, v, f"{v:.1f}%", ha='center', va='bottom', fontweight='bold')
        
    ax = axes[1]
    rates = [metrics[n]['Signals/Day'] for n in names]
    bars = ax.bar(x, rates, color=colors)
    ax.set_title('Signals Per Day')
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=15)
    for bar, v in zip(bars, rates): ax.text(bar.get_x()+bar.get_width()/2, v, f"{v:.1f}", ha='center', va='bottom', fontweight='bold')
        
    save_fig(fig, '01_model_performance', symbol)

# --- Chart 2 ---
def chart_feature_imp(symbol):
    print("\n📊 Chart 2: Feature Importance")
    path = f"{MODEL_DIR}/meta_filter_ensemble_{symbol.lower()}.joblib"
    if not Path(path).exists(): return
    
    data = joblib.load(path)
    model = data['model'] if isinstance(data, dict) else data
    imps = model.feature_importances_
    names = np.array(model.get_booster().feature_names)
    idx = np.argsort(imps)[::-1][:15]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(x=imps[idx], y=names[idx], palette='viridis', ax=ax)
    ax.set_title(f'{symbol} Top 15 Meta-Features')
    save_fig(fig, '02_feature_importance', symbol)

# --- Chart 3 ---
def chart_feature_distributions(symbol, f_train, f_val, f_vault):
    print("\n📊 Chart 3: Feature Distributions")
    if len(f_val) == 0 or 'label' not in f_val.columns: return
    
    key_features = ['z_score_20', 'vol_ratio', 'vol_expansion', 'regime_trend']
    f_val['Movement'] = f_val['label'].map({0: 'DOWN', 1: 'NEUTRAL', 2: 'UP'})
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{symbol} Feature Distributions by Movement (Validation)', fontsize=14, fontweight='bold')
    
    for idx, feature in enumerate(key_features):
        ax = axes[idx // 2, idx % 2]
        data = [f_val[f_val['label']==l][feature].dropna() for l in [2, 1, 0]]
        parts = ax.violinplot(data, showmeans=True)
        for i, pc in enumerate(parts['bodies']): pc.set_facecolor([COLORS['up'], COLORS['neutral'], COLORS['down']][i])
        ax.set_xticks([1, 2, 3]); ax.set_xticklabels(['UP', 'NEUTRAL', 'DOWN'])
        ax.set_title(feature)
        
    save_fig(fig, '03_feature_distributions', symbol)

# --- Chart 4 ---
def chart_confusion_matrices(symbol, val_signals, val_features):
    print("\n📊 Chart 4: Confusion Matrix")
    # Note: val_signals only has binary ensemble output (UP/DOWN). 
    # For full 3-class CM, we need the raw ensemble predictions vs true labels.
    if len(val_features) == 0: return
    
    ensemble = EnsemblePredictor.load(MODEL_DIR, symbol)
    feats = ensemble.model_5m.get_booster().feature_names
    p5 = ensemble.model_5m.predict(val_features[feats])
    y_true = val_features['label'].astype(int)
    
    # Align lengths
    min_len = min(len(p5), len(y_true))
    cm = confusion_matrix(y_true[:min_len], p5[:min_len])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=['D','N','U'], yticklabels=['D','N','U'])
    ax.set_title(f'{symbol} Ensemble Confusion Matrix (5m)')
    save_fig(fig, '04_confusion_matrices', symbol)

# --- Chart 5 ---
def chart_equity_curve(symbol, vault_signals):
    print("\n📊 Chart 5: Equity Curve (Backtest)")
    if len(vault_signals) == 0: return

    vault_signals = vault_signals.sort_values('timestamp')
    capital, risk, RR = 500000.0, 2500.0, 1.5
    
    base_eq: list[float] = [float(capital)]
    for win in vault_signals['win']:
        base_eq.append(base_eq[-1] + (risk * RR if win else -risk))
        
    meta_eq: list[float] = [float(capital)]
    meta_sigs = vault_signals[vault_signals['approved']]
    for win in meta_sigs['win']:
        meta_eq.append(meta_eq[-1] + (risk * RR if win else -risk))
        
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))
    ax = axes[0]
    ax.plot(base_eq, label=f'Base ({len(vault_signals)} trades)', color=COLORS['validation'])
    ax.plot(np.linspace(0, len(base_eq), len(meta_eq)), meta_eq, label=f'Meta ({len(meta_sigs)} trades)', color=COLORS['approved'], linewidth=2)
    ax.set_title(f'{symbol} 2025 Equity Curve'); ax.legend(); ax.grid(True, alpha=0.3)
    
    ax = axes[1] # Drawdown
    def dd(eq): return (eq - np.maximum.accumulate(eq)) / np.maximum.accumulate(eq) * 100
    ax.plot(dd(np.array(base_eq)), color=COLORS['validation'], label='Base DD')
    ax.plot(np.linspace(0, len(base_eq), len(meta_eq)), dd(np.array(meta_eq)), color=COLORS['approved'], label='Meta DD')
    ax.set_title('Drawdown (%)'); ax.legend(); ax.grid(True, alpha=0.3)
    
    save_fig(fig, '05_equity_curve', symbol)

# --- Chart 6 ---
def chart_regime_stability(symbol, f_train, f_val, f_vault):
    print("\n📊 Chart 6: Regime Stability")
    features = ['vol_ratio', 'z_score_20', 'regime_trend', 'vol_expansion']
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{symbol} Feature Stability (Train vs Val vs 2025)', fontsize=14)
    
    for idx, f in enumerate(features):
        ax = axes[idx // 2, idx % 2]
        data = [df[f].dropna() for df in [f_train, f_val, f_vault] if len(df) > 0]
        if not data: continue
        bp = ax.boxplot(data, tick_labels=['Train', 'Val', '2025'], patch_artist=True, showmeans=True)
        for patch, color in zip(bp['boxes'], [COLORS['train'], COLORS['validation'], COLORS['vault_2025']]):
            patch.set_facecolor(color); patch.set_alpha(0.6)
        ax.set_title(f)
    save_fig(fig, '06_regime_stability', symbol)

# --- Chart 7 ---
def chart_signal_quality(symbol, vault_signals):
    print("\n📊 Chart 7: Signal Quality")
    if len(vault_signals) == 0: return
    
    vault_signals['date'] = pd.to_datetime(vault_signals['timestamp']).dt.date
    daily = vault_signals.groupby('date').agg({'win': ['mean', 'count']})
    daily.columns = ['win_rate', 'count']
    
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.bar(daily.index, daily['win_rate']*100, color=COLORS['approved'], alpha=0.6, label='Daily Win Rate')
    ax.axhline(50, color='red', linestyle='--')
    ax.set_title(f'{symbol} Daily Win Rate (2025 Vault)'); ax.legend()
    save_fig(fig, '07_signal_quality_over_time', symbol)

# --- Chart 8 ---
def chart_decision_boundary(symbol, vault_signals):
    print("\n📊 Chart 8: Decision Boundary")
    if len(vault_signals) == 0: return
    
    fig, ax = plt.subplots(figsize=(14, 10))
    # Use real meta features
    x = vault_signals['consecutive_losses'] + np.random.normal(0, 0.1, len(vault_signals)) # Jitter
    y = vault_signals['recent_win_rate_5']
    c = vault_signals['approved'].map({True: 'green', False: 'red'})
    
    ax.scatter(x, y, c=c, alpha=0.5)
    ax.set_xlabel('Consecutive Losses'); ax.set_ylabel('Recent Win Rate (5)')
    ax.set_title(f'{symbol} Meta-Filter Decisions (Green=Approved)')
    save_fig(fig, '08_meta_filter_decision_boundary', symbol)

# --- Chart 9 ---
def chart_directional(symbol, vault_signals):
    print("\n📊 Chart 9: Directional Performance")
    if len(vault_signals) == 0: return
    
    approved = vault_signals[vault_signals['approved']]
    if len(approved) == 0: return
    
    up = approved[approved['direction']=='UP']
    dn = approved[approved['direction']=='DOWN']
    
    up_win = up['win'].mean()*100 if len(up)>0 else 0
    dn_win = dn['win'].mean()*100 if len(dn)>0 else 0
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(['UP', 'DOWN'], [up_win, dn_win], color=[COLORS['up'], COLORS['down']])
    ax.axhline(50, color='red', linestyle='--')
    ax.set_title(f'{symbol} Meta-Filter Win Rate by Direction (2025)'); ax.set_ylabel('Win Rate %')
    for bar, v, c in zip(bars, [up_win, dn_win], [len(up), len(dn)]):
        ax.text(bar.get_x()+bar.get_width()/2, v, f"{v:.1f}%\n({c})", ha='center', va='bottom')
    save_fig(fig, '09_directional_performance', symbol)

# --- Chart 10 ---
def chart_summary(symbol, val_signals, vault_signals):
    print("\n📊 Chart 10: Summary Dashboard")
    meta_val = val_signals[val_signals['approved']]['win'].mean() if len(val_signals)>0 else 0
    meta_vault = vault_signals[vault_signals['approved']]['win'].mean() if len(vault_signals)>0 else 0
    
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f'{symbol} System Performance Summary', fontsize=22, fontweight='bold')
    
    kpis = [
        ("Validation Precision", f"{meta_val:.1%}", COLORS['train']),
        ("2025 Vault Precision", f"{meta_vault:.1%}", COLORS['vault_2025']),
        ("Regime Gap", f"{meta_val - meta_vault:.1%}", COLORS['rejected'] if abs(meta_val-meta_vault)>0.1 else COLORS['approved']),
        ("Approved Trades (2025)", f"{len(vault_signals[vault_signals['approved']])}", COLORS['neutral'])
    ]
    
    for i, (label, val, col) in enumerate(kpis):
        ax = fig.add_subplot(2, 2, i+1)
        ax.text(0.5, 0.6, val, ha='center', fontsize=40, fontweight='bold', color=col)
        ax.text(0.5, 0.2, label, ha='center', fontsize=14, color='gray')
        ax.axis('off')
    save_fig(fig, '10_summary_dashboard', symbol)

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate Visualizations")
    parser.add_argument("--symbol", type=str, default="NIFTY", choices=["NIFTY", "BANKNIFTY"])
    args = parser.parse_args()
    
    print("=" * 70)
    print(f"📊 GENERATING LIVE VISUALIZATIONS FOR {args.symbol}")
    print("=" * 70)
    
    # 1. Generate Real Data
    val_signals, val_feats = generate_full_results(args.symbol, "validation")
    vault_signals, vault_feats = generate_full_results(args.symbol, "vault")
    
    # Load raw feature data for stability chart
    f_train, f_val, f_vault = load_raw_regime_data(args.symbol)
    
    if len(val_signals) == 0:
        print("❌ Critical Error: No signals generated. Check data/models.")
        return

    # 2. Generate Charts with Real Data
    chart_model_performance(args.symbol, val_signals, vault_signals)
    chart_feature_imp(args.symbol)
    chart_feature_distributions(args.symbol, f_train, f_val, f_vault)
    chart_confusion_matrices(args.symbol, val_signals, val_feats)
    chart_equity_curve(args.symbol, vault_signals)
    chart_regime_stability(args.symbol, f_train, f_val, f_vault)
    chart_signal_quality(args.symbol, vault_signals)
    chart_decision_boundary(args.symbol, vault_signals)
    chart_directional(args.symbol, vault_signals)
    chart_summary(args.symbol, val_signals, vault_signals)
    
    print(f"\n✅ All 10 charts saved to {OUTPUT_DIR} with prefix {args.symbol}_")

if __name__ == "__main__":
    main()