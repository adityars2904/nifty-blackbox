import React, { useState, useEffect } from 'react';
import { fetchBacktestMetrics } from '../api/client';
import { motion, AnimatePresence } from 'framer-motion';

export default function BacktestDashboard() {
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchBacktestMetrics()
      .then((data) => {
        setMetrics(data);
        setLoading(false);
      })
      .catch((err) => {
        setError(err.message || 'Failed to load metrics');
        setLoading(false);
      });
  }, []);

  if (loading) {
    return (
      <div className="flex-1 flex items-center justify-center bg-[#0a0c10]">
        <div className="text-text-secondary text-sm font-display tracking-widest animate-pulse">Gathering Institutional Metrics...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex-1 flex items-center justify-center bg-[#0a0c10]">
        <div className="text-danger flex flex-col items-center gap-3">
          <svg className="w-8 h-8 opacity-80" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
          </svg>
          <span className="font-display text-lg">{error}</span>
        </div>
      </div>
    );
  }

  const { portfolio, assets } = metrics;

  return (
    <div className="flex-1 overflow-y-auto bg-gradient-to-br from-[#0a0c10] to-[#12161f] text-text-primary px-6 py-10 selection:bg-accent/30">
      <motion.div 
        initial={{ opacity: 0, y: 15 }} 
        animate={{ opacity: 1, y: 0 }} 
        transition={{ duration: 0.6 }}
        className="max-w-[1100px] mx-auto space-y-12"
      >
        
        {/* Header Region */}
        <div className="flex flex-col md:flex-row md:items-end justify-between gap-6 border-b border-border/60 pb-8">
          <div className="space-y-2">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-accent/10 border border-accent/20 mb-2">
              <div className="w-2 h-2 rounded-full bg-accent shadow-[0_0_8px_rgba(0,210,106,0.6)] animate-pulse" />
              <span className="text-xs font-mono text-accent uppercase tracking-widest">Live Vault Execution</span>
            </div>
            <h1 className="text-4xl md:text-5xl font-display font-medium tracking-tight text-white drop-shadow-sm">
              Portfolio Returns
            </h1>
            <p className="text-text-secondary font-sans text-sm max-w-xl leading-relaxed">
              Institutional breakdown of the VectorBT combined portfolio simulating real margin constraints and dynamically allocated 5m/15m meta-filtering engines.
            </p>
          </div>
          <div className="text-right">
            <div className="text-xs uppercase tracking-widest text-text-secondary mb-1">Total PnL</div>
            <div className={`text-4xl font-display font-semibold ${portfolio.net_pnl >= 0 ? 'text-accent drop-shadow-[0_0_12px_rgba(0,210,106,0.2)]' : 'text-danger'}`}>
              {portfolio.net_pnl >= 0 ? '+' : ''}₹{portfolio.net_pnl.toLocaleString(undefined, { maximumFractionDigits: 0 })}
            </div>
            <div className="text-sm font-mono text-text-secondary opacity-80 mt-1">({portfolio.return_pct.toFixed(2)}% ROI)</div>
          </div>
        </div>

        {/* Global KPIs */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <InteractiveCard label="Initial Capital" value={`₹${portfolio.initial_capital.toLocaleString(undefined, { maximumFractionDigits: 0 })}`} />
          <InteractiveCard label="Final Capital" value={`₹${portfolio.final_capital.toLocaleString(undefined, { maximumFractionDigits: 0 })}`} />
          
          {/* Custom Sharpe Card */}
          <motion.div whileHover={{ y: -2 }} className="group relative z-10 hover:z-50 bg-[#131822]/80 backdrop-blur-xl border border-border hover:border-border/80 rounded-2xl p-6 shadow-xl hover:shadow-[0_8px_30px_rgba(0,0,0,0.5)] transition-all">
            <div className="flex justify-between items-start mb-2">
              <span className="text-[11px] font-sans uppercase tracking-[0.2em] text-text-secondary font-semibold">Sharpe Ratio</span>
              <div className="relative cursor-help">
                <svg className="w-4 h-4 text-highlight opacity-70 group-hover:opacity-100 transition-opacity" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                {/* Tooltip */}
                <div className="absolute top-full right-0 mt-3 w-64 p-4 bg-[#1e2430] border border-border rounded-xl shadow-2xl opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all z-[100] transform translate-y-2 group-hover:translate-y-0">
                  <p className="text-xs font-sans text-text-primary leading-relaxed relative z-[100]">
                    <span className="text-highlight font-medium">Cash Drag Problem:</span> The engine restricts internal capital compounding. By evaluating fixed geometric constraints, structural volatility is diminished, yielding a natively elevated Sharpe configuration.
                  </p>
                  <div className="absolute -top-2 right-1.5 w-4 h-4 bg-[#1e2430] border-t border-l border-border transform rotate-45 z-[90]"></div>
                </div>
              </div>
            </div>
            <div className="text-3xl font-display font-semibold text-white tracking-tight">{portfolio.sharpe.toFixed(2)}</div>
          </motion.div>

          <InteractiveCard label="Sortino Ratio" value={portfolio.sortino.toFixed(2)} />
          <InteractiveCard label="Total Executions" value={portfolio.total_trades} />
          <InteractiveCard label="Margin Rejections" value={portfolio.rejected_trades} color="text-warning" />
          <InteractiveCard label="Max Drawdown" value={`${portfolio.max_dd.toFixed(1)}%`} color="text-danger" />
          <InteractiveCard label="Market Capacity" value="Shared Pool" />
        </div>

        {/* Asset Cards View */}
        <div>
          <h2 className="text-2xl font-display font-medium text-white mb-6 border-b border-border/40 pb-4">Engine Granularity</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            {assets.map((asset, i) => (
              <motion.div 
                key={i} 
                initial={{ opacity: 0, y: 15 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1 * (i + 1) }}
                className="relative overflow-hidden bg-gradient-to-br from-[#161b22] to-[#12161f] border border-border rounded-3xl p-8 hover:border-border/80 transition-colors"
              >
                {/* Decorative background glow */}
                <div className={`absolute -top-24 -right-24 w-48 h-48 rounded-full blur-[100px] opacity-20 pointer-events-none ${Number(asset["Total PnL"]) < 0 ? 'bg-danger' : 'bg-highlight'}`} />
                
                <div className="flex items-center justify-between mb-8 relative z-10">
                  <div className="space-y-1">
                    <h3 className="text-2xl font-display font-medium text-white tracking-tight">{asset.Symbol}</h3>
                    <div className="text-xs font-mono text-text-secondary">Dual-Horizon XGBoost</div>
                  </div>
                  <div className="px-3 py-1 bg-surface border border-border rounded-full text-xs font-medium text-white">
                    {asset.Trades} Trades
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-y-6 gap-x-12 relative z-10">
                  <AssetMetric 
                    label="Total Profit" 
                    value={`₹${Number(asset["Total PnL"]).toLocaleString(undefined, { maximumFractionDigits: 0 })}`} 
                    highlight 
                    color={Number(asset["Total PnL"]) < 0 ? 'text-danger' : 'text-accent'} 
                  />
                  <AssetMetric label="Win Rate" value={`${(Number(asset["Win Rate"]) * 100).toFixed(1)}%`} />
                  <AssetMetric label="Profit Factor" value={Number(asset["Profit Factor"]).toFixed(2)} />
                  <AssetMetric label="Maximum Drawdown" value={`${(Number(asset["Max DD"]) * 100).toFixed(1)}%`} color="text-danger" />
                </div>
              </motion.div>
            ))}
          </div>
        </div>

      </motion.div>
    </div>
  );
}

// ── Helpers ─────────────────────────────────────────────────────────────

function InteractiveCard({ label, value, color = "text-white" }) {
  return (
    <motion.div whileHover={{ y: -2 }} className="bg-[#131822]/80 backdrop-blur-xl border border-border hover:border-border/80 rounded-2xl p-6 shadow-sm hover:shadow-[0_8px_30px_rgba(0,0,0,0.4)] transition-all">
      <div className="text-[11px] font-sans uppercase tracking-[0.2em] text-text-secondary font-semibold mb-2">{label}</div>
      <div className={`text-3xl font-display font-semibold tracking-tight ${color}`}>{value}</div>
    </motion.div>
  );
}

function AssetMetric({ label, value, highlight, color = "text-white" }) {
  return (
    <div className="space-y-1.5">
      <div className="text-[11px] font-sans uppercase tracking-widest text-text-secondary">{label}</div>
      <div className={`text-xl font-display font-medium ${color}`}>
        {value}
      </div>
    </div>
  );
}
