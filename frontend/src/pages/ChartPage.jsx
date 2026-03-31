import React, { useState, useEffect, useCallback } from 'react';
import LiveChart from '../components/LiveChart';
import { fetchCandles } from '../api/client';

/**
 * Candle counts per timeframe — sized to cover ~3 months of data
 * without causing memory issues.
 */
const CANDLE_COUNTS = {
  '1m':  1875,    // ~5 trading days only
  '5m':  24000,   // ~3 months
  '15m': 8500,    // ~3 months
  '1h':  2000,    // ~3 months
  '1d':  500,     // ~2 years
};

export default function ChartPage() {
  const [symbol, setSymbol]       = useState('NIFTY');
  const [timeframe, setTimeframe] = useState('5m');
  const [candles, setCandles]     = useState(null);
  const [loading, setLoading]     = useState(true);
  const [error, setError]         = useState(null);

  const loadCandles = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const count = CANDLE_COUNTS[timeframe] || 24000;
      const data = await fetchCandles(symbol, count, timeframe);
      setCandles(data);
    } catch (err) {
      console.error('Failed to load candles:', err);
      setError(err.message || 'Failed to load chart data');
    } finally {
      setLoading(false);
    }
  }, [symbol, timeframe]);

  // Load on mount and on symbol/timeframe change
  useEffect(() => {
    loadCandles();
  }, [loadCandles]);

  if (loading && !candles) {
    return (
      <div className="flex items-center justify-center flex-1 bg-bg">
        <div className="text-text-secondary text-sm animate-pulse">Loading chart…</div>
      </div>
    );
  }

  if (error && !candles) {
    return (
      <div className="flex items-center justify-center flex-1 bg-bg">
        <div className="text-center space-y-2">
          <div className="text-red-400 text-sm">{error}</div>
          <button
            onClick={loadCandles}
            className="px-4 py-1.5 bg-highlight/20 text-highlight text-xs font-semibold rounded border border-highlight/30 hover:bg-highlight/30 transition-colors"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 p-3">
      <LiveChart
        candles={candles}
        signals={null}
        openTrades={null}
        closedTrades={null}
        symbol={symbol}
        onSymbolChange={setSymbol}
        timeframe={timeframe}
        onTimeframeChange={setTimeframe}
      />
    </div>
  );
}
