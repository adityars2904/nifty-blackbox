import React, { useEffect, useRef, useState } from 'react';
import { createChart, ColorType, CrosshairMode } from 'lightweight-charts';
import { motion, AnimatePresence } from 'framer-motion';
import {
  INDICATORS,
  formatIndicatorForChart,
  getOverlayIndicators,
  getOscillatorIndicators,
} from '../utils/indicators';

const TIMEFRAMES = ['1m', '5m', '15m', '1h', '1d'];

export default function LiveChart({
  candles,
  signals,
  openTrades,
  closedTrades,
  symbol,
  onSymbolChange,
  timeframe,
  onTimeframeChange,
}) {
  // ── Refs ──────────────────────────────────────────────────────────────
  const mainChartContainerRef = useRef(null);
  const oscillatorChartContainerRef = useRef(null);
  const mainChartRef = useRef(null);
  const oscillatorChartRef = useRef(null);
  const candleSeriesRef = useRef(null);
  const volumeSeriesRef = useRef(null);
  const overlaySeriesRef = useRef([]);

  // ── State ─────────────────────────────────────────────────────────────
  const [selectedOverlays, setSelectedOverlays] = useState([]);
  const [selectedOscillator, setSelectedOscillator] = useState(null);
  const [showVolume, setShowVolume] = useState(true);

  const toggleOverlay = (id) =>
    setSelectedOverlays((prev) =>
      prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]
    );

  const overlayIndicators = getOverlayIndicators();
  const oscillatorIndicators = getOscillatorIndicators();

  // ─── 1. Main chart setup ─────────────────────────────────────────────
  useEffect(() => {
    if (!mainChartContainerRef.current) return;

    const chart = createChart(mainChartContainerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: '#161b22' },
        textColor: '#8b949e',
        fontFamily: "'JetBrains Mono', monospace",
        fontSize: 11,
      },
      grid: {
        vertLines: { color: '#21262d' },
        horzLines: { color: '#21262d' },
      },
      crosshair: { mode: CrosshairMode.Normal },
      rightPriceScale: { borderColor: '#30363d' },
      timeScale: { borderColor: '#30363d', timeVisible: true, secondsVisible: false },
      width: mainChartContainerRef.current.clientWidth,
      height: 420,
    });

    const candleSeries = chart.addCandlestickSeries({
      upColor: '#00d26a',
      downColor: '#f85149',
      borderUpColor: '#00d26a',
      borderDownColor: '#f85149',
      wickUpColor: '#00d26a',
      wickDownColor: '#f85149',
    });

    mainChartRef.current = chart;
    candleSeriesRef.current = candleSeries;

    const handleResize = () => {
      if (mainChartContainerRef.current) {
        chart.applyOptions({ width: mainChartContainerRef.current.clientWidth });
      }
    };
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      chart.remove();
      mainChartRef.current = null;
      candleSeriesRef.current = null;
      volumeSeriesRef.current = null;
      overlaySeriesRef.current = [];
    };
  }, []);

  // ─── 2. Set candle data + markers whenever candles change ─────────────
  useEffect(() => {
    const chart = mainChartRef.current;
    const series = candleSeriesRef.current;
    if (!chart || !series || !candles?.length) return;

    // Filter duplicates and invalid times to prevent lightweight-charts crash
    const dedupedCandles = [];
    const seenTimes = new Set();
    for (const c of candles) {
      if (!c || !c.time || isNaN(c.time) || seenTimes.has(c.time)) continue;
      seenTimes.add(c.time);
      dedupedCandles.push(c);
    }
    dedupedCandles.sort((a, b) => a.time - b.time);

    try {
      series.setData(dedupedCandles);
    } catch (err) {
      console.error("Error setting candle data:", err, dedupedCandles);
      return; // Stop here so we do not attempt markers if chart crashes
    }

    // ── Markers: signals + trade exits ───────────────────────────────────
    const markers = [];

    if (signals?.data) {
      for (const sig of signals.data) {
        if (!sig.approved || sig.symbol !== symbol) continue;
        const sigTime = Math.floor(new Date(sig.candle_ts || sig.created_at).getTime() / 1000);
        const closest = candles.reduce((prev, curr) =>
          Math.abs(curr.time - sigTime) < Math.abs(prev.time - sigTime) ? curr : prev
        );
        markers.push({
          time: closest.time,
          position: sig.direction === 'UP' ? 'belowBar' : 'aboveBar',
          color: sig.direction === 'UP' ? '#00d26a' : '#f85149',
          shape: sig.direction === 'UP' ? 'arrowUp' : 'arrowDown',
          text: `${sig.direction} entry`,
        });
      }
    }

    if (closedTrades?.data) {
      for (const t of closedTrades.data) {
        if (t.symbol !== symbol || !t.exit_time) continue;
        const exitTime = Math.floor(new Date(t.exit_time).getTime() / 1000);
        if (exitTime < candles[0].time || exitTime > candles[candles.length - 1].time) continue;

        const closest = candles.reduce((prev, curr) =>
          Math.abs(curr.time - exitTime) < Math.abs(prev.time - exitTime) ? curr : prev
        );

        const reason = t.exit_reason || '';
        let markerColor, markerShape, markerText;

        if (reason.includes('SL')) {
          markerColor = '#f85149';
          markerShape = 'square';
          markerText = `SL ${(t.pnl_points ?? 0) > 0 ? '+' : ''}${t.pnl_points?.toFixed(1) ?? ''}`;
        } else if (reason.includes('TP')) {
          markerColor = '#00d26a';
          markerShape = 'circle';
          markerText = `TP +${t.pnl_points?.toFixed(1) ?? ''}`;
        } else {
          markerColor = '#e3b341';
          markerShape = 'square';
          markerText = `EOD ${(t.pnl_points ?? 0) > 0 ? '+' : ''}${t.pnl_points?.toFixed(1) ?? ''}`;
        }

        markers.push({
          time: closest.time,
          position: t.direction === 'UP' ? 'aboveBar' : 'belowBar',
          color: markerColor,
          shape: markerShape,
          text: markerText,
        });
      }
    }

    if (markers.length) {
      markers.sort((a, b) => a.time - b.time);
      series.setMarkers(markers);
    } else {
      series.setMarkers([]);
    }

    // SL/TP price lines for open trades
    if (openTrades?.data) {
      const trade = openTrades.data.find((t) => t.symbol === symbol && t.status === 'OPEN');
      if (trade) {
        series.createPriceLine({ price: trade.stop_loss, color: '#f85149', lineWidth: 1, lineStyle: 2, axisLabelVisible: true, title: 'SL' });
        series.createPriceLine({ price: trade.take_profit, color: '#00d26a', lineWidth: 1, lineStyle: 2, axisLabelVisible: true, title: 'TP' });
        series.createPriceLine({ price: trade.entry_price, color: '#1f6feb', lineWidth: 1, lineStyle: 0, axisLabelVisible: true, title: 'Entry' });
      }
    }

    chart.timeScale().fitContent();
  }, [candles, signals, openTrades, closedTrades, symbol]);

  // ─── 3. Volume overlay ────────────────────────────────────────────────
  useEffect(() => {
    const chart = mainChartRef.current;
    if (!chart || !candles?.length) return;

    // Remove previous volume series
    if (volumeSeriesRef.current) {
      try { chart.removeSeries(volumeSeriesRef.current); } catch {}
      volumeSeriesRef.current = null;
    }

    if (!showVolume) return;

    const volSeries = chart.addHistogramSeries({
      priceFormat: { type: 'volume' },
      priceScaleId: 'vol',
      color: '#30363d',
    });
    chart.priceScale('vol').applyOptions({
      scaleMargins: { top: 0.85, bottom: 0 },
    });
    const volData = candles
      .filter((c) => c.volume > 0)
      .map((c) => ({
        time: c.time,
        value: c.volume,
        color: c.close >= c.open ? 'rgba(0,210,106,0.3)' : 'rgba(248,81,73,0.3)',
      }));
    if (volData.length > 0) {
      volSeries.setData(volData);
      volumeSeriesRef.current = volSeries;
    }
  }, [candles, showVolume]);

  // ─── 4. Overlay indicators (EMA, SMA, Bollinger) ─────────────────────
  useEffect(() => {
    const chart = mainChartRef.current;
    if (!chart || !candles?.length) return;

    // Remove old overlay series
    overlaySeriesRef.current.forEach((s) => {
      try { chart.removeSeries(s); } catch {}
    });
    overlaySeriesRef.current = [];

    selectedOverlays.forEach((indicatorId) => {
      const indicator = INDICATORS[indicatorId];
      if (!indicator || indicator.type !== 'overlay') return;

      try {
        if (indicatorId === 'BOLLINGER') {
          const { upper, middle, lower } = indicator.calculate(candles);
          const addLine = (color, data) => {
            const s = chart.addLineSeries({
              color,
              lineWidth: 1,
              priceLineVisible: false,
              lastValueVisible: false,
              crosshairMarkerVisible: false,
            });
            s.setData(formatIndicatorForChart(candles, data));
            return s;
          };
          overlaySeriesRef.current.push(
            addLine(indicator.colors.upper, upper),
            addLine(indicator.colors.middle, middle),
            addLine(indicator.colors.lower, lower)
          );
        } else {
          const values = indicator.calculate(candles);
          const lineSeries = chart.addLineSeries({
            color: indicator.color,
            lineWidth: 2,
            priceLineVisible: false,
            lastValueVisible: false,
            crosshairMarkerVisible: false,
          });
          lineSeries.setData(formatIndicatorForChart(candles, values));
          overlaySeriesRef.current.push(lineSeries);
        }
      } catch (err) {
        console.error(`Error adding overlay ${indicatorId}:`, err);
      }
    });
  }, [selectedOverlays, candles]);

  // ─── 5. Oscillator indicator (RSI, MACD, Stochastic) ─────────────────
  useEffect(() => {
    // Cleanup previous oscillator chart
    if (oscillatorChartRef.current) {
      try { oscillatorChartRef.current.remove(); } catch {}
      oscillatorChartRef.current = null;
    }

    if (!oscillatorChartContainerRef.current || !selectedOscillator || !candles?.length) return;

    const indicator = INDICATORS[selectedOscillator];
    if (!indicator || indicator.type !== 'oscillator') return;

    let oscChart;
    try {
      oscChart = createChart(oscillatorChartContainerRef.current, {
        layout: {
          background: { type: ColorType.Solid, color: '#161b22' },
          textColor: '#8b949e',
          fontFamily: "'JetBrains Mono', monospace",
          fontSize: 11,
        },
        grid: {
          vertLines: { color: '#21262d' },
          horzLines: { color: '#21262d' },
        },
        crosshair: { mode: CrosshairMode.Normal },
        rightPriceScale: { borderColor: '#30363d' },
        timeScale: { borderColor: '#30363d', timeVisible: true, secondsVisible: false },
        width: oscillatorChartContainerRef.current.clientWidth,
        height: 150,
      });

      oscillatorChartRef.current = oscChart;

      if (selectedOscillator === 'RSI_14') {
        const values = indicator.calculate(candles);
        const series = oscChart.addLineSeries({ color: indicator.color, lineWidth: 2 });
        series.setData(formatIndicatorForChart(candles, values));
        indicator.levels?.forEach((level) => {
          const l = oscChart.addLineSeries({ color: '#30363d', lineWidth: 1, lineStyle: 2, priceLineVisible: false, lastValueVisible: false });
          l.setData(candles.map((d) => ({ time: d.time, value: level })));
        });
      } else if (selectedOscillator === 'MACD') {
        const { macd, signal, histogram } = indicator.calculate(candles);
        const hSeries = oscChart.addHistogramSeries({ color: indicator.colors.histogram });
        hSeries.setData(formatIndicatorForChart(candles, histogram));
        const mSeries = oscChart.addLineSeries({ color: indicator.colors.macd, lineWidth: 2 });
        mSeries.setData(formatIndicatorForChart(candles, macd));
        const sSeries = oscChart.addLineSeries({ color: indicator.colors.signal, lineWidth: 2 });
        sSeries.setData(formatIndicatorForChart(candles, signal));
      } else if (selectedOscillator === 'STOCHASTIC') {
        const { k, d } = indicator.calculate(candles);
        const kSeries = oscChart.addLineSeries({ color: indicator.colors.k, lineWidth: 2 });
        kSeries.setData(formatIndicatorForChart(candles, k));
        const dSeries = oscChart.addLineSeries({ color: indicator.colors.d, lineWidth: 2 });
        dSeries.setData(formatIndicatorForChart(candles, d));
        indicator.levels?.forEach((level) => {
          const l = oscChart.addLineSeries({ color: '#30363d', lineWidth: 1, lineStyle: 2, priceLineVisible: false, lastValueVisible: false });
          l.setData(candles.map((d) => ({ time: d.time, value: level })));
        });
      }

      oscChart.timeScale().fitContent();

      // Sync time scales
      const syncHandler = (range) => {
        if (range && oscillatorChartRef.current) {
          try { oscChart.timeScale().setVisibleLogicalRange(range); } catch {}
        }
      };

      if (mainChartRef.current) {
        mainChartRef.current.timeScale().subscribeVisibleLogicalRangeChange(syncHandler);
        const range = mainChartRef.current.timeScale().getVisibleLogicalRange();
        if (range) oscChart.timeScale().setVisibleLogicalRange(range);
      }

      const handleResize = () => {
        if (oscChart && oscillatorChartContainerRef.current) {
          oscChart.applyOptions({ width: oscillatorChartContainerRef.current.clientWidth });
        }
      };
      window.addEventListener('resize', handleResize);

      return () => {
        window.removeEventListener('resize', handleResize);
        if (mainChartRef.current) {
          try { mainChartRef.current.timeScale().unsubscribeVisibleLogicalRangeChange(syncHandler); } catch {}
        }
        if (oscillatorChartRef.current) {
          try { oscillatorChartRef.current.remove(); } catch {}
          oscillatorChartRef.current = null;
        }
      };
    } catch (err) {
      console.error('Error creating oscillator chart:', err);
    }
  }, [selectedOscillator, candles]);

  // ── Render ────────────────────────────────────────────────────────────
  return (
    <div className="bg-surface border border-border rounded-lg overflow-hidden">
      {/* ── Toolbar ── */}
      <div className="flex flex-wrap items-center gap-3 px-3 py-2 border-b border-border">
        {/* Symbol selector */}
        <div className="flex items-center gap-2">
          <span className="text-[10px] text-text-secondary uppercase tracking-wider font-semibold">Symbol</span>
          <div className="flex gap-1">
            {['NIFTY', 'BANKNIFTY'].map((s) => (
              <button
                key={s}
                onClick={() => onSymbolChange(s)}
                className={`px-2 py-0.5 rounded text-[10px] font-bold border transition-colors ${
                  symbol === s
                    ? 'border-highlight bg-highlight/20 text-highlight'
                    : 'border-border text-text-secondary hover:border-highlight'
                }`}
              >
                {s}
              </button>
            ))}
          </div>
        </div>

        <div className="w-px h-4 bg-border" />

        {/* Timeframe selector */}
        <div className="flex items-center gap-2">
          <span className="text-[10px] text-text-secondary uppercase tracking-wider font-semibold">TF</span>
          <div className="flex gap-1">
            {TIMEFRAMES.map((tf) => (
              <button
                key={tf}
                onClick={() => onTimeframeChange(tf)}
                className={`px-2 py-0.5 rounded text-[10px] font-bold border transition-colors ${
                  timeframe === tf
                    ? 'border-accent bg-accent/20 text-accent'
                    : 'border-border text-text-secondary hover:border-accent'
                }`}
              >
                {tf}
              </button>
            ))}
          </div>
        </div>

        <div className="w-px h-4 bg-border" />

        {/* Volume toggle */}
        <button
          onClick={() => setShowVolume((v) => !v)}
          className={`px-2 py-0.5 rounded text-[10px] font-bold border transition-colors ${
            showVolume
              ? 'border-text-secondary bg-text-secondary/15 text-text-primary'
              : 'border-border text-text-secondary hover:border-text-secondary'
          }`}
        >
          Vol
        </button>
      </div>

      {/* ── Indicator controls ── */}
      <div className="flex flex-wrap items-center gap-3 px-3 py-2 border-b border-border bg-bg/40">
        {/* Overlays */}
        <div className="flex items-center gap-2">
          <span className="text-[10px] text-text-secondary uppercase tracking-wider font-semibold">Overlays</span>
          <div className="flex gap-1">
            {overlayIndicators.map((ind) => {
              const isActive = selectedOverlays.includes(ind.id);
              const activeColor = ind.color || ind.colors?.middle || '#888';
              return (
                <button
                  key={ind.id}
                  onClick={() => toggleOverlay(ind.id)}
                  className="px-2 py-0.5 rounded text-[10px] font-bold border transition-colors"
                  style={
                    isActive
                      ? {
                          backgroundColor: activeColor + '22',
                          borderColor: activeColor + '44',
                          color: activeColor,
                        }
                      : {
                          borderColor: '#30363d',
                          color: '#8b949e',
                        }
                  }
                >
                  {ind.name}
                </button>
              );
            })}
          </div>
        </div>

        <div className="w-px h-4 bg-border" />

        {/* Oscillators */}
        <div className="flex items-center gap-2">
          <span className="text-[10px] text-text-secondary uppercase tracking-wider font-semibold">Oscillator</span>
          <div className="flex gap-1">
            {oscillatorIndicators.map((ind) => {
              const isActive = selectedOscillator === ind.id;
              const activeColor = ind.color || ind.colors?.macd || '#888';
              return (
                <button
                  key={ind.id}
                  onClick={() =>
                    setSelectedOscillator((prev) => (prev === ind.id ? null : ind.id))
                  }
                  className="px-2 py-0.5 rounded text-[10px] font-bold border transition-colors"
                  style={
                    isActive
                      ? {
                          backgroundColor: activeColor + '22',
                          borderColor: activeColor + '44',
                          color: activeColor,
                        }
                      : {
                          borderColor: '#30363d',
                          color: '#8b949e',
                        }
                  }
                >
                  {ind.name}
                </button>
              );
            })}
          </div>
        </div>
      </div>

      {/* ── Main chart ── */}
      <div ref={mainChartContainerRef} className="w-full" />

      {/* ── Oscillator chart ── */}
      <AnimatePresence>
        {selectedOscillator && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.3, ease: [0.16, 1, 0.3, 1] }}
            className="border-t border-border"
          >
            <div ref={oscillatorChartContainerRef} className="w-full" />
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

// =============================================================================
// CHANGES
// =============================================================================
// Cleaned up chart data filtering to prevent lightweight-charts crashes due to 
// duplicate timestamps or unsorted data. VWAP indicator logic moved to the standard
// `indicators.js` overlay functionality for a cleaner UI integration.
// No model impact. No retraining required.
// =============================================================================
