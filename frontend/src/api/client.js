/**
 * API client — all HTTP calls to the backend.
 * Simplified for chart-only mode.
 */

class ApiError extends Error {
  constructor(message, status, url) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.url = url;
  }
}

async function request(url) {
  const res = await fetch(url);
  if (!res.ok) {
    throw new ApiError(`HTTP ${res.status}`, res.status, url);
  }
  return res.json();
}

export function fetchHealth() {
  return request('/api/health');
}

export function fetchCandles(symbol = 'NIFTY', n = 200, timeframe = '5m') {
  return request(`/api/candles?symbol=${encodeURIComponent(symbol)}&n=${n}&timeframe=${encodeURIComponent(timeframe)}`);
}

export function fetchBacktestMetrics() {
  return request('/api/research/metrics');
}

export function fetchBacktestTrades(symbol) {
  const query = symbol ? `?symbol=${encodeURIComponent(symbol)}` : '';
  return request(`/api/research/trades${query}`);
}
