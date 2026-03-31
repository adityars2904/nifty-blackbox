/**
 * Technical Indicators Library
 * Compatible with all timeframes - works on any OHLCV data array
 */

/**
 * Calculate Simple Moving Average (SMA)
 * @param {Array} data - Array of price values
 * @param {number} period - Period for SMA
 * @returns {Array} - Array of SMA values (null for incomplete periods)
 */
export function calculateSMA(data, period) {
  const result = [];

  for (let i = 0; i < data.length; i++) {
    if (i < period - 1) {
      result.push(null);
      continue;
    }

    let sum = 0;
    for (let j = 0; j < period; j++) {
      sum += data[i - j];
    }
    result.push(sum / period);
  }

  return result;
}

/**
 * Calculate Exponential Moving Average (EMA)
 * @param {Array} data - Array of price values
 * @param {number} period - Period for EMA
 * @returns {Array} - Array of EMA values
 */
export function calculateEMA(data, period) {
  const result = [];
  const multiplier = 2 / (period + 1);

  // First EMA value is SMA
  let sum = 0;
  for (let i = 0; i < period; i++) {
    if (i >= data.length) break;
    sum += data[i];
  }

  let ema = sum / period;

  for (let i = 0; i < data.length; i++) {
    if (i < period - 1) {
      result.push(null);
    } else if (i === period - 1) {
      result.push(ema);
    } else {
      ema = (data[i] - ema) * multiplier + ema;
      result.push(ema);
    }
  }

  return result;
}

/**
 * Calculate Relative Strength Index (RSI)
 * @param {Array} data - Array of price values (typically close prices)
 * @param {number} period - Period for RSI (typically 14)
 * @returns {Array} - Array of RSI values (0-100)
 */
export function calculateRSI(data, period = 14) {
  if (data.length < period + 1) {
    return new Array(data.length).fill(null);
  }

  const result = [];
  const changes = [];

  // Calculate price changes
  for (let i = 1; i < data.length; i++) {
    changes.push(data[i] - data[i - 1]);
  }

  // Calculate initial average gain and loss
  let avgGain = 0;
  let avgLoss = 0;

  for (let i = 0; i < period; i++) {
    if (changes[i] > 0) {
      avgGain += changes[i];
    } else {
      avgLoss += Math.abs(changes[i]);
    }
  }

  avgGain /= period;
  avgLoss /= period;

  // First RSI value
  result.push(null); // No RSI for first data point
  for (let i = 0; i < period; i++) {
    result.push(null);
  }

  let rs = avgGain / avgLoss;
  let rsi = 100 - (100 / (1 + rs));
  result.push(rsi);

  // Calculate subsequent RSI values using smoothed averages
  for (let i = period; i < changes.length; i++) {
    const change = changes[i];
    const gain = change > 0 ? change : 0;
    const loss = change < 0 ? Math.abs(change) : 0;

    avgGain = (avgGain * (period - 1) + gain) / period;
    avgLoss = (avgLoss * (period - 1) + loss) / period;

    rs = avgGain / avgLoss;
    rsi = 100 - (100 / (1 + rs));
    result.push(rsi);
  }

  return result;
}

/**
 * Calculate Moving Average Convergence Divergence (MACD)
 * @param {Array} data - Array of price values
 * @param {number} fastPeriod - Fast EMA period (default 12)
 * @param {number} slowPeriod - Slow EMA period (default 26)
 * @param {number} signalPeriod - Signal line period (default 9)
 * @returns {Object} - { macd, signal, histogram }
 */
export function calculateMACD(data, fastPeriod = 12, slowPeriod = 26, signalPeriod = 9) {
  const fastEMA = calculateEMA(data, fastPeriod);
  const slowEMA = calculateEMA(data, slowPeriod);

  const macdLine = [];
  for (let i = 0; i < data.length; i++) {
    if (fastEMA[i] === null || slowEMA[i] === null) {
      macdLine.push(null);
    } else {
      macdLine.push(fastEMA[i] - slowEMA[i]);
    }
  }

  // Calculate signal line (EMA of MACD)
  const macdValues = macdLine.filter(v => v !== null);
  const signalEMA = calculateEMA(macdValues, signalPeriod);

  // Align signal line with MACD line
  const signalLine = [];
  let signalIndex = 0;
  for (let i = 0; i < macdLine.length; i++) {
    if (macdLine[i] === null) {
      signalLine.push(null);
    } else {
      signalLine.push(signalEMA[signalIndex] || null);
      signalIndex++;
    }
  }

  // Calculate histogram
  const histogram = [];
  for (let i = 0; i < macdLine.length; i++) {
    if (macdLine[i] === null || signalLine[i] === null) {
      histogram.push(null);
    } else {
      histogram.push(macdLine[i] - signalLine[i]);
    }
  }

  return {
    macd: macdLine,
    signal: signalLine,
    histogram: histogram
  };
}

/**
 * Calculate Bollinger Bands
 * @param {Array} data - Array of price values
 * @param {number} period - Period for moving average (default 20)
 * @param {number} stdDev - Standard deviation multiplier (default 2)
 * @returns {Object} - { upper, middle, lower }
 */
export function calculateBollingerBands(data, period = 20, stdDev = 2) {
  const middle = calculateSMA(data, period);
  const upper = [];
  const lower = [];

  for (let i = 0; i < data.length; i++) {
    if (i < period - 1 || middle[i] === null) {
      upper.push(null);
      lower.push(null);
      continue;
    }

    // Calculate standard deviation
    let sumSquares = 0;
    for (let j = 0; j < period; j++) {
      const diff = data[i - j] - middle[i];
      sumSquares += diff * diff;
    }
    const sd = Math.sqrt(sumSquares / period);

    upper.push(middle[i] + (stdDev * sd));
    lower.push(middle[i] - (stdDev * sd));
  }

  return { upper, middle, lower };
}

/**
 * Calculate Stochastic Oscillator
 * @param {Array} highData - Array of high prices
 * @param {Array} lowData - Array of low prices
 * @param {Array} closeData - Array of close prices
 * @param {number} period - Period for %K (default 14)
 * @param {number} smoothK - Smoothing period for %K (default 3)
 * @param {number} smoothD - Smoothing period for %D (default 3)
 * @returns {Object} - { k, d }
 */
export function calculateStochastic(highData, lowData, closeData, period = 14, smoothK = 3, smoothD = 3) {
  const rawK = [];

  for (let i = 0; i < closeData.length; i++) {
    if (i < period - 1) {
      rawK.push(null);
      continue;
    }

    let highestHigh = highData[i];
    let lowestLow = lowData[i];

    for (let j = 0; j < period; j++) {
      if (highData[i - j] > highestHigh) highestHigh = highData[i - j];
      if (lowData[i - j] < lowestLow) lowestLow = lowData[i - j];
    }

    const k = ((closeData[i] - lowestLow) / (highestHigh - lowestLow)) * 100;
    rawK.push(k);
  }

  // Smooth %K
  const kValues = rawK.filter(v => v !== null);
  const smoothedK = calculateSMA(kValues, smoothK);

  // Calculate %D (SMA of %K)
  const dValues = calculateSMA(smoothedK.filter(v => v !== null), smoothD);

  // Align with original data
  const k = [];
  const d = [];
  let kIndex = 0;
  let dIndex = 0;

  for (let i = 0; i < rawK.length; i++) {
    if (rawK[i] === null) {
      k.push(null);
      d.push(null);
    } else {
      k.push(smoothedK[kIndex] || null);
      d.push(dValues[dIndex] || null);
      kIndex++;
      if (smoothedK[kIndex - smoothK] !== null) dIndex++;
    }
  }

  return { k, d };
}

/**
 * Format indicator data for Lightweight Charts
 * @param {Array} candleData - Original OHLCV candle data
 * @param {Array} indicatorValues - Calculated indicator values
 * @returns {Array} - Formatted data for chart
 */
export function formatIndicatorForChart(candleData, indicatorValues) {
  return candleData.map((candle, i) => ({
    time: candle.time,
    value: indicatorValues[i]
  })).filter(d => d.value !== null && d.value !== undefined);
}

/**
 * All available indicators configuration
 */
export const INDICATORS = {
  EMA_9: {
    id: 'EMA_9',
    name: 'EMA (9)',
    type: 'overlay',
    color: '#2196F3',
    calculate: (data) => calculateEMA(data.map(d => d.close), 9)
  },
  EMA_21: {
    id: 'EMA_21',
    name: 'EMA (21)',
    type: 'overlay',
    color: '#FF9800',
    calculate: (data) => calculateEMA(data.map(d => d.close), 21)
  },
  EMA_50: {
    id: 'EMA_50',
    name: 'EMA (50)',
    type: 'overlay',
    color: '#9C27B0',
    calculate: (data) => calculateEMA(data.map(d => d.close), 50)
  },
  SMA_20: {
    id: 'SMA_20',
    name: 'SMA (20)',
    type: 'overlay',
    color: '#00BCD4',
    calculate: (data) => calculateSMA(data.map(d => d.close), 20)
  },
  RSI_14: {
    id: 'RSI_14',
    name: 'RSI (14)',
    type: 'oscillator',
    color: '#9C27B0',
    calculate: (data) => calculateRSI(data.map(d => d.close), 14),
    levels: [30, 50, 70]
  },
  MACD: {
    id: 'MACD',
    name: 'MACD (12,26,9)',
    type: 'oscillator',
    colors: {
      macd: '#2196F3',
      signal: '#FF5252',
      histogram: '#4CAF50'
    },
    calculate: (data) => calculateMACD(data.map(d => d.close)),
    levels: [0]
  },
  BOLLINGER: {
    id: 'BOLLINGER',
    name: 'Bollinger Bands (20,2)',
    type: 'overlay',
    colors: {
      upper: '#F44336',
      middle: '#2196F3',
      lower: '#4CAF50'
    },
    calculate: (data) => calculateBollingerBands(data.map(d => d.close), 20, 2)
  },
  STOCHASTIC: {
    id: 'STOCHASTIC',
    name: 'Stochastic (14,3,3)',
    type: 'oscillator',
    colors: {
      k: '#2196F3',
      d: '#FF5252'
    },
    calculate: (data) => calculateStochastic(
      data.map(d => d.high),
      data.map(d => d.low),
      data.map(d => d.close),
      14, 3, 3
    ),
    levels: [20, 50, 80]
  },
  VWAP: {
    id: 'VWAP',
    name: 'VWAP',
    type: 'overlay',
    color: '#4488ff',
    calculate: (data) => data.map(d => d.vwap !== undefined ? d.vwap : d.close)
  }
};

/**
 * Get indicators by type
 */
export function getIndicatorsByType(type) {
  return Object.values(INDICATORS).filter(ind => ind.type === type);
}

/**
 * Get overlay indicators (those that go on the main chart)
 */
export function getOverlayIndicators() {
  return getIndicatorsByType('overlay');
}

/**
 * Get oscillator indicators (those that go in separate panes)
 */
export function getOscillatorIndicators() {
  return getIndicatorsByType('oscillator');
}
