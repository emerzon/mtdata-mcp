import { describe, expect, it } from 'vitest'
import {
  backtestDisplayRows,
  chartPriceFromCompactRow,
  mapCompactForecastToSeries,
  rowIntervalBounds,
} from './compactForecast'
import type { BacktestResult } from '../types'

describe('chartPriceFromCompactRow', () => {
  it('prefers price over value', () => {
    expect(chartPriceFromCompactRow({ time: 't', price: 101, value: 1 })).toBe(101)
  })

  it('falls back to value', () => {
    expect(chartPriceFromCompactRow({ time: 't', value: 1.2 })).toBe(1.2)
  })

  it('rejects non-finite prices', () => {
    expect(chartPriceFromCompactRow({ time: 't', value: Number.NaN })).toBeUndefined()
    expect(chartPriceFromCompactRow({ time: 't' })).toBeUndefined()
  })
})

describe('rowIntervalBounds', () => {
  it('reads price bounds from the row', () => {
    expect(
      rowIntervalBounds({ time: 't', value: 100, lower_price: 99, upper_price: 101 })
    ).toEqual({ lower: 99, upper: 101 })
  })

  it('reads generic lower/upper bounds', () => {
    expect(rowIntervalBounds({ time: 't', value: 100, lower: 98, upper: 102 })).toEqual({
      lower: 98,
      upper: 102,
    })
  })
})

describe('mapCompactForecastToSeries', () => {
  it('builds chart points from time + price/value', () => {
    const series = mapCompactForecastToSeries([
      { time: '2024-01-01T00:00:00Z', value: 1.1 },
      { time: '2024-01-01T01:00:00Z', price: 1.2 },
    ])
    expect(series.values).toEqual([1.1, 1.2])
    expect(series.times).toHaveLength(2)
    expect(series.lower).toBeUndefined()
    expect(series.upper).toBeUndefined()
  })

  it('attaches interval bounds from the row', () => {
    const series = mapCompactForecastToSeries([
      { time: '2024-01-01T00:00:00Z', value: 100, lower_price: 99, upper_price: 101 },
      { time: '2024-01-01T01:00:00Z', value: 101, lower_price: 99.5, upper_price: 102.5 },
    ])
    expect(series.lower).toEqual([99, 99.5])
    expect(series.upper).toEqual([101, 102.5])
  })

  it('throws when a compact row has no finite chart price', () => {
    expect(() =>
      mapCompactForecastToSeries([{ time: '2024-01-01T00:00:00Z', return: 0.01 }])
    ).toThrow(/no finite chart price/)
  })

  it('does not synthesize times or read parallel arrays', () => {
    const series = mapCompactForecastToSeries([{ time: '2024-01-01T00:00:00Z', value: 1 }])
    expect(series.times[0]).toBeGreaterThan(0)
    expect(series.values).toEqual([1])
  })
})

describe('backtestDisplayRows', () => {
  it('uses full results ordered by ranked_methods', () => {
    const result: BacktestResult = {
      symbol: 'EURUSD',
      timeframe: 'H1',
      horizon: 12,
      steps: 5,
      spacing: 20,
      ranked_methods: [
        { method: 'arima', avg_rmse: 0.01 },
        { method: 'theta', avg_rmse: 0.02 },
      ],
      results: {
        theta: { avg_mae: 0.2, avg_rmse: 0.02, avg_directional_accuracy: 0.55 },
        arima: { avg_mae: 0.1, avg_rmse: 0.01, avg_directional_accuracy: 0.61 },
      },
    }
    expect(backtestDisplayRows(result)).toEqual([
      { method: 'arima', avg_mae: 0.1, avg_rmse: 0.01, avg_directional_accuracy: 0.61 },
      { method: 'theta', avg_mae: 0.2, avg_rmse: 0.02, avg_directional_accuracy: 0.55 },
    ])
  })

  it('does not treat reduced ranked rows as the display source', () => {
    const result: BacktestResult = {
      symbol: 'EURUSD',
      timeframe: 'H1',
      horizon: 12,
      steps: 5,
      spacing: 20,
      ranked_methods: [{ method: 'theta', avg_rmse: 0.02 }],
      results: {
        theta: { avg_mae: 0.2, avg_directional_accuracy: 0.55 },
      },
    }
    const rows = backtestDisplayRows(result)
    expect(rows[0].avg_mae).toBe(0.2)
    expect(rows[0].avg_directional_accuracy).toBe(0.55)
  })

  it('falls back to results map when ranking is absent', () => {
    const result: BacktestResult = {
      symbol: 'EURUSD',
      timeframe: 'H1',
      horizon: 12,
      steps: 5,
      spacing: 20,
      results: {
        theta: { avg_mae: 0.2 },
      },
    }
    expect(backtestDisplayRows(result)).toEqual([{ method: 'theta', avg_mae: 0.2 }])
  })
})
