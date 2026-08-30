import { describe, expect, it } from 'vitest'
import {
  backtestDisplayRows,
  backtestMethodStatus,
  backtestResultFeedback,
  chartPriceFromCompactRow,
  computeAnchorForecastMetrics,
  forecastResultFeedback,
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

describe('computeAnchorForecastMetrics', () => {
  it('scores terminal direction from the anchor without future actual baselines', () => {
    const metrics = computeAnchorForecastMetrics([110, 109], [90, 95], 100)
    expect(metrics?.overlap).toBe(2)
    expect(metrics?.dirAcc).toBe(0)
  })

  it('matches a correct terminal call even when intermediate path directions differ', () => {
    const metrics = computeAnchorForecastMetrics([90, 110], [110, 105], 100)
    expect(metrics?.dirAcc).toBe(100)
  })

  it('marks a flat terminal forecast as no directional call', () => {
    expect(computeAnchorForecastMetrics([100], [101], 100)?.dirAcc).toBeNull()
  })
})

describe('forecastResultFeedback', () => {
  it('surfaces low-trust successful output, blockers, and warnings', () => {
    const feedback = forecastResultFeedback({
      forecast_status: 'non_informative',
      signal_status: 'not_actionable',
      trust_level: 'low',
      trust_blockers: ['insufficient_history_sample'],
      ci_status: 'unavailable',
      forecast_mode: 'point_only',
      warnings: ['Low-history forecast.'],
      uncertainty: { reason: 'Requested intervals were not produced.' },
    })
    expect(feedback.tone).toBe('warning')
    expect(feedback.summary).toContain('Forecast: non informative')
    expect(feedback.summary).toContain('Trust: low')
    expect(feedback.summary).toContain('Mode: point only')
    expect(feedback.details).toContain('Low-history forecast.')
    expect(feedback.details).toContain('Trust blocker: insufficient history sample')
    expect(feedback.details).toContain('Requested intervals were not produced.')
  })

  it('reports an ordinary successful forecast without a warning tone', () => {
    expect(
      forecastResultFeedback({
        forecast_status: 'informative',
        signal_status: 'actionable',
        trust_level: 'adequate',
        ci_status: 'available',
      }).tone
    ).toBe('success')
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

  it('keeps ranking eligibility metadata while using full result metrics', () => {
    const result: BacktestResult = {
      symbol: 'EURUSD',
      timeframe: 'H1',
      horizon: 12,
      steps: 2,
      spacing: 20,
      ranked_methods: [{ method: 'theta', ranking_status: 'unranked' }],
      results: {
        theta: {
          status: 'partial',
          avg_mae: 0.2,
          successful_tests: 1,
          failed_tests: 1,
          num_tests: 2,
        },
      },
    }
    expect(backtestDisplayRows(result)[0]).toMatchObject({
      method: 'theta',
      ranking_status: 'unranked',
      status: 'partial',
      avg_mae: 0.2,
    })
  })
})

describe('backtest status feedback', () => {
  it('marks partial and failed method rows from compact diagnostics', () => {
    expect(backtestMethodStatus({ complete_success: false, failed_tests: 1 })).toBe('partial')
    expect(backtestMethodStatus({ success: false })).toBe('failed')
  })

  it('summarizes root partial status and anchor failures', () => {
    const feedback = backtestResultFeedback({
      symbol: 'EURUSD',
      timeframe: 'H1',
      horizon: 12,
      steps: 2,
      spacing: 20,
      status: 'partial',
      methods_partial: 1,
      methods_failed: 1,
      anchor_tests_planned: 4,
      anchor_tests_succeeded: 2,
      anchor_tests_failed: 2,
      warnings: ['2 of 4 planned anchor tests failed.'],
    })
    expect(feedback.tone).toBe('warning')
    expect(feedback.summary).toBe('Backtest: partial')
    expect(feedback.details).toContain('1 method(s) completed only partially.')
    expect(feedback.details).toContain('1 method(s) failed.')
    expect(feedback.details).toContain('2/4 anchor tests succeeded; 2 failed.')
  })
})
