import { describe, expect, it } from 'vitest'
import {
  DEFAULT_PIVOT_METHOD,
  DEFAULT_SR_CONTROLS,
  buildPivotQuery,
  buildSupportResistanceQuery,
  clampSrLookback,
  normalizePivotMethod,
  normalizeSrControls,
} from './overlayParams'

describe('normalizePivotMethod', () => {
  it('defaults unknown methods to classic', () => {
    expect(normalizePivotMethod(undefined)).toBe(DEFAULT_PIVOT_METHOD)
    expect(normalizePivotMethod('nope')).toBe('classic')
  })

  it('accepts known methods case-insensitively', () => {
    expect(normalizePivotMethod('Fibonacci')).toBe('fibonacci')
    expect(normalizePivotMethod('demark')).toBe('demark')
  })
})

describe('buildPivotQuery', () => {
  it('builds API-ready pivot params from UI selections', () => {
    expect(buildPivotQuery('EURUSD', 'H1', 'woodie')).toEqual({
      symbol: 'EURUSD',
      timeframe: 'H1',
      method: 'woodie',
    })
  })
})

describe('SR controls', () => {
  it('clamps lookback to API bounds', () => {
    expect(clampSrLookback(50)).toBe(100)
    expect(clampSrLookback(50000)).toBe(20000)
    expect(clampSrLookback(350)).toBe(350)
  })

  it('normalizes partial controls onto defaults', () => {
    expect(normalizeSrControls({ lookback: 400 })).toEqual({
      ...DEFAULT_SR_CONTROLS,
      lookback: 400,
    })
  })

  it('builds support-resistance query for the client', () => {
    expect(
      buildSupportResistanceQuery('XAUUSD', 'M15', {
        lookback: 500,
        min_touches: 3,
        max_levels: 6,
        tolerance_pct: 0.25,
      })
    ).toEqual({
      symbol: 'XAUUSD',
      timeframe: 'M15',
      lookback: 500,
      min_touches: 3,
      max_levels: 6,
      tolerance_pct: 0.25,
    })
  })
})
