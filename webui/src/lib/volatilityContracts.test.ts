import { describe, expect, it } from 'vitest'
import {
  buildVolatilityRequest,
  volatilityResultMetrics,
} from './volatilityContracts'

describe('volatility request contracts', () => {
  it('omits proxy for the default direct EWMA estimator', () => {
    const request = buildVolatilityRequest({
      symbol: 'EURUSD',
      timeframe: 'H1',
      method: 'ewma',
      horizon: 12,
      proxy: 'squared_return',
      methodInfo: {
        method: 'ewma',
        available: true,
        requires: [],
        params: [],
      },
    })

    expect(request).not.toHaveProperty('proxy')
  })

  it('includes a valid proxy for methods that require one', () => {
    const request = buildVolatilityRequest({
      symbol: 'EURUSD',
      timeframe: 'H1',
      method: 'theta',
      horizon: 12,
      proxy: 'not_valid',
      methodInfo: {
        method: 'theta',
        available: true,
        requires: [],
        params: [],
        requires_proxy: true,
        valid_proxies: ['squared_return', 'abs_return'],
      },
    })

    expect(request.proxy).toBe('squared_return')
  })

  it('presents requested-horizon volatility before annualized context', () => {
    const metrics = volatilityResultMetrics({
      symbol: 'EURUSD',
      timeframe: 'H1',
      method: 'ewma',
      horizon: 12,
      volatility_horizon: 0.021234,
      volatility_annualized: 0.194444,
    }, 5)

    expect(metrics.map(({ label, primary }) => ({ label, primary }))).toEqual([
      { label: '12-bar volatility', primary: true },
      { label: 'Annualized volatility', primary: false },
    ])
    expect(metrics[0].percent).toBeCloseTo(2.1234)
    expect(metrics[1].percent).toBeCloseTo(19.4444)
  })
})
