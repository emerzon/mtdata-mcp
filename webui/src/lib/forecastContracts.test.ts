import { describe, expect, it } from 'vitest'
import type { MethodInfo } from '../types'
import {
  forecastMethodParams,
  normalizeForecastSettings,
  scopedBacktestParams,
  sharedBacktestParamDefs,
  updateMethodParameter,
} from './forecastContracts'

const catalog: MethodInfo[] = [
  {
    method: 'arima',
    available: true,
    requires: [],
    description: 'ARIMA',
    params: [
      { name: 'p', type: 'int' },
      { name: 'alpha', type: 'float' },
    ],
  },
  {
    method: 'theta',
    available: true,
    requires: [],
    description: 'Theta',
    params: [{ name: 'alpha', type: 'float' }],
  },
  {
    method: 'naive',
    available: true,
    requires: [],
    description: 'Naive',
    params: [],
  },
]

describe('forecast method parameter ownership', () => {
  it('migrates a legacy flat map under its saved method only', () => {
    const arima = normalizeForecastSettings({ method: 'arima', params: { p: 2 } })
    expect(forecastMethodParams(arima, catalog[0].params)).toEqual({ p: 2 })

    const naive = { ...arima, method: 'naive' }
    expect(forecastMethodParams(naive, catalog[2].params)).toBeUndefined()
  })

  it('sanitizes a legacy cross-method leak saved under a parameterless method', () => {
    const naive = normalizeForecastSettings({ method: 'naive', params: { p: 2 } })
    expect(forecastMethodParams(naive, catalog[2].params)).toBeUndefined()
  })

  it('retains separate values per method and removes empty inputs', () => {
    let params = updateMethodParameter({}, 'arima', 'p', '2', 'int')
    params = updateMethodParameter(params, 'theta', 'alpha', '0.2', 'float')
    expect(params).toEqual({ arima: { p: 2 }, theta: { alpha: 0.2 } })

    params = updateMethodParameter(params, 'arima', 'p', '', 'int')
    expect(params).toEqual({ theta: { alpha: 0.2 } })
  })
})

describe('backtest advanced parameter contracts', () => {
  it('offers only metadata fields shared by every selected method', () => {
    expect(sharedBacktestParamDefs(['arima', 'theta'], catalog).map((item) => item.name)).toEqual([
      'alpha',
    ])
    expect(sharedBacktestParamDefs(['arima', 'naive'], catalog)).toEqual([])
  })

  it('filters hidden shared and per-method values from the request', () => {
    expect(
      scopedBacktestParams(
        ['arima', 'theta'],
        catalog,
        { alpha: 0.1, p: 3, stale: true },
        {
          arima: { p: 2, unknown: 'drop' },
          theta: { alpha: 0.3 },
          naive: { stale: true },
        }
      )
    ).toEqual({
      params: { alpha: 0.1 },
      params_per_method: {
        arima: { p: 2 },
        theta: { alpha: 0.3 },
      },
    })
  })
})
