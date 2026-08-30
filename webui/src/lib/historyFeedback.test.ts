import { describe, expect, it } from 'vitest'
import type { HistoryResponse } from '../types'
import { resolveChartDenoiseFeedback, responseWarningMessages } from './historyFeedback'

function response(partial: Partial<HistoryResponse>): HistoryResponse {
  return { data: [], count: 0, ...partial }
}

describe('responseWarningMessages', () => {
  it('normalizes string and structured successful-response warnings', () => {
    expect(responseWarningMessages('History', response({
      warnings: [
        'Duplicate timestamps were removed.',
        { code: 'range_repaired', message: 'An inconsistent OHLC row was repaired.' },
      ],
    }))).toEqual([
      'History: Duplicate timestamps were removed.',
      'History: An inconsistent OHLC row was repaired.',
    ])
  })

  it('can suppress an expected live-forming-candle warning', () => {
    expect(responseWarningMessages(
      'Live history',
      response({
        warnings: [{ code: 'forming_candle_included', message: 'The final candle may change.' }],
      }),
      new Set(['forming_candle_included'])
    )).toEqual([])
  })
})

describe('resolveChartDenoiseFeedback', () => {
  it('marks a successful raw fallback as skipped and keeps its reason visible', () => {
    const feedback = resolveChartDenoiseFeedback(
      { method: 'wavelet', columns: ['rsi_14'] },
      response({
        denoise_status: 'skipped',
        denoise_applied: false,
        denoise_status_reason: 'PyWavelets is unavailable.',
      })
    )
    expect(feedback.state).toBe('skipped')
    expect(feedback.title).toMatch(/PyWavelets is unavailable/)
    expect(feedback.warning).toMatch(/wavelet was skipped/)
  })

  it('reports the effective method and lets a degraded live tail override the base status', () => {
    const primary = response({ denoise_status: 'applied', denoise_method: 'ema' })
    expect(resolveChartDenoiseFeedback({ method: 'wavelet' }, primary).warning).toMatch(
      /server applied ema/
    )

    const live = response({
      denoise_status: 'failed',
      denoise_status_reason: 'Not enough live context.',
    })
    expect(resolveChartDenoiseFeedback({ method: 'wavelet' }, primary, live).state).toBe('failed')
  })
})
