import { describe, expect, it } from 'vitest'
import type { HistoryBar } from '../types'
import {
  DEFAULT_CHART_INDICATORS,
  SAMPLE_TRADE_INDICATORS,
  barsHaveUsableVolume,
  buildIndicatorOverlays,
  buildIndicatorsQuery,
  expectedIndicatorColumns,
  historyOhlcvForIndicators,
  missingIndicatorMessages,
  normalizeChartIndicators,
} from './indicatorSpec'

function bar(partial: Partial<HistoryBar> & { time: number }): HistoryBar {
  return {
    open: 1,
    high: 1.1,
    low: 0.9,
    close: 1.05,
    ...partial,
  }
}

describe('normalizeChartIndicators', () => {
  it('fills missing flags from the all-off default', () => {
    expect(normalizeChartIndicators({ ema20: true })).toEqual({
      ...DEFAULT_CHART_INDICATORS,
      ema20: true,
    })
  })

  it('ignores unknown or non-boolean values', () => {
    expect(normalizeChartIndicators({ ema20: true, rsi14: 'yes' } as never)).toEqual({
      ...DEFAULT_CHART_INDICATORS,
      ema20: true,
    })
  })
})

describe('history query shaping', () => {
  it('builds the sample-trade indicator spec and leaves ohlcv unset', () => {
    expect(buildIndicatorsQuery(SAMPLE_TRADE_INDICATORS)).toBe(
      'EMA(20), EMA(50), RSI(14), MACD(12,26,9)'
    )
    expect(historyOhlcvForIndicators(SAMPLE_TRADE_INDICATORS)).toBeUndefined()
  })

  it('requests ohlcv only when the volume pane is on', () => {
    expect(historyOhlcvForIndicators({ ...DEFAULT_CHART_INDICATORS, volume: true })).toBe('ohlcv')
    expect(buildIndicatorsQuery({ ...DEFAULT_CHART_INDICATORS, volume: true })).toBeUndefined()
  })

  it('returns no spec when every server-side indicator is off', () => {
    expect(buildIndicatorsQuery(DEFAULT_CHART_INDICATORS)).toBeUndefined()
    expect(expectedIndicatorColumns(DEFAULT_CHART_INDICATORS)).toEqual([])
  })
})

describe('missingIndicatorMessages', () => {
  const bars = [bar({ time: 1, rsi_14: 55 })]

  it('reports requested columns that the payload omitted', () => {
    const messages = missingIndicatorMessages(
      { ...DEFAULT_CHART_INDICATORS, ema20: true, rsi14: true },
      ['rsi_14'],
      bars
    )
    expect(messages.some((message) => /EMA 20/i.test(message))).toBe(true)
    expect(messages.some((message) => /RSI 14/i.test(message))).toBe(false)
  })

  it('warns when volume is on but every bar is empty', () => {
    const messages = missingIndicatorMessages(
      { ...DEFAULT_CHART_INDICATORS, volume: true },
      [],
      [bar({ time: 1, tick_volume: 0 })]
    )
    expect(messages.join(' ')).toMatch(/no usable tick volume/i)
  })

  it('stays quiet before bars arrive', () => {
    expect(
      missingIndicatorMessages(SAMPLE_TRADE_INDICATORS, undefined, [])
    ).toEqual([])
  })
})

describe('buildIndicatorOverlays', () => {
  it('maps EMA to the price pane and RSI to its own pane', () => {
    const overlays = buildIndicatorOverlays(
      [
        bar({ time: 10, ema_20: 1.2, rsi_14: 40 }),
        bar({ time: 20, ema_20: 1.3, rsi_14: 62 }),
      ],
      { ...DEFAULT_CHART_INDICATORS, ema20: true, rsi14: true }
    )
    const ema = overlays.find((overlay) => overlay.name === 'indicator:ema_20')
    const rsi = overlays.find((overlay) => overlay.name === 'indicator:rsi_14')
    expect(ema?.pane).toBe('price')
    expect(ema?.points).toHaveLength(2)
    expect(rsi?.pane).toBe('rsi')
    expect(rsi?.referenceLines?.map((line) => line.price)).toEqual([70, 30])
  })

  it('skips non-finite warmup gaps and colors MACD histogram by sign', () => {
    const overlays = buildIndicatorOverlays(
      [
        bar({ time: 1, macd_12_26_9: Number.NaN }),
        bar({
          time: 2,
          macd_12_26_9: 0.2,
          macd_h_12_26_9: 0.05,
          macd_s_12_26_9: 0.1,
        }),
        bar({
          time: 3,
          macd_12_26_9: -0.1,
          macd_h_12_26_9: -0.02,
          macd_s_12_26_9: 0.0,
        }),
      ],
      { ...DEFAULT_CHART_INDICATORS, macd: true }
    )
    const hist = overlays.find((overlay) => overlay.name === 'indicator:macd_h')
    expect(hist?.kind).toBe('histogram')
    expect(hist?.points.map((point) => point.color)).toEqual(['#22c55e', '#ef4444'])
    expect(overlays.find((overlay) => overlay.name === 'indicator:macd')?.points).toHaveLength(2)
  })

  it('omits the volume series when no bar has usable volume', () => {
    const overlays = buildIndicatorOverlays(
      [bar({ time: 1, tick_volume: 0 })],
      { ...DEFAULT_CHART_INDICATORS, volume: true }
    )
    expect(overlays.find((overlay) => overlay.name === 'indicator:volume')).toBeUndefined()
    expect(barsHaveUsableVolume([bar({ time: 1, tick_volume: 12 })])).toBe(true)
  })

  it('renders a selected post-indicator denoise column and labels it as filtered', () => {
    const overlays = buildIndicatorOverlays(
      [
        bar({ time: 1, rsi_14: 40, rsi_14_dn: 44 }),
        bar({ time: 2, rsi_14: 60, rsi_14_dn: 55 }),
      ],
      { ...DEFAULT_CHART_INDICATORS, rsi14: true },
      {
        spec: { method: 'ema', columns: ['rsi_14'], when: 'post_ti', keep_original: true },
        status: 'applied',
      }
    )
    const rsi = overlays.find((overlay) => overlay.name === 'indicator:rsi_14')
    expect(rsi?.points.map((point) => point.value)).toEqual([44, 55])
    expect(rsi?.label).toBe('RSI 14 · filtered')
  })

  it('uses raw indicator values when the requested denoise was skipped', () => {
    const overlays = buildIndicatorOverlays(
      [bar({ time: 1, ema_20: 1.2, ema_20_dn: 9.9 })],
      { ...DEFAULT_CHART_INDICATORS, ema20: true },
      {
        spec: { method: 'ema', columns: 'ema_20', keep_original: true },
        status: 'skipped',
      }
    )
    const ema = overlays.find((overlay) => overlay.name === 'indicator:ema_20')
    expect(ema?.points[0].value).toBe(1.2)
    expect(ema?.label).toBe('EMA 20')
  })
})
