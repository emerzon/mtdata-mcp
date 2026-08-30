/**
 * Chart-workspace indicator presets, history query shape, and overlay mapping.
 * Spec strings must match the data_fetch_candles / GET /history `indicators` DSL.
 */

import type { ChartOverlay, DenoiseSpecUI, HistoryBar } from '../types'

export const CHART_INDICATOR_IDS = ['ema20', 'ema50', 'rsi14', 'macd', 'volume'] as const
export type ChartIndicatorId = (typeof CHART_INDICATOR_IDS)[number]

export type ChartIndicatorSelection = Record<ChartIndicatorId, boolean>

export const DEFAULT_CHART_INDICATORS: ChartIndicatorSelection = {
  ema20: false,
  ema50: false,
  rsi14: false,
  macd: false,
  volume: false,
}

/** Same set as SAMPLE-TRADE.md step 1 (volume stays optional). */
export const SAMPLE_TRADE_INDICATORS: ChartIndicatorSelection = {
  ema20: true,
  ema50: true,
  rsi14: true,
  macd: true,
  volume: false,
}

const SPEC_PARTS: Record<Exclude<ChartIndicatorId, 'volume'>, string> = {
  ema20: 'EMA(20)',
  ema50: 'EMA(50)',
  rsi14: 'RSI(14)',
  macd: 'MACD(12,26,9)',
}

const COLUMN_LABELS: Record<string, string> = {
  ema_20: 'EMA 20',
  ema_50: 'EMA 50',
  rsi_14: 'RSI 14',
  macd_12_26_9: 'MACD',
}

export function normalizeChartIndicators(
  partial?: Partial<ChartIndicatorSelection> | null
): ChartIndicatorSelection {
  const next = { ...DEFAULT_CHART_INDICATORS }
  if (!partial || typeof partial !== 'object') return next
  for (const id of CHART_INDICATOR_IDS) {
    if (typeof partial[id] === 'boolean') next[id] = partial[id]
  }
  return next
}

export function chartIndicatorsActive(selection: ChartIndicatorSelection): boolean {
  return CHART_INDICATOR_IDS.some((id) => selection[id])
}

export function buildIndicatorsQuery(selection: ChartIndicatorSelection): string | undefined {
  const parts: string[] = []
  if (selection.ema20) parts.push(SPEC_PARTS.ema20)
  if (selection.ema50) parts.push(SPEC_PARTS.ema50)
  if (selection.rsi14) parts.push(SPEC_PARTS.rsi14)
  if (selection.macd) parts.push(SPEC_PARTS.macd)
  return parts.length ? parts.join(', ') : undefined
}

/** Request tick volume only when the volume pane is on. */
export function historyOhlcvForIndicators(selection: ChartIndicatorSelection): string | undefined {
  return selection.volume ? 'ohlcv' : undefined
}

export function expectedIndicatorColumns(selection: ChartIndicatorSelection): string[] {
  const columns: string[] = []
  if (selection.ema20) columns.push('ema_20')
  if (selection.ema50) columns.push('ema_50')
  if (selection.rsi14) columns.push('rsi_14')
  if (selection.macd) columns.push('macd_12_26_9', 'macd_h_12_26_9', 'macd_s_12_26_9')
  return columns
}

export function numericBarField(bar: HistoryBar, key: string): number | undefined {
  const value = (bar as unknown as Record<string, unknown>)[key]
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined
}

export function volumeField(bar: HistoryBar): number | undefined {
  const tick = numericBarField(bar, 'tick_volume')
  if (tick !== undefined) return tick
  return numericBarField(bar, 'volume')
}

export function barsHaveUsableVolume(bars: HistoryBar[]): boolean {
  return bars.some((bar) => {
    const value = volumeField(bar)
    return value !== undefined && value > 0
  })
}

function presentColumnSet(columns: string[] | undefined, bars: HistoryBar[]): Set<string> {
  const present = new Set((columns ?? []).map((column) => column.toLowerCase()))
  if (present.size === 0 && bars.length) {
    const sample = bars[bars.length - 1] as unknown as Record<string, unknown>
    for (const key of Object.keys(sample)) present.add(key.toLowerCase())
  }
  return present
}

export function missingIndicatorMessages(
  selection: ChartIndicatorSelection,
  columns: string[] | undefined,
  bars: HistoryBar[]
): string[] {
  if (!chartIndicatorsActive(selection) || !bars.length) return []

  const present = presentColumnSet(columns, bars)
  const messages: string[] = []
  for (const column of expectedIndicatorColumns(selection)) {
    if (column === 'macd_h_12_26_9' || column === 'macd_s_12_26_9') continue
    if (!present.has(column)) {
      messages.push(`Indicators: ${COLUMN_LABELS[column] ?? column} was requested but not returned.`)
    }
  }
  if (selection.volume && !barsHaveUsableVolume(bars)) {
    messages.push(
      'Volume pane is on, but this symbol has no usable tick volume. Forex volume is typically indicative only.'
    )
  }
  return messages
}

function linePoints(bars: HistoryBar[], column: string): ChartOverlay['points'] {
  const points: ChartOverlay['points'] = []
  for (const bar of bars) {
    const value = numericBarField(bar, column)
    if (!Number.isFinite(bar.time) || value === undefined) continue
    points.push({ time: bar.time, value })
  }
  return points
}

export type IndicatorDenoiseContext = {
  spec?: DenoiseSpecUI
  status?: string
}

type ResolvedIndicatorColumn = {
  column: string
  filtered: boolean
}

function requestedDenoiseColumns(spec: DenoiseSpecUI | undefined): Set<string> {
  const values = Array.isArray(spec?.columns) ? spec.columns : [spec?.columns ?? 'close']
  return new Set(
    values
      .flatMap((value) => String(value).split(','))
      .map((value) => value.trim().toLowerCase())
      .filter(Boolean)
  )
}

function resolveIndicatorColumn(
  bars: HistoryBar[],
  column: string,
  denoise: IndicatorDenoiseContext | undefined
): ResolvedIndicatorColumn {
  const status = String(denoise?.status || '').trim().toLowerCase()
  const selected = requestedDenoiseColumns(denoise?.spec).has(column.toLowerCase())
  if (!selected || status !== 'applied') return { column, filtered: false }

  if (denoise?.spec?.keep_original === false) {
    return { column, filtered: true }
  }

  const suffixed = `${column}_dn`
  if (bars.some((bar) => numericBarField(bar, suffixed) !== undefined)) {
    return { column: suffixed, filtered: true }
  }
  return { column, filtered: false }
}

function filteredLabel(label: string, resolved: ResolvedIndicatorColumn): string {
  return resolved.filtered ? `${label} · filtered` : label
}

export function buildIndicatorOverlays(
  bars: HistoryBar[],
  selection: ChartIndicatorSelection,
  denoise?: IndicatorDenoiseContext
): ChartOverlay[] {
  if (!bars.length) return []

  const overlays: ChartOverlay[] = []

  if (selection.ema20) {
    const resolved = resolveIndicatorColumn(bars, 'ema_20', denoise)
    const points = linePoints(bars, resolved.column)
    if (points.length) {
      overlays.push({
        name: 'indicator:ema_20',
        points,
        color: '#38bdf8',
        lineWidth: 2,
        pane: 'price',
        label: filteredLabel('EMA 20', resolved),
      })
    }
  }

  if (selection.ema50) {
    const resolved = resolveIndicatorColumn(bars, 'ema_50', denoise)
    const points = linePoints(bars, resolved.column)
    if (points.length) {
      overlays.push({
        name: 'indicator:ema_50',
        points,
        color: '#a78bfa',
        lineWidth: 2,
        pane: 'price',
        label: filteredLabel('EMA 50', resolved),
      })
    }
  }

  if (selection.rsi14) {
    const resolved = resolveIndicatorColumn(bars, 'rsi_14', denoise)
    const points = linePoints(bars, resolved.column)
    if (points.length) {
      overlays.push({
        name: 'indicator:rsi_14',
        points,
        color: '#f472b6',
        lineWidth: 2,
        pane: 'rsi',
        label: filteredLabel('RSI 14', resolved),
        referenceLines: [
          { price: 70, color: '#64748b', title: '70' },
          { price: 30, color: '#64748b', title: '30' },
        ],
      })
    }
  }

  if (selection.macd) {
    const macdResolved = resolveIndicatorColumn(bars, 'macd_12_26_9', denoise)
    const signalResolved = resolveIndicatorColumn(bars, 'macd_s_12_26_9', denoise)
    const histogramResolved = resolveIndicatorColumn(bars, 'macd_h_12_26_9', denoise)
    const macdPoints = linePoints(bars, macdResolved.column)
    const signalPoints = linePoints(bars, signalResolved.column)
    const histPoints: ChartOverlay['points'] = []
    for (const bar of bars) {
      const value = numericBarField(bar, histogramResolved.column)
      if (!Number.isFinite(bar.time) || value === undefined) continue
      histPoints.push({
        time: bar.time,
        value,
        color: value >= 0 ? '#22c55e' : '#ef4444',
      })
    }
    if (histPoints.length) {
      overlays.push({
        name: 'indicator:macd_h',
        points: histPoints,
        color: '#22c55e',
        pane: 'macd',
        kind: 'histogram',
        label: filteredLabel('MACD hist', histogramResolved),
      })
    }
    if (macdPoints.length) {
      overlays.push({
        name: 'indicator:macd',
        points: macdPoints,
        color: '#38bdf8',
        lineWidth: 2,
        pane: 'macd',
        label: filteredLabel('MACD', macdResolved),
        referenceLines: [{ price: 0, color: '#475569', title: '0' }],
      })
    }
    if (signalPoints.length) {
      overlays.push({
        name: 'indicator:macd_s',
        points: signalPoints,
        color: '#f59e0b',
        lineWidth: 1,
        pane: 'macd',
        label: filteredLabel('Signal', signalResolved),
      })
    }
  }

  if (selection.volume && barsHaveUsableVolume(bars)) {
    const volumePoints: ChartOverlay['points'] = []
    for (let index = 0; index < bars.length; index += 1) {
      const bar = bars[index]
      const value = volumeField(bar)
      if (!Number.isFinite(bar.time) || value === undefined) continue
      const previous = index > 0 ? bars[index - 1].close : bar.open
      const up = bar.close >= previous
      volumePoints.push({
        time: bar.time,
        value,
        color: up ? 'rgba(34,197,94,0.55)' : 'rgba(239,68,68,0.55)',
      })
    }
    if (volumePoints.length) {
      overlays.push({
        name: 'indicator:volume',
        points: volumePoints,
        color: '#22c55e',
        pane: 'volume',
        kind: 'histogram',
        label: 'Volume',
      })
    }
  }

  return overlays
}
