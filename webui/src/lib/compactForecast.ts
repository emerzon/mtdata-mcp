/**
 * Compact dedicated-panel forecast / backtest adapters.
 * Focused screens always receive the backend compact payload.
 */

import { toUtcSec } from './time'
import type { BacktestMethodResult, BacktestResult, CompactForecastRow } from '../types'

export type CompactForecastSeries = {
  times: number[]
  values: number[]
  lower?: number[]
  upper?: number[]
}

export function chartPriceFromCompactRow(row: CompactForecastRow): number | undefined {
  const price = row.price ?? row.value
  return typeof price === 'number' && Number.isFinite(price) ? price : undefined
}

export function rowIntervalBounds(row: CompactForecastRow): { lower: number; upper: number } | undefined {
  const lower = row.lower_price ?? row.lower
  const upper = row.upper_price ?? row.upper
  if (
    typeof lower !== 'number' ||
    !Number.isFinite(lower) ||
    typeof upper !== 'number' ||
    !Number.isFinite(upper)
  ) {
    return undefined
  }
  return { lower, upper }
}

/**
 * Map compact `forecast` rows to chart series.
 * Throws when a row has no finite chart price (`price ?? value`).
 */
export function mapCompactForecastToSeries(rows: CompactForecastRow[]): CompactForecastSeries {
  const times: number[] = []
  const values: number[] = []
  const lower: number[] = []
  const upper: number[] = []

  for (const row of rows) {
    const price = chartPriceFromCompactRow(row)
    if (price === undefined) {
      throw new Error('compact forecast row has no finite chart price')
    }
    if (row.time == null || row.time === '') {
      throw new Error('compact forecast row is missing time')
    }
    times.push(toUtcSec(row.time))
    values.push(price)
    const bounds = rowIntervalBounds(row)
    if (bounds) {
      lower.push(bounds.lower)
      upper.push(bounds.upper)
    }
  }

  const hasBounds = lower.length === values.length && upper.length === values.length && values.length > 0
  return hasBounds ? { times, values, lower, upper } : { times, values }
}

export type BacktestDisplayRow = BacktestMethodResult & { method: string }

/**
 * Display rows from compact backtest `results`, ordered by `ranked_methods` when present.
 */
export function backtestDisplayRows(result: BacktestResult | null | undefined): BacktestDisplayRow[] {
  if (!result) return []
  const results = result.results ?? {}
  const ranked = result.ranked_methods ?? []
  if (ranked.length) {
    const rows: BacktestDisplayRow[] = []
    const seen = new Set<string>()
    for (const item of ranked) {
      const method = item.method
      if (!method || seen.has(method)) continue
      seen.add(method)
      const full = results[method]
      if (full) rows.push({ method, ...full })
    }
    for (const [method, item] of Object.entries(results)) {
      if (seen.has(method)) continue
      rows.push({ method, ...item })
    }
    return rows
  }
  return Object.entries(results).map(([method, item]) => ({ method, ...item }))
}
