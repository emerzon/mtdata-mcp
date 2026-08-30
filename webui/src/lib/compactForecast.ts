/**
 * Compact dedicated-panel forecast / backtest adapters.
 * Focused screens always receive the backend compact payload.
 */

import { toUtcSec } from './time'
import type {
  AnchorMetrics,
  BacktestMethodResult,
  BacktestResult,
  CompactForecastRow,
  ForecastPayload,
  OutputWarningLike,
} from '../types'

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

/** Match the backend price metric: errors over the overlap, direction at the terminal horizon. */
export function computeAnchorForecastMetrics(
  predictions: number[],
  actuals: number[],
  baselinePrice: number
): AnchorMetrics | null {
  if (!Number.isFinite(baselinePrice)) return null
  const width = Math.min(predictions.length, actuals.length)
  const pairs: Array<{ prediction: number; actual: number }> = []
  for (let index = 0; index < width; index += 1) {
    const prediction = predictions[index]
    const actual = actuals[index]
    if (Number.isFinite(prediction) && Number.isFinite(actual)) pairs.push({ prediction, actual })
  }
  if (!pairs.length) return null

  const diffs = pairs.map(({ prediction, actual }) => prediction - actual)
  const overlap = pairs.length
  const mae = diffs.reduce((total, diff) => total + Math.abs(diff), 0) / overlap
  const mape =
    (pairs.reduce((total, { prediction, actual }) => {
      const denominator = Math.abs(actual) || 1
      return total + Math.abs((prediction - actual) / denominator)
    }, 0) /
      overlap) *
    100
  const rmse = Math.sqrt(diffs.reduce((total, diff) => total + diff * diff, 0) / overlap)
  const terminal = pairs[pairs.length - 1]
  const forecastDirection = Math.sign(terminal.prediction - baselinePrice)
  const actualDirection = Math.sign(terminal.actual - baselinePrice)
  const dirAcc = forecastDirection === 0 ? null : forecastDirection === actualDirection ? 100 : 0
  return { overlap, mae, mape, rmse, dirAcc }
}

export function outputWarningMessage(warning: OutputWarningLike): string {
  return typeof warning === 'string' ? warning : warning.message || warning.code
}

function humanizeStatus(value: string): string {
  return value.replace(/[_-]+/g, ' ').replace(/\s+/g, ' ').trim()
}

export type ResultFeedback = {
  tone: 'success' | 'warning' | 'error'
  summary: string
  details: string[]
}

/** Preserve successful forecast reliability disclosures alongside the rendered path. */
export function forecastResultFeedback(result: ForecastPayload): ResultFeedback {
  const forecastStatus = String(result.forecast_status || 'ready').toLowerCase()
  const trustLevel = String(result.trust_level || 'unknown').toLowerCase()
  const signalStatus = String(result.signal_status || 'unknown').toLowerCase()
  const ciStatus = String(result.ci_status || 'unknown').toLowerCase()
  const forecastMode = String(result.forecast_mode || 'unknown').toLowerCase()
  const statuses = [
    `Forecast: ${humanizeStatus(forecastStatus)}`,
    `Trust: ${humanizeStatus(trustLevel)}`,
    `Signal: ${humanizeStatus(signalStatus)}`,
    `Intervals: ${humanizeStatus(ciStatus)}`,
    `Mode: ${humanizeStatus(forecastMode)}`,
  ]
  const details = (result.warnings ?? []).map(outputWarningMessage)
  for (const blocker of result.trust_blockers ?? []) {
    details.push(`Trust blocker: ${humanizeStatus(blocker)}`)
  }
  if (result.uncertainty?.reason && ciStatus !== 'available') {
    details.push(result.uncertainty.reason)
  }
  const uniqueDetails = [...new Set(details.filter(Boolean))]
  const degraded =
    !['ready', 'complete', 'informative'].includes(forecastStatus) ||
    ['low', 'degraded', 'unusable'].includes(trustLevel) ||
    signalStatus === 'not_actionable' ||
    forecastMode === 'point_only' ||
    !['available', 'not_requested', 'unknown'].includes(ciStatus) ||
    uniqueDetails.length > 0
  return {
    tone: degraded ? 'warning' : 'success',
    summary: statuses.join(' · '),
    details: uniqueDetails,
  }
}

export type BacktestDisplayRow = BacktestMethodResult & { method: string }

export function backtestMethodStatus(result: BacktestMethodResult): 'complete' | 'partial' | 'failed' {
  const status = String(result.status || '').toLowerCase()
  if (status === 'complete' || status === 'partial' || status === 'failed') return status
  if (result.success === false) return 'failed'
  if (result.complete_success === false || (result.failed_tests ?? 0) > 0) return 'partial'
  return 'complete'
}

export function backtestResultFeedback(result: BacktestResult): ResultFeedback {
  const status = String(
    result.status || (result.complete_success === false ? 'partial' : 'complete')
  ).toLowerCase()
  const details: string[] = []
  if (result.methods_partial) details.push(`${result.methods_partial} method(s) completed only partially.`)
  if (result.methods_failed) details.push(`${result.methods_failed} method(s) failed.`)
  if (result.anchor_tests_planned != null) {
    const succeeded = result.anchor_tests_succeeded ?? 0
    const failed = result.anchor_tests_failed ?? Math.max(0, result.anchor_tests_planned - succeeded)
    details.push(`${succeeded}/${result.anchor_tests_planned} anchor tests succeeded; ${failed} failed.`)
  }
  const warningMessages = (result.warnings ?? []).map(outputWarningMessage)
  details.push(...warningMessages)
  const hasIncompleteEvidence =
    status !== 'complete' ||
    (result.methods_partial ?? 0) > 0 ||
    (result.methods_failed ?? 0) > 0 ||
    (result.anchor_tests_failed ?? 0) > 0 ||
    warningMessages.length > 0
  return {
    tone: status === 'failed' ? 'error' : hasIncompleteEvidence ? 'warning' : 'success',
    summary: `Backtest: ${humanizeStatus(status)}`,
    details: [...new Set(details.filter(Boolean))],
  }
}

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
      if (full) rows.push({ ...item, ...full, method })
    }
    for (const [method, item] of Object.entries(results)) {
      if (seen.has(method)) continue
      rows.push({ method, ...item })
    }
    return rows
  }
  return Object.entries(results).map(([method, item]) => ({ method, ...item }))
}
