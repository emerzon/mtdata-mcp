import type { HistoryBar } from '../types'

/** Merge complete and live candle windows without losing timestamps from older polls. */
export function mergeHistoryBars(...windows: ReadonlyArray<readonly HistoryBar[]>): HistoryBar[] {
  const byTime = new Map<number, HistoryBar>()
  for (const window of windows) {
    for (const bar of window) {
      if (!Number.isFinite(bar.time)) continue
      // Candle opens are second-granularity. Normalize harmless float transport noise
      // so a live update replaces its base row instead of drawing a duplicate candle.
      byTime.set(Math.round(bar.time), bar)
    }
  }
  return Array.from(byTime.values()).sort((left, right) => left.time - right.time)
}

export type ChartQueryActivity = {
  history: boolean
  liveHistory: boolean
  tick: boolean
}

/** Live controls polling only; complete history remains available for every symbol. */
export function chartQueryActivity(symbol: string, isLive: boolean): ChartQueryActivity {
  const hasSymbol = Boolean(symbol.trim())
  return {
    history: hasSymbol,
    liveHistory: hasSymbol && isLive,
    tick: hasSymbol && isLive,
  }
}
