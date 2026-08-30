import { describe, expect, it } from 'vitest'
import type { HistoryBar } from '../types'
import { chartQueryActivity, mergeHistoryBars } from './historyBars'

function bar(time: number, close = time): HistoryBar {
  return { time, open: close, high: close, low: close, close }
}

describe('mergeHistoryBars', () => {
  it('accumulates completed bars across moving two-row live windows', () => {
    const base = [bar(0)]
    let live: HistoryBar[] = []
    live = mergeHistoryBars(live, [bar(0), bar(1)])
    live = mergeHistoryBars(live, [bar(1), bar(2)])
    live = mergeHistoryBars(live, [bar(2), bar(3)])

    expect(mergeHistoryBars(base, live).map((item) => item.time)).toEqual([0, 1, 2, 3])
  })

  it('keeps the latest version of a forming candle and sorts timestamps', () => {
    const merged = mergeHistoryBars(
      [bar(2, 20), bar(1, 10)],
      [bar(2, 21), bar(3, 30)]
    )
    expect(merged.map((item) => [item.time, item.close])).toEqual([
      [1, 10],
      [2, 21],
      [3, 30],
    ])
  })

  it('deduplicates sub-second transport noise for the same candle open', () => {
    const merged = mergeHistoryBars([bar(10, 1)], [bar(10.00001, 2)])
    expect(merged).toHaveLength(1)
    expect(merged[0].close).toBe(2)
  })
})

describe('chartQueryActivity', () => {
  it('keeps history active but stops live history and tick polling when Live is off', () => {
    expect(chartQueryActivity('EURUSD', false)).toEqual({
      history: true,
      liveHistory: false,
      tick: false,
    })
    expect(chartQueryActivity('EURUSD', true)).toEqual({
      history: true,
      liveHistory: true,
      tick: true,
    })
  })
})
