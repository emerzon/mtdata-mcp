import { describe, expect, it } from 'vitest'
import {
  addWatchlistSymbol,
  moveWatchlistSymbol,
  normalizeWatchlist,
  removeWatchlistSymbol,
} from './watchlist'

describe('watchlist helpers', () => {
  it('normalizes, dedupes, and caps names', () => {
    expect(normalizeWatchlist(['eurusd', 'EURUSD', ' gbpusd ', '', 'XAUUSD'], 2)).toEqual([
      'EURUSD',
      'GBPUSD',
    ])
  })

  it('adds, removes, and reorders', () => {
    const added = addWatchlistSymbol(['EURUSD'], 'gbpusd')
    expect(added).toEqual(['EURUSD', 'GBPUSD'])
    expect(removeWatchlistSymbol(added, 'EURUSD')).toEqual(['GBPUSD'])
    expect(moveWatchlistSymbol(['EURUSD', 'GBPUSD', 'USDJPY'], 'USDJPY', -1)).toEqual([
      'EURUSD',
      'USDJPY',
      'GBPUSD',
    ])
  })
})
