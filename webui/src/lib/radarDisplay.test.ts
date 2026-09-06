import { describe, expect, it } from 'vitest'
import { radarDisplayPrice, radarMissingSymbolSet, radarQuoteUnusable } from './radarDisplay'

describe('radarDisplayPrice', () => {
  it('falls back to bar_close when mid and last are absent', () => {
    expect(radarDisplayPrice({ bar_close: 1.08 })).toBe(1.08)
  })

  it('prefers mid, then last, then bar_close', () => {
    expect(radarDisplayPrice({ mid: 1.1, last: 1.09, bar_close: 1.08 })).toBe(1.1)
    expect(radarDisplayPrice({ last: 1.09, bar_close: 1.08 })).toBe(1.09)
  })
})

describe('radarQuoteUnusable', () => {
  it('treats quote_not_live_ready as unusable', () => {
    expect(radarQuoteUnusable({ quote_not_live_ready: true })).toBe(true)
  })
})

describe('radarMissingSymbolSet', () => {
  it('normalizes authoritative missing broker names', () => {
    expect(radarMissingSymbolSet([' nope ', 'eur/usd'])).toEqual(new Set(['NOPE', 'EUR/USD']))
  })

  it('ignores blank symbols and coalesces differently cased duplicates', () => {
    expect(radarMissingSymbolSet([' eurusd ', 'EURUSD', ' '])).toEqual(new Set(['EURUSD']))
    expect(radarMissingSymbolSet(undefined)).toEqual(new Set())
    expect(radarMissingSymbolSet(null)).toEqual(new Set())
  })
})
