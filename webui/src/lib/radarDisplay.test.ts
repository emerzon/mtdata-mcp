import { describe, expect, it } from 'vitest'
import { radarDisplayPrice, radarQuoteUnusable } from './radarDisplay'

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
