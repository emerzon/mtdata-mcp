import { describe, expect, it } from 'vitest'
import { liveQuotePriceLines } from './chartPriceLines'

const options = { showBid: false, showAsk: false, showLast: true }
const baseTick = {
  success: true,
  symbol: 'EURUSD',
  time: '2026-08-30T12:00:00Z',
  time_epoch: 1788091200,
  bid: 1.1,
  ask: 1.2,
}

describe('liveQuotePriceLines', () => {
  it('does not invent a Last trade when the feed has none', () => {
    expect(liveQuotePriceLines({ ...baseTick, last: null }, options)).toEqual([])
    expect(liveQuotePriceLines({ ...baseTick, last: 0 }, options)).toEqual([])
  })

  it('renders a positive broker Last trade', () => {
    expect(liveQuotePriceLines({ ...baseTick, last: 1.15 }, options)).toEqual([
      { price: 1.15, color: '#facc15', title: 'Last' },
    ])
  })
})
