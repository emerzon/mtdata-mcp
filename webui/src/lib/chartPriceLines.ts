import type { PriceLineSpec } from '../components/OHLCChart'
import type { Tick } from '../types'

export function liveQuotePriceLines(
  tick: Tick | undefined,
  options: { showBid: boolean; showAsk: boolean; showLast: boolean },
): PriceLineSpec[] {
  if (!tick || tick.usable_for_live_trading !== true) return []

  const lines: PriceLineSpec[] = []
  if (options.showBid && Number.isFinite(tick.bid) && tick.bid > 0) {
    lines.push({ price: tick.bid, color: '#ef4444', title: 'Bid' })
  }
  if (options.showAsk && Number.isFinite(tick.ask) && tick.ask > 0) {
    lines.push({ price: tick.ask, color: '#22c55e', title: 'Ask' })
  }
  if (
    options.showLast
    && typeof tick.last === 'number'
    && Number.isFinite(tick.last)
    && tick.last > 0
  ) {
    lines.push({ price: tick.last, color: '#facc15', title: 'Last' })
  }
  return lines
}
