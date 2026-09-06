import type { RadarRow } from '../types'
import { normalizeSymbol } from './watchlist'

export function radarDisplayPrice(
  row?: Pick<RadarRow, 'mid' | 'last' | 'bar_close'> | null,
): number | undefined {
  return row?.mid ?? row?.last ?? row?.bar_close
}

export function radarQuoteUnusable(
  row?: Pick<RadarRow, 'quote_not_live_ready' | 'quote_usable_for_live_trading'> | null,
): boolean {
  return (
    row?.quote_not_live_ready === true
    || row?.quote_usable_for_live_trading === false
  )
}

export function radarMissingSymbolSet(symbols: string[] | null | undefined): Set<string> {
  return new Set(
    (symbols ?? [])
      .map(normalizeSymbol)
      .filter(Boolean)
  )
}
