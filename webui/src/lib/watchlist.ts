export const WATCHLIST_STORAGE_KEY = 'mtdata.watchlist'
export const WATCHLIST_MAX = 20

export function normalizeSymbol(value: string | null | undefined): string {
  return String(value || '').trim().toUpperCase()
}

export function normalizeWatchlist(value: unknown, max = WATCHLIST_MAX): string[] {
  if (!Array.isArray(value)) return []
  const seen = new Set<string>()
  const symbols: string[] = []
  for (const item of value) {
    const symbol = normalizeSymbol(typeof item === 'string' ? item : '')
    if (!symbol || seen.has(symbol)) continue
    seen.add(symbol)
    symbols.push(symbol)
    if (symbols.length >= max) break
  }
  return symbols
}

export function addWatchlistSymbol(list: string[], symbol: string, max = WATCHLIST_MAX): string[] {
  const next = normalizeWatchlist(list, max)
  const value = normalizeSymbol(symbol)
  if (!value || next.includes(value) || next.length >= max) return next
  return [...next, value]
}

export function removeWatchlistSymbol(list: string[], symbol: string): string[] {
  const value = normalizeSymbol(symbol)
  return normalizeWatchlist(list).filter((item) => item !== value)
}

export function moveWatchlistSymbol(list: string[], symbol: string, direction: -1 | 1): string[] {
  const next = normalizeWatchlist(list)
  const value = normalizeSymbol(symbol)
  const index = next.indexOf(value)
  const target = index + direction
  if (index < 0 || target < 0 || target >= next.length) return next
  const copy = [...next]
  const [row] = copy.splice(index, 1)
  copy.splice(target, 0, row)
  return copy
}
