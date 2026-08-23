import { useEffect, useMemo, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { getErrorMessage, getRadar, getSessionStrip } from '../api/client'
import { radarPanelPlacementClass, type LayoutBreakpoint } from '../lib/layout'
import { useEscapeKey } from '../lib/useEscapeKey'
import { loadJSON, saveJSON } from '../lib/storage'
import {
  WATCHLIST_MAX,
  WATCHLIST_STORAGE_KEY,
  addWatchlistSymbol,
  moveWatchlistSymbol,
  normalizeWatchlist,
  removeWatchlistSymbol,
} from '../lib/watchlist'
import type { RadarRow } from '../types'

type Props = {
  open: boolean
  onClose: () => void
  symbol: string
  timeframe: string
  onSelectSymbol: (symbol: string) => void
  onComposeIdea: (symbol: string) => void
  layoutBreakpoint?: LayoutBreakpoint
}

function formatChange(value: number | undefined): string {
  if (typeof value !== 'number' || !Number.isFinite(value)) return '—'
  const sign = value > 0 ? '+' : ''
  return `${sign}${value.toFixed(2)}%`
}

function formatPrice(value: number | undefined): string {
  if (typeof value !== 'number' || !Number.isFinite(value)) return '—'
  return String(value)
}

export function RadarPanel({
  open,
  onClose,
  symbol,
  timeframe,
  onSelectSymbol,
  onComposeIdea,
  layoutBreakpoint = 'desktop',
}: Props) {
  const [watchlist, setWatchlist] = useState<string[]>(() =>
    normalizeWatchlist(loadJSON<string[]>(WATCHLIST_STORAGE_KEY))
  )
  const [draft, setDraft] = useState('')
  useEscapeKey(open, onClose)

  const persist = (next: string[]) => {
    setWatchlist(next)
    saveJSON(WATCHLIST_STORAGE_KEY, next)
  }

  const symbolsKey = watchlist.join(',')
  const radarQuery = useQuery({
    queryKey: ['radar', symbolsKey || 'seed', timeframe],
    queryFn: () =>
      getRadar({
        symbols: watchlist.length > 0 ? symbolsKey : undefined,
        timeframe,
        rank_by: 'watchlist',
        limit: WATCHLIST_MAX,
      }),
    enabled: open,
    refetchInterval: 20_000,
  })

  useEffect(() => {
    if (watchlist.length > 0 || !radarQuery.data?.rows?.length) return
    const symbols = radarQuery.data.rows.map((row) => row.symbol).filter(Boolean)
    const last = symbol || loadJSON<string>('last_symbol') || undefined
    persist(normalizeWatchlist(last ? [last, ...symbols] : symbols))
  }, [radarQuery.data, symbol, watchlist.length])
  const sessionQuery = useQuery({
    queryKey: ['session-strip', symbol],
    queryFn: () => getSessionStrip(symbol || undefined),
    enabled: open,
    refetchInterval: 60_000,
  })

  const rowsBySymbol = useMemo(() => {
    const map = new Map<string, RadarRow>()
    for (const row of radarQuery.data?.rows ?? []) {
      map.set(row.symbol, row)
    }
    return map
  }, [radarQuery.data?.rows])

  if (!open) return null

  const panelClass = radarPanelPlacementClass(layoutBreakpoint)
  const session = sessionQuery.data

  return (
    <>
      {layoutBreakpoint === 'mobile' && (
        <button
          type="button"
          className="fixed inset-0 z-20 bg-slate-950/50 backdrop-blur-[1px]"
          aria-label="Dismiss watchlist"
          onClick={onClose}
        />
      )}
      <div className={`${panelClass} animate-slide-in-right`} role="dialog" aria-modal="true" aria-label="Watchlist radar">
        <div className="flex items-center justify-between px-4 py-3 border-b border-slate-800 shrink-0">
          <div>
            <h2 className="text-sm font-medium text-slate-100">Watchlist</h2>
            <p className="text-[11px] text-slate-500">Activity on your list, not opportunity.</p>
          </div>
          <button className="text-slate-400 hover:text-slate-200 p-2 min-h-9 min-w-9" onClick={onClose} aria-label="Close watchlist">
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        <div className="flex-1 overflow-y-auto overscroll-contain min-h-0">
          <div className="px-4 py-3 border-b border-slate-800 space-y-1 text-[11px] text-slate-400">
            {session?.account && (
              <div>
                {session.account.server ?? 'Account'}
                {session.account.equity != null && (
                  <span className="ml-2 text-slate-200">
                    {session.account.equity}
                    {session.account.currency ? ` ${session.account.currency}` : ''}
                  </span>
                )}
                {session.exposure_count != null && (
                  <span className="ml-2">open {session.exposure_count}</span>
                )}
              </div>
            )}
            {session?.market_status?.status && (
              <div>Session {session.market_status.status}</div>
            )}
            {session?.account_error && <div>Account unavailable</div>}
            {session?.news?.slice(0, 3).map((item) => (
              <div key={item.title} className="truncate text-slate-500">
                {item.title}
              </div>
            ))}
          </div>

          <form
            className="flex gap-2 px-4 py-3 border-b border-slate-800"
            onSubmit={(event) => {
              event.preventDefault()
              if (!draft.trim()) return
              persist(addWatchlistSymbol(watchlist, draft))
              setDraft('')
            }}
          >
            <input
              className="flex-1 bg-slate-800 text-slate-200 text-xs rounded px-2 py-1.5 border border-slate-700"
              value={draft}
              placeholder="Add symbol"
              onChange={(event) => setDraft(event.target.value)}
            />
            <button type="submit" className="text-xs px-2 py-1 rounded border border-slate-700 text-slate-300">
              Add
            </button>
          </form>

          {radarQuery.error && (
            <div className="px-4 py-2 text-xs text-rose-300">{getErrorMessage(radarQuery.error)}</div>
          )}

          <ul className="divide-y divide-slate-800">
            {watchlist.map((name, index) => {
              const row = rowsBySymbol.get(name)
              const change = row?.live_price_change_pct ?? row?.price_change_pct
              const unusable = row?.quote_not_live_ready === true || row?.usable_for_live_trading === false
              const active = name === symbol
              return (
                <li key={name} className={`px-4 py-2 ${active ? 'bg-slate-800/70' : ''}`}>
                  <div className="flex items-start justify-between gap-2">
                    <button
                      type="button"
                      className="text-left min-w-0"
                      onClick={() => onSelectSymbol(name)}
                    >
                      <div className="text-sm text-slate-100">{name}</div>
                      <div className="text-[11px] text-slate-500">
                        {formatPrice(row?.mid ?? row?.last ?? row?.close)}
                        {row?.spread_pips != null && <span className="ml-2">spr {row.spread_pips}</span>}
                        <span className={`ml-2 ${typeof change === 'number' && change < 0 ? 'text-rose-300' : 'text-emerald-300'}`}>
                          {formatChange(change)}
                        </span>
                      </div>
                      {unusable && <div className="text-[11px] text-amber-400">quote unusable</div>}
                    </button>
                    <div className="flex flex-col items-end gap-1 shrink-0">
                      <button
                        type="button"
                        className="text-[11px] text-sky-300 hover:text-sky-200"
                        onClick={() => onComposeIdea(name)}
                      >
                        Compose
                      </button>
                      <div className="flex gap-1">
                        <button
                          type="button"
                          className="text-[11px] text-slate-500"
                          disabled={index === 0}
                          onClick={() => persist(moveWatchlistSymbol(watchlist, name, -1))}
                        >
                          Up
                        </button>
                        <button
                          type="button"
                          className="text-[11px] text-slate-500"
                          disabled={index === watchlist.length - 1}
                          onClick={() => persist(moveWatchlistSymbol(watchlist, name, 1))}
                        >
                          Down
                        </button>
                        <button
                          type="button"
                          className="text-[11px] text-slate-500"
                          onClick={() => persist(removeWatchlistSymbol(watchlist, name))}
                        >
                          Remove
                        </button>
                      </div>
                    </div>
                  </div>
                </li>
              )
            })}
          </ul>
          {watchlist.length === 0 && (
            <div className="px-4 py-6 text-xs text-slate-500">Add a broker symbol to start a list.</div>
          )}
        </div>
      </div>
    </>
  )
}
