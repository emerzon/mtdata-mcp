import { useCallback, useEffect, useRef, useState } from 'react'
import { composeTradeIdea, getErrorMessage } from '../api/client'
import { forecastPanelPlacementClass, type LayoutBreakpoint } from '../lib/layout'
import { useEscapeKey } from '../lib/useEscapeKey'
import type { TradeIdeaPayload } from '../types'

type Props = {
  open: boolean
  onClose: () => void
  symbol: string
  timeframe: string
  onIdea: (idea: TradeIdeaPayload | null) => void
  layoutBreakpoint?: LayoutBreakpoint
  autoComposeKey?: number
}

export function IdeaPanel({
  open,
  onClose,
  symbol,
  timeframe,
  onIdea,
  layoutBreakpoint = 'desktop',
  autoComposeKey,
}: Props) {
  const [direction, setDirection] = useState<'auto' | 'long' | 'short'>('auto')
  const [template, setTemplate] = useState<'quick' | 'standard'>('quick')
  const [horizon, setHorizon] = useState(12)
  const [riskPct, setRiskPct] = useState(0.5)
  const [idea, setIdea] = useState<TradeIdeaPayload | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  useEscapeKey(open, onClose)

  const onIdeaRef = useRef(onIdea)
  const requestKey = JSON.stringify({
    symbol,
    timeframe,
    direction,
    template,
    horizon,
    riskPct,
  })
  const requestKeyRef = useRef(requestKey)
  requestKeyRef.current = requestKey
  const runId = useRef(0)

  useEffect(() => {
    onIdeaRef.current = onIdea
  }, [onIdea])

  useEffect(() => {
    runId.current += 1
    setIsLoading(false)
    setError(null)
    setIdea(null)
    onIdeaRef.current(null)
  }, [requestKey])

  const run = useCallback(async () => {
    if (!symbol) return

    const runRequestKey = requestKey
    const currentRunId = ++runId.current
    setIsLoading(true)
    setError(null)
    setIdea(null)
    onIdeaRef.current(null)

    try {
      const result = await composeTradeIdea({
        symbol,
        timeframe,
        horizon,
        direction,
        template,
        risk_pct: riskPct,
      })
      if (currentRunId !== runId.current || runRequestKey !== requestKeyRef.current) return
      setIdea(result)
      onIdeaRef.current(result)
    } catch (err) {
      if (currentRunId === runId.current && runRequestKey === requestKeyRef.current) {
        setIdea(null)
        onIdeaRef.current(null)
        setError(getErrorMessage(err))
      }
    } finally {
      if (currentRunId === runId.current && runRequestKey === requestKeyRef.current) {
        setIsLoading(false)
      }
    }
  }, [direction, horizon, requestKey, riskPct, symbol, template, timeframe])

  useEffect(() => {
    if (!open || !autoComposeKey) return
    void run()
  }, [autoComposeKey, open, run])

  if (!open) return null

  const panelClass = forecastPanelPlacementClass(layoutBreakpoint)

  const gates = Object.entries(idea?.gates ?? {})

  return (
    <>
      {layoutBreakpoint === 'mobile' && (
        <button
          type="button"
          className="fixed inset-0 z-20 bg-slate-950/50 backdrop-blur-[1px]"
          aria-label="Dismiss idea panel"
          onClick={onClose}
        />
      )}
      <div className={`${panelClass} animate-slide-in-right`} role="dialog" aria-modal="true" aria-label="Trade idea panel">
        <div className="flex items-center justify-between px-4 py-3 border-b border-slate-800 shrink-0">
          <div>
            <h2 className="text-sm font-medium text-slate-100">Idea</h2>
            <p className="text-[11px] text-slate-500">Preview-only research. Cannot place an order.</p>
          </div>
          <button className="text-slate-400 hover:text-slate-200 p-2 min-h-9 min-w-9" onClick={onClose} aria-label="Close idea panel">
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        <div className="flex-1 overflow-y-auto overscroll-contain p-4 min-h-0 space-y-3">
          <label className="block text-[11px] text-slate-500">
            Direction
            <select
              className="mt-0.5 w-full bg-slate-800 text-slate-200 text-xs rounded px-2 py-1.5 border border-slate-700"
              value={direction}
              onChange={(event) => setDirection(event.target.value as 'auto' | 'long' | 'short')}
            >
              <option value="auto">auto</option>
              <option value="long">long</option>
              <option value="short">short</option>
            </select>
          </label>
          <label className="block text-[11px] text-slate-500">
            Template
            <select
              className="mt-0.5 w-full bg-slate-800 text-slate-200 text-xs rounded px-2 py-1.5 border border-slate-700"
              value={template}
              onChange={(event) => setTemplate(event.target.value as 'quick' | 'standard')}
            >
              <option value="quick">quick</option>
              <option value="standard">standard (confluence snap)</option>
            </select>
          </label>
          <div className="grid grid-cols-2 gap-2">
            <label className="block text-[11px] text-slate-500">
              Horizon
              <input
                type="number"
                min={1}
                max={500}
                className="mt-0.5 w-full bg-slate-800 text-slate-200 text-xs rounded px-2 py-1.5 border border-slate-700"
                value={horizon}
                onChange={(event) => setHorizon(Number(event.target.value) || 12)}
              />
            </label>
            <label className="block text-[11px] text-slate-500">
              Risk %
              <input
                type="number"
                min={0.01}
                max={100}
                step={0.1}
                className="mt-0.5 w-full bg-slate-800 text-slate-200 text-xs rounded px-2 py-1.5 border border-slate-700"
                value={riskPct}
                onChange={(event) => setRiskPct(Number(event.target.value) || 0.5)}
              />
            </label>
          </div>

          <button
            type="button"
            className="w-full bg-sky-600 hover:bg-sky-500 disabled:opacity-50 text-white text-sm font-medium py-2 rounded-lg"
            disabled={!symbol || isLoading}
            onClick={() => void run()}
          >
            {isLoading ? 'Composing…' : 'Compose idea'}
          </button>

          {error && <div className="text-xs text-rose-300">{error}</div>}

          {idea && (
            <div className="space-y-2 text-xs text-slate-300">
              <div className="flex flex-wrap gap-2">
                <span className="rounded border border-slate-700 px-2 py-0.5">{idea.direction ?? 'n/a'}</span>
                <span className="rounded border border-slate-700 px-2 py-0.5">{idea.actionability ?? 'research'}</span>
                {idea.preview?.preview_ok != null && (
                  <span className="rounded border border-slate-700 px-2 py-0.5">
                    preview {idea.preview.preview_ok ? 'ok' : 'blocked'}
                  </span>
                )}
              </div>
              {idea.narrative && <p className="text-slate-400 leading-relaxed">{idea.narrative}</p>}
              {idea.geometry && (
                <div className="grid grid-cols-3 gap-2 text-[11px]">
                  <div>Entry {idea.geometry.entry ?? '—'}</div>
                  <div>TP {idea.geometry.take_profit ?? '—'}</div>
                  <div>SL {idea.geometry.stop_loss ?? '—'}</div>
                </div>
              )}
              {idea.sizing?.suggested_volume != null && (
                <div>Suggested volume {idea.sizing.suggested_volume}</div>
              )}
              {gates.length > 0 && (
                <ul className="space-y-0.5 text-[11px] text-slate-400">
                  {gates.map(([name, gate]) => (
                    <li key={name}>
                      {name}: {gate.status}
                      {gate.reason ? ` — ${gate.reason}` : ''}
                    </li>
                  ))}
                </ul>
              )}
            </div>
          )}
        </div>
      </div>
    </>
  )
}
