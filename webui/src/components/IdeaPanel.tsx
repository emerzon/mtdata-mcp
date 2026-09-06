import { useCallback, useEffect, useRef, useState } from 'react'
import { composeTradeIdea, getErrorMessage } from '../api/client'
import type { LayoutBreakpoint } from '../lib/layout'
import { tradeIdeaSectionFailures } from '../lib/ideaFeedback'
import { createToolRunGate } from '../lib/toolRunState'
import type { TradeIdeaPayload } from '../types'
import { WorkspacePanelShell } from './WorkspacePanelShell'

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
  const runGateRef = useRef(createToolRunGate())

  useEffect(() => {
    onIdeaRef.current = onIdea
  }, [onIdea])

  useEffect(() => {
    runGateRef.current.invalidate()
    setIsLoading(false)
    setError(null)
    setIdea(null)
    onIdeaRef.current(null)
    return () => runGateRef.current.invalidate()
  }, [requestKey])

  const run = useCallback(async () => {
    if (!symbol) return

    const runIdentity = runGateRef.current.begin(requestKey)
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
      if (!runGateRef.current.isCurrent(runIdentity, requestKeyRef.current)) return
      setIdea(result)
      onIdeaRef.current(result)
    } catch (err) {
      if (runGateRef.current.isCurrent(runIdentity, requestKeyRef.current)) {
        setIdea(null)
        onIdeaRef.current(null)
        setError(getErrorMessage(err))
      }
    } finally {
      if (runGateRef.current.isCurrent(runIdentity, requestKeyRef.current)) {
        setIsLoading(false)
      }
    }
  }, [direction, horizon, requestKey, riskPct, symbol, template, timeframe])

  useEffect(() => {
    if (!open || !autoComposeKey) return
    void run()
  }, [autoComposeKey, open, run])

  const gates = Object.entries(idea?.gates ?? {})
  const sectionFailures = tradeIdeaSectionFailures(idea)

  return (
    <WorkspacePanelShell
      open={open}
      onClose={onClose}
      layoutBreakpoint={layoutBreakpoint}
      label="Trade idea panel"
      dismissLabel="Dismiss idea panel"
      closeLabel="Close idea panel"
      header={
        <div>
          <h2 className="text-sm font-medium text-slate-100">Idea</h2>
          <p className="text-[11px] text-slate-500">Preview-only research. Cannot place an order.</p>
        </div>
      }
      bodyClassName="flex-1 overflow-y-auto overscroll-contain p-4 min-h-0 space-y-3"
    >
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
              {idea.partial_failure && (
                <div
                  className="rounded-lg border border-amber-800 bg-amber-950/50 px-3 py-2 text-amber-100"
                  role="status"
                >
                  <div className="font-medium">Partial idea — some research sections failed.</div>
                  <ul className="mt-1 list-disc space-y-1 pl-4">
                    {sectionFailures.map((failure) => (
                      <li key={failure.section}>
                        <span className="font-medium">{failure.section}</span>: {failure.reason}
                        {failure.remediation ? ` ${failure.remediation}` : ''}
                      </li>
                    ))}
                  </ul>
                  <div className="mt-1 text-amber-200">
                    Do not infer missing sections from the remaining values.
                    {idea.failed_sections?.includes('volatility')
                      ? ' Exit geometry may use fallback barrier percentages.'
                      : ''}
                  </div>
                </div>
              )}
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
    </WorkspacePanelShell>
  )
}
