import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { getErrorMessage, getTool, invokeTool, listTools } from '../api/client'
import {
  defaultParamValues,
  filterToolCatalog,
  formatToolResult,
  humanizeIdentifier,
  invocationNeedsConfirmation,
  shapeInvokeArguments,
  schemaToToolFields,
  toolChangesTradingState,
  toolIsRunnable,
  uniqueCategories,
  type ToolCatalogEntry,
  type ToolParamValues,
} from '../lib/toolCatalog'
import type { LayoutBreakpoint } from '../lib/layout'
import { createToolRunGate } from '../lib/toolRunState'
import { WorkspacePanelShell } from './WorkspacePanelShell'

type Props = {
  open: boolean
  onClose: () => void
  layoutBreakpoint?: LayoutBreakpoint
  /** Prefill symbol into forms that expose a symbol field */
  symbol?: string
  timeframe?: string
}

export function ToolOmissionNotice({ tool }: { tool: ToolCatalogEntry }) {
  const rationale = tool.safety?.omit_rationale?.trim()
  if (!rationale) return null
  return (
    <div
      className="text-xs text-amber-100 bg-amber-950/40 border border-amber-800 rounded-lg px-3 py-2"
      role="note"
    >
      <span className="font-medium">Not available in the synchronous Tools runner.</span>{' '}
      {rationale}
    </div>
  )
}

export function ToolsRunnerPanel({
  open,
  onClose,
  layoutBreakpoint = 'desktop',
  symbol = '',
  timeframe = 'H1',
}: Props) {
  const queryClient = useQueryClient()
  const [search, setSearch] = useState('')
  const [category, setCategory] = useState('')
  const [selectedName, setSelectedName] = useState<string | null>(null)
  const [values, setValues] = useState<ToolParamValues>({})
  const [confirm, setConfirm] = useState(false)
  const [isRunning, setIsRunning] = useState(false)
  const [runError, setRunError] = useState<string | null>(null)
  const [resultText, setResultText] = useState<string | null>(null)
  const selectedNameRef = useRef<string | null>(selectedName)
  selectedNameRef.current = selectedName
  const runGateRef = useRef(createToolRunGate())
  const abandonRun = useCallback(() => {
    runGateRef.current.invalidate()
    setIsRunning(false)
    setConfirm(false)
  }, [])
  const closePanel = useCallback(() => {
    abandonRun()
    onClose()
  }, [abandonRun, onClose])

  const catalogQuery = useQuery({
    queryKey: ['tools-catalog'],
    queryFn: ({ signal }) => listTools({}, signal),
    enabled: open,
    staleTime: 60_000,
  })

  const tools = catalogQuery.data?.tools ?? []
  const categories = useMemo(() => uniqueCategories(tools), [tools])
  const filtered = useMemo(
    () => filterToolCatalog(tools, { search, category }),
    [tools, search, category]
  )

  const detailQuery = useQuery({
    queryKey: ['tool-detail', selectedName],
    queryFn: ({ signal }) => getTool(selectedName!, signal),
    enabled: open && !!selectedName,
    staleTime: 30_000,
  })

  const selected: ToolCatalogEntry | null = detailQuery.data?.tool ?? null
  const fields = useMemo(
    () => schemaToToolFields(selected?.input_schema),
    [selected?.input_schema]
  )

  useEffect(() => {
    if (!selected?.name) return
    abandonRun()
    const next = defaultParamValues(fields)
    if (symbol && 'symbol' in next && !next.symbol) next.symbol = symbol
    if (timeframe && 'timeframe' in next && !next.timeframe) next.timeframe = timeframe
    setValues(next)
    setConfirm(false)
    setRunError(null)
    setResultText(null)
  }, [selected?.name, fields, symbol, timeframe, abandonRun])

  useEffect(() => {
    if (!open) abandonRun()
  }, [open, abandonRun])

  useEffect(() => () => runGateRef.current.invalidate(), [])

  const onSelect = useCallback((name: string) => {
    abandonRun()
    selectedNameRef.current = name
    setSelectedName(name)
    setRunError(null)
    setResultText(null)
  }, [abandonRun])

  const run = useCallback(async () => {
    if (!selected?.name || !toolIsRunnable(selected)) return
    const runIdentity = runGateRef.current.begin(selected.name)
    setIsRunning(true)
    setRunError(null)
    setResultText(null)
    try {
      const argumentsPayload = shapeInvokeArguments(fields, values)
      const response = await invokeTool(selected.name, {
        arguments: argumentsPayload,
        confirm,
      })
      if (!runGateRef.current.isCurrent(runIdentity, selectedNameRef.current)) return
      if (toolChangesTradingState(selected)) {
        await Promise.all([
          queryClient.invalidateQueries({ queryKey: ['exposure'], refetchType: 'active' }),
          queryClient.invalidateQueries({ queryKey: ['session-strip'], refetchType: 'active' }),
        ])
      }
      if (!runGateRef.current.isCurrent(runIdentity, selectedNameRef.current)) return
      setResultText(formatToolResult(response.result ?? response))
    } catch (error) {
      if (!runGateRef.current.isCurrent(runIdentity, selectedNameRef.current)) return
      setRunError(getErrorMessage(error))
    } finally {
      if (runGateRef.current.isCurrent(runIdentity, selectedNameRef.current)) {
        setIsRunning(false)
        setConfirm(false)
      }
    }
  }, [selected, fields, values, confirm, queryClient])

  const onFieldChange = useCallback((name: string, value: string) => {
    setValues((prev) => ({ ...prev, [name]: value }))
    setConfirm(false)
  }, [])

  const needsConfirm = invocationNeedsConfirmation(selected, fields, values)
  const runnable = toolIsRunnable(selected)
  const registeredCount =
    catalogQuery.data?.pagination?.total ?? catalogQuery.data?.count ?? tools.length

  return (
    <WorkspacePanelShell
      open={open}
      onClose={closePanel}
      layoutBreakpoint={layoutBreakpoint}
      label="Tools runner"
      dismissLabel="Dismiss tools panel"
      closeLabel="Close tools panel"
      header={
        <div>
          <h2 className="text-sm font-semibold text-slate-100">All tools</h2>
          <p className="text-[11px] text-slate-500">
            {registeredCount} registered · schema-driven runner
          </p>
        </div>
      }
      bodyClassName="flex-1 min-h-0 flex flex-col md:flex-row overflow-hidden"
      dialogData={{ 'data-tools-runner': '' }}
    >
          <div className="md:w-2/5 border-b md:border-b-0 md:border-r border-slate-800 flex flex-col min-h-0 max-h-[40vh] md:max-h-none">
            <div className="p-3 space-y-2 shrink-0">
              <input
                className="w-full bg-slate-800 text-slate-200 text-sm rounded-lg px-3 py-2 border border-slate-700"
                placeholder="Search tools…"
                value={search}
                onChange={(event) => setSearch(event.target.value)}
                aria-label="Search tools"
              />
              <select
                className="w-full bg-slate-800 text-slate-200 text-xs rounded-lg px-2 py-1.5 border border-slate-700"
                value={category}
                onChange={(event) => setCategory(event.target.value)}
                aria-label="Filter by category"
              >
                <option value="">All categories</option>
                {categories.map((item) => (
                  <option key={item} value={item}>
                    {humanizeIdentifier(item)}
                  </option>
                ))}
              </select>
            </div>

            <div className="flex-1 overflow-y-auto min-h-0">
              {catalogQuery.isLoading && (
                <p className="px-3 py-2 text-xs text-slate-500">Loading catalog…</p>
              )}
              {catalogQuery.error && (
                <p className="px-3 py-2 text-xs text-rose-300">{getErrorMessage(catalogQuery.error)}</p>
              )}
              {!catalogQuery.isLoading && filtered.length === 0 && (
                <p className="px-3 py-2 text-xs text-slate-500">No tools match this filter.</p>
              )}
              <ul className="pb-2">
                {filtered.map((tool) => {
                  const active = tool.name === selectedName
                  return (
                    <li key={tool.name}>
                      <button
                        type="button"
                        className={`w-full text-left px-3 py-2 text-xs border-l-2 ${
                          active
                            ? 'bg-slate-800/80 border-sky-500 text-sky-200'
                            : 'border-transparent text-slate-300 hover:bg-slate-800/50'
                        }`}
                        onClick={() => onSelect(tool.name)}
                      >
                        <div className="font-medium truncate">{tool.name}</div>
                        <div className="text-[10px] text-slate-500 truncate">
                          {tool.category}
                          {tool.surface === 'dedicated_ui' ? ' · dedicated UI' : ''}
                          {tool.safety?.requires_confirmation ? ' · confirm' : ''}
                          {tool.enabled === false ? ' · disabled' : ''}
                        </div>
                      </button>
                    </li>
                  )
                })}
              </ul>
            </div>
          </div>

          <div className="flex-1 overflow-y-auto p-4 space-y-3 min-h-0">
            {!selectedName && (
              <div className="text-sm text-slate-400">
                Select a tool to configure parameters and run it against the live Web API.
              </div>
            )}

            {selectedName && detailQuery.isLoading && (
              <p className="text-xs text-slate-500">Loading parameter schema…</p>
            )}

            {detailQuery.error && (
              <div className="text-sm text-rose-300 bg-rose-950/40 border border-rose-900 rounded-lg px-3 py-2">
                {getErrorMessage(detailQuery.error)}
              </div>
            )}

            {selected && (
              <>
                <div>
                  <h3 className="text-sm font-semibold text-slate-100">{selected.name}</h3>
                  {selected.description && (
                    <p className="text-xs text-slate-400 mt-1">{selected.description}</p>
                  )}
                  <div className="mt-2 flex flex-wrap gap-1.5 text-[10px]">
                    <span className="badge badge-info">{selected.category || 'tool'}</span>
                    <span className="badge badge-info">{selected.surface || 'generic_runner'}</span>
                    {selected.safety?.dedicated_path && (
                      <span className="badge badge-success" title="Also available as dedicated UI">
                        UI: {selected.safety.dedicated_path}
                      </span>
                    )}
                  </div>
                </div>

                {selected.safety?.warning && (
                  <div
                    className="text-xs text-amber-200 bg-amber-950/40 border border-amber-800 rounded-lg px-3 py-2"
                    role="alert"
                  >
                    {selected.safety.warning}
                  </div>
                )}

                <ToolOmissionNotice tool={selected} />

                {selected.enabled === false && (
                  <div className="text-xs text-slate-400 bg-slate-900 border border-slate-700 rounded-lg px-3 py-2">
                    Disabled
                    {selected.enable_env ? ` (set ${selected.enable_env}=1)` : ''}.
                    {selected.why_disabled ? ` ${selected.why_disabled}` : ''}
                  </div>
                )}

                <div className="space-y-2">
                  <div className="text-xs font-medium text-slate-300">Parameters</div>
                  {fields.length === 0 && (
                    <p className="text-xs text-slate-500">
                      No parameters required — run with an empty argument set.
                    </p>
                  )}
                  {fields.map((field) => (
                    <label key={field.name} className="block text-[11px] text-slate-500">
                      <span className="flex items-center gap-1">
                        {humanizeIdentifier(field.name)}
                        {field.required && <span className="text-rose-400">*</span>}
                        {field.type && <span className="text-slate-600">· {field.type}</span>}
                      </span>
                      {field.description && (
                        <span className="block text-[10px] text-slate-600 mb-0.5">{field.description}</span>
                      )}
                      <input
                        className="mt-0.5 w-full bg-slate-800 text-slate-200 text-xs rounded px-2 py-1.5 border border-slate-700"
                        value={values[field.name] ?? ''}
                        onChange={(event) => onFieldChange(field.name, event.target.value)}
                        placeholder={field.default !== undefined && field.default !== null ? String(field.default) : ''}
                        disabled={!runnable || isRunning}
                      />
                    </label>
                  ))}
                </div>

                {needsConfirm && (
                  <label className="flex items-start gap-2 text-xs text-amber-200 bg-amber-950/30 border border-amber-900 rounded-lg px-3 py-2">
                    <input
                      type="checkbox"
                      className="mt-0.5"
                      checked={confirm}
                      onChange={(event) => setConfirm(event.target.checked)}
                      disabled={isRunning}
                    />
                    <span>
                      I understand this tool can mutate trading or stored state, and I confirm running
                      it with the parameters above.
                    </span>
                  </label>
                )}

                <button
                  type="button"
                  className="w-full bg-sky-600 hover:bg-sky-500 text-white text-sm font-medium py-2 rounded-lg disabled:opacity-50"
                  disabled={!runnable || isRunning || (needsConfirm && !confirm)}
                  onClick={() => void run()}
                >
                  {isRunning ? 'Running…' : needsConfirm ? 'Run with confirmation' : 'Run tool'}
                </button>

                {runError && (
                  <div className="text-sm text-rose-300 bg-rose-950/40 border border-rose-900 rounded-lg px-3 py-2 whitespace-pre-wrap break-words">
                    {runError}
                  </div>
                )}

                {resultText !== null && (
                  <div className="space-y-1">
                    <div className="text-xs font-medium text-emerald-300">Result</div>
                    <pre className="text-[11px] text-slate-200 bg-slate-950 border border-slate-800 rounded-lg p-3 overflow-auto max-h-72 whitespace-pre-wrap break-words">
                      {resultText}
                    </pre>
                  </div>
                )}
              </>
            )}
          </div>
    </WorkspacePanelShell>
  )
}
