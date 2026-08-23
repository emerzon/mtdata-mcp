import { useEffect, useMemo, useRef, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import {
  getDimredMethods,
  getVolatilityMethods,
  getSktimeEstimators,
  forecastVolatility,
  runBacktest,
  getErrorMessage,
} from '../api/client'
import { useForecast, useForecastMethods, useForecastSettings } from '../hooks/useForecast'
import type { BacktestResult, ForecastPayload, VolatilityPayload } from '../types'
import { coerceParamValue } from '../lib/toolCatalog'
import { formatDateTime } from '../lib/utils'
import { forecastPanelPlacementClass, type LayoutBreakpoint } from '../lib/layout'
import { useEscapeKey } from '../lib/useEscapeKey'
import { DenoiseModal } from './DenoiseModal'
import { ModelsBrowser } from './ModelsBrowser'

type Props = {
  open: boolean
  onClose: () => void
  symbol: string
  timeframe: string
  anchor?: number
  onResult: (res: ForecastPayload | null) => void
  layoutBreakpoint?: LayoutBreakpoint
}

type Tab = 'forecast' | 'volatility' | 'backtest'

export function ForecastPanel({
  open,
  onClose,
  symbol,
  timeframe,
  anchor,
  onResult,
  layoutBreakpoint = 'desktop',
}: Props) {
  const [tab, setTab] = useState<Tab>('forecast')
  useEscapeKey(open, onClose)

  if (!open) return null

  const panelClass = forecastPanelPlacementClass(layoutBreakpoint)

  return (
    <>
      {layoutBreakpoint === 'mobile' && (
        <button
          type="button"
          className="fixed inset-0 z-20 bg-slate-950/50 backdrop-blur-[1px]"
          aria-label="Dismiss forecast panel"
          onClick={onClose}
        />
      )}
      <div className={`${panelClass} animate-slide-in-right`} role="dialog" aria-modal="true" aria-label="Forecast panel">
        {layoutBreakpoint === 'mobile' && (
          <div className="flex justify-center pt-2 pb-1" aria-hidden>
            <div className="h-1 w-10 rounded-full bg-slate-700" />
          </div>
        )}
        <div className="flex items-center justify-between px-4 py-3 border-b border-slate-800 shrink-0">
          <div className="flex gap-1 flex-wrap">
            {(['forecast', 'volatility', 'backtest'] as Tab[]).map((item) => (
              <button
                key={item}
                className={`px-3 py-1.5 min-h-9 text-xs font-medium rounded ${
                  tab === item ? 'bg-sky-600 text-white' : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800'
                }`}
                onClick={() => setTab(item)}
              >
                {item === 'forecast' ? 'Price' : item === 'volatility' ? 'Volatility' : 'Backtest'}
              </button>
            ))}
          </div>
          <button className="text-slate-400 hover:text-slate-200 p-2 min-h-9 min-w-9" onClick={onClose} aria-label="Close forecast panel">
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        <div className="flex-1 overflow-y-auto overscroll-contain p-4 min-h-0">
          {tab === 'forecast' && (
            <ForecastTab
              key={`${symbol}:${timeframe}`}
              symbol={symbol}
              timeframe={timeframe}
              anchor={anchor}
              onResult={onResult}
            />
          )}
          {tab === 'volatility' && <VolatilityTab symbol={symbol} timeframe={timeframe} anchor={anchor} />}
          {tab === 'backtest' && <BacktestTab symbol={symbol} timeframe={timeframe} />}
        </div>
      </div>
    </>
  )
}

function SktimeEstimatorsList() {
  const { data, error, isFetching, refetch, isFetched } = useQuery({
    queryKey: ['sktime_estimators'],
    queryFn: getSktimeEstimators,
    staleTime: 60_000,
  })
  const estimators = data?.estimators ?? []

  return (
    <div className="space-y-1" data-sktime-estimators>
      <div className="flex items-center justify-between">
        <span className="text-xs text-slate-400">sktime estimators</span>
        <button
          type="button"
          className="text-xs text-sky-400 hover:text-sky-300"
          onClick={() => void refetch()}
          disabled={isFetching}
        >
          {isFetching ? 'Loading…' : 'Refresh'}
        </button>
      </div>
      {error && (
        <p className="text-xs text-rose-300">{getErrorMessage(error)}</p>
      )}
      {isFetched && data && !data.available && (
        <p className="text-xs text-slate-500">
          sktime not available{data.error ? `: ${data.error}` : ''}.
        </p>
      )}
      {estimators.length > 0 && (
        <ul className="max-h-24 overflow-y-auto rounded border border-slate-800 bg-slate-950/50 p-1 text-[11px] text-slate-400">
          {estimators.slice(0, 40).map((est) => (
            <li key={est.class_path || est.name} className="px-1.5 py-0.5 truncate" title={est.class_path}>
              {est.name}
            </li>
          ))}
          {estimators.length > 40 && (
            <li className="px-1.5 py-0.5 text-slate-600">+{estimators.length - 40} more</li>
          )}
        </ul>
      )}
    </div>
  )
}

function ForecastTab({
  symbol,
  timeframe,
  anchor,
  onResult,
}: {
  symbol: string
  timeframe: string
  anchor?: number
  onResult: (res: ForecastPayload | null) => void
}) {
  const { methods, error: methodsError } = useForecastMethods()
  const { settings, setSettings, supportsDimred } = useForecastSettings(symbol, timeframe)
  const { data: dimredMethods } = useQuery({ queryKey: ['dimred_methods'], queryFn: getDimredMethods })
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [showDenoise, setShowDenoise] = useState(false)
  const { run, isLoading, error } = useForecast(symbol, timeframe, settings, onResult, anchor)

  const selectedMeta = useMemo(
    () => methods.find((method) => method.method === settings.method),
    [methods, settings.method]
  )
  const availableDimred = (dimredMethods?.methods ?? []).filter((method) => method.available)

  return (
    <div className="space-y-4">
      <div>
        <label className="text-xs text-slate-400 mb-1 block">Method</label>
        <select
          className="w-full bg-slate-800 text-slate-200 text-sm rounded-lg px-3 py-2 border border-slate-700"
          value={settings.method}
          onChange={(event) =>
            setSettings((previous) => ({
              ...previous,
              method: event.target.value,
            }))
          }
        >
          {methods.map((method) => (
            <option key={method.method} value={method.method} disabled={!method.available}>
              {method.method}
              {!method.available ? ' (unavailable)' : ''}
            </option>
          ))}
        </select>
        {selectedMeta && !selectedMeta.available && (
          <p className="text-xs text-amber-400 mt-1">
            Requires: {selectedMeta.requires?.join(', ') || 'additional dependencies'}
          </p>
        )}
      </div>

      <div className="grid grid-cols-2 gap-3">
        <div>
          <label className="text-xs text-slate-400 mb-1 block">Horizon</label>
          <input
            type="number"
            className="w-full bg-slate-800 text-slate-200 text-sm rounded-lg px-3 py-2 border border-slate-700"
            value={settings.horizon}
            onChange={(event) =>
              setSettings((previous) => ({
                ...previous,
                horizon: Number(event.target.value),
              }))
            }
            min={1}
          />
        </div>
        <div>
          <label className="text-xs text-slate-400 mb-1 block">Quantity</label>
          <select
            className="w-full bg-slate-800 text-slate-200 text-sm rounded-lg px-3 py-2 border border-slate-700"
            value={settings.quantity}
            onChange={(event) =>
              setSettings((previous) => ({
                ...previous,
                quantity: event.target.value as 'price' | 'return',
              }))
            }
          >
            <option value="price">Price</option>
            <option value="return">Return</option>
          </select>
        </div>
      </div>

      <button
        className="w-full text-left text-xs text-slate-400 hover:text-slate-300 flex items-center justify-between py-2 border-t border-slate-800"
        onClick={() => setShowAdvanced((value) => !value)}
      >
        <span>Advanced Options</span>
        <span>{showAdvanced ? '−' : '+'}</span>
      </button>

      {showAdvanced && (
        <div className="space-y-3 pb-3 border-b border-slate-800">
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="text-xs text-slate-400 mb-1 block">Lookback</label>
              <input
                type="number"
                className="w-full bg-slate-800 text-slate-200 text-sm rounded-lg px-3 py-2 border border-slate-700"
                value={settings.lookback}
                onChange={(event) =>
                  setSettings((previous) => ({
                    ...previous,
                    lookback: event.target.value === '' ? '' : Number(event.target.value),
                  }))
                }
                placeholder="auto"
                min={50}
              />
            </div>
            <div>
              <label className="text-xs text-slate-400 mb-1 block">CI Alpha</label>
              <input
                type="number"
                className="w-full bg-slate-800 text-slate-200 text-sm rounded-lg px-3 py-2 border border-slate-700"
                value={settings.ci_alpha}
                onChange={(event) =>
                  setSettings((previous) => ({
                    ...previous,
                    ci_alpha: Number(event.target.value),
                  }))
                }
                step={0.01}
                min={0}
                max={0.5}
              />
            </div>
          </div>

          <div className="flex items-center justify-between">
            <span className="text-xs text-slate-400">
              Forecast Denoise: <span className="text-slate-300">{settings.denoise?.method || 'None'}</span>
            </span>
            <button className="text-xs text-sky-400 hover:text-sky-300" onClick={() => setShowDenoise(true)}>
              Configure
            </button>
          </div>

          {supportsDimred && (
            <div>
              <label className="text-xs text-slate-400 mb-1 block">Dim. Reduction</label>
              <select
                className="w-full bg-slate-800 text-slate-200 text-sm rounded-lg px-3 py-2 border border-slate-700"
                value={settings.dimredMethod ?? ''}
                onChange={(event) =>
                  setSettings((previous) => ({
                    ...previous,
                    dimredMethod: event.target.value || undefined,
                  }))
                }
              >
                <option value="">None</option>
                {availableDimred.map((method) => (
                  <option key={method.method} value={method.method}>
                    {method.method}
                  </option>
                ))}
              </select>
            </div>
          )}

          <ModelsBrowser methodFilter={settings.method} compact />

          <SktimeEstimatorsList />

          {selectedMeta?.params && selectedMeta.params.length > 0 && (
            <div>
              <div className="text-xs text-slate-400 mb-2">Method Parameters</div>
              <div className="grid grid-cols-2 gap-2">
                {selectedMeta.params.map((param) => (
                  <div key={param.name}>
                    <label className="text-xs text-slate-500 mb-0.5 block">{param.name}</label>
                    <input
                      className="w-full bg-slate-800 text-slate-200 text-xs rounded px-2 py-1.5 border border-slate-700"
                      value={String(settings.params[param.name] ?? '')}
                      onChange={(event) =>
                        setSettings((previous) => ({
                          ...previous,
                          params: {
                            ...previous.params,
                            [param.name]: coerceParamValue(event.target.value, param.type),
                          },
                        }))
                      }
                      placeholder={String(param.default ?? '')}
                    />
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {error && (
        <div className="text-sm text-rose-400 bg-rose-950/50 border border-rose-800 rounded-lg px-3 py-2">
          {error}
        </div>
      )}

      {methodsError && (
        <div className="text-sm text-rose-400 bg-rose-950/50 border border-rose-800 rounded-lg px-3 py-2">
          Forecast methods: {methodsError}
        </div>
      )}

      <div className="flex gap-2">
        <button
          className="flex-1 bg-sky-600 hover:bg-sky-500 text-white text-sm font-medium py-2 rounded-lg disabled:opacity-50"
          onClick={() => run('full')}
          disabled={!symbol || !selectedMeta?.available || isLoading}
        >
          {isLoading ? 'Running...' : 'Full Forecast'}
        </button>
        <button
          className="flex-1 bg-slate-700 hover:bg-slate-600 text-white text-sm font-medium py-2 rounded-lg disabled:opacity-50"
          onClick={() => run('partial', anchor)}
          disabled={!symbol || !selectedMeta?.available || !anchor || isLoading}
        >
          From Anchor
        </button>
      </div>

      <DenoiseModal
        open={showDenoise}
        title="Forecast Denoising"
        value={settings.denoise}
        onClose={() => setShowDenoise(false)}
        onApply={(denoise) => {
          setSettings((previous) => ({
            ...previous,
            denoise,
          }))
          setShowDenoise(false)
        }}
      />
    </div>
  )
}

function VolatilityTab({ symbol, timeframe, anchor }: { symbol: string; timeframe: string; anchor?: number }) {
  const { data: methods, error: methodsQueryError } = useQuery({
    queryKey: ['vol_methods'],
    queryFn: getVolatilityMethods,
  })

  const [method, setMethod] = useState('ewma')
  const [horizon, setHorizon] = useState(12)
  const [proxy, setProxy] = useState('squared_return')
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<VolatilityPayload | null>(null)
  const runId = useRef(0)
  const requestContract = JSON.stringify({ symbol, timeframe, anchor, method, horizon, proxy })
  const requestContractRef = useRef(requestContract)
  requestContractRef.current = requestContract

  useEffect(() => {
    runId.current += 1
    setIsLoading(false)
    setError(null)
    setResult(null)
  }, [anchor, horizon, method, proxy, symbol, timeframe])

  const run = async () => {
    if (!symbol) return
    const runContract = requestContract
    const currentRunId = ++runId.current
    setIsLoading(true)
    setError(null)
    setResult(null)
    try {
      const response = await forecastVolatility({
        symbol,
        timeframe,
        method,
        horizon,
        proxy,
        as_of: anchor ? formatDateTime(anchor) : undefined,
      })
      if (currentRunId === runId.current && runContract === requestContractRef.current) {
        setResult(response)
      }
    } catch (err) {
      if (currentRunId === runId.current && runContract === requestContractRef.current) {
        setError(getErrorMessage(err))
      }
    } finally {
      if (currentRunId === runId.current && runContract === requestContractRef.current) {
        setIsLoading(false)
      }
    }
  }

  return (
    <div className="space-y-4">
      <div>
        <label className="text-xs text-slate-400 mb-1 block">Method</label>
        <select
          className="w-full bg-slate-800 text-slate-200 text-sm rounded-lg px-3 py-2 border border-slate-700"
          value={method}
          onChange={(event) => setMethod(event.target.value)}
        >
          {methods?.methods?.map((item) => (
            <option key={item.method} value={item.method} disabled={!item.available}>
              {item.method}
              {!item.available ? ' (unavailable)' : ''}
            </option>
          ))}
        </select>
      </div>

      <div className="grid grid-cols-2 gap-3">
        <div>
          <label className="text-xs text-slate-400 mb-1 block">Horizon</label>
          <input
            type="number"
            className="w-full bg-slate-800 text-slate-200 text-sm rounded-lg px-3 py-2 border border-slate-700"
            value={horizon}
            onChange={(event) => setHorizon(Number(event.target.value))}
            min={1}
          />
        </div>
        <div>
          <label className="text-xs text-slate-400 mb-1 block">Proxy</label>
          <select
            className="w-full bg-slate-800 text-slate-200 text-sm rounded-lg px-3 py-2 border border-slate-700"
            value={proxy}
            onChange={(event) => setProxy(event.target.value)}
          >
            <option value="squared_return">Squared Return</option>
            <option value="abs_return">Abs Return</option>
            <option value="log_r2">Log R²</option>
          </select>
        </div>
      </div>

      {error && (
        <div className="text-sm text-rose-400 bg-rose-950/50 border border-rose-800 rounded-lg px-3 py-2">
          {error}
        </div>
      )}

      {methodsQueryError && (
        <div className="text-sm text-rose-400 bg-rose-950/50 border border-rose-800 rounded-lg px-3 py-2">
          Volatility methods: {getErrorMessage(methodsQueryError)}
        </div>
      )}

      <button
        className="w-full bg-sky-600 hover:bg-sky-500 text-white text-sm font-medium py-2 rounded-lg disabled:opacity-50"
        onClick={run}
        disabled={!symbol || isLoading}
      >
        {isLoading ? 'Running...' : 'Run Volatility Forecast'}
      </button>

      {result && (
        <div className="bg-slate-800/50 rounded-lg p-3 text-sm">
          <div className="text-slate-400 text-xs mb-2">Result</div>
          <div className="text-slate-200">
            Annualized Vol:{' '}
            <span className="font-mono">
              {result.volatility_annualized === undefined
                ? 'Unavailable'
                : `${(result.volatility_annualized * 100).toFixed(2)}%`}
            </span>
          </div>
        </div>
      )}
    </div>
  )
}

function BacktestTab({ symbol, timeframe }: { symbol: string; timeframe: string }) {
  const { methods, error: methodsError } = useForecastMethods()

  const [selectedMethods, setSelectedMethods] = useState<string[]>(['theta'])
  const [horizon, setHorizon] = useState(12)
  const [steps, setSteps] = useState(5)
  const [spacing, setSpacing] = useState(20)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<BacktestResult | null>(null)
  const runId = useRef(0)
  const requestContract = JSON.stringify({
    symbol,
    timeframe,
    selectedMethods,
    horizon,
    steps,
    spacing,
  })
  const requestContractRef = useRef(requestContract)
  requestContractRef.current = requestContract

  const availableMethods = useMemo(() => methods.filter((method) => method.available), [methods])
  const resultRows = useMemo(() => {
    if (result?.ranked_methods) return result.ranked_methods
    return Object.entries(result?.results ?? {}).map(([method, item]) => ({ method, ...item }))
  }, [result])

  useEffect(() => {
    runId.current += 1
    setIsLoading(false)
    setError(null)
    setResult(null)
  }, [horizon, selectedMethods, spacing, steps, symbol, timeframe])

  const toggleMethod = (method: string) => {
    setSelectedMethods((previous) =>
      previous.includes(method) ? previous.filter((item) => item !== method) : [...previous, method]
    )
  }

  const run = async () => {
    if (!symbol || !selectedMethods.length) return
    const runContract = requestContract
    const currentRunId = ++runId.current
    setIsLoading(true)
    setError(null)
    setResult(null)
    try {
      const response = await runBacktest({ symbol, timeframe, horizon, steps, spacing, methods: selectedMethods })
      if (currentRunId === runId.current && runContract === requestContractRef.current) {
        setResult(response)
      }
    } catch (err) {
      if (currentRunId === runId.current && runContract === requestContractRef.current) {
        setError(getErrorMessage(err))
      }
    } finally {
      if (currentRunId === runId.current && runContract === requestContractRef.current) {
        setIsLoading(false)
      }
    }
  }

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-3 gap-2">
        <div>
          <label className="text-xs text-slate-400 mb-1 block">Horizon</label>
          <input
            type="number"
            className="w-full bg-slate-800 text-slate-200 text-sm rounded-lg px-2 py-2 border border-slate-700"
            value={horizon}
            onChange={(event) => setHorizon(Number(event.target.value))}
            min={1}
          />
        </div>
        <div>
          <label className="text-xs text-slate-400 mb-1 block">Steps</label>
          <input
            type="number"
            className="w-full bg-slate-800 text-slate-200 text-sm rounded-lg px-2 py-2 border border-slate-700"
            value={steps}
            onChange={(event) => setSteps(Number(event.target.value))}
            min={1}
          />
        </div>
        <div>
          <label className="text-xs text-slate-400 mb-1 block">Spacing</label>
          <input
            type="number"
            className="w-full bg-slate-800 text-slate-200 text-sm rounded-lg px-2 py-2 border border-slate-700"
            value={spacing}
            onChange={(event) => setSpacing(Number(event.target.value))}
            min={1}
          />
        </div>
      </div>

      <div>
        <div className="text-xs text-slate-400 mb-2">Methods to compare</div>
        <div className="flex flex-wrap gap-1">
          {availableMethods.map((method) => (
            <button
              key={method.method}
              className={`px-2 py-1 text-xs rounded ${
                selectedMethods.includes(method.method)
                  ? 'bg-sky-600 text-white'
                  : 'bg-slate-800 text-slate-400 hover:bg-slate-700'
              }`}
              onClick={() => toggleMethod(method.method)}
            >
              {method.method}
            </button>
          ))}
        </div>
      </div>

      {error && (
        <div className="text-sm text-rose-400 bg-rose-950/50 border border-rose-800 rounded-lg px-3 py-2">
          {error}
        </div>
      )}

      {methodsError && (
        <div className="text-sm text-rose-400 bg-rose-950/50 border border-rose-800 rounded-lg px-3 py-2">
          Forecast methods: {methodsError}
        </div>
      )}

      <button
        className="w-full bg-sky-600 hover:bg-sky-500 text-white text-sm font-medium py-2 rounded-lg disabled:opacity-50"
        onClick={run}
        disabled={!symbol || !selectedMethods.length || isLoading}
      >
        {isLoading ? 'Running Backtest...' : 'Run Backtest'}
      </button>

      {result && (
        <div className="bg-slate-800/50 rounded-lg overflow-hidden">
          <table className="w-full text-xs">
            <thead>
              <tr className="text-slate-400 border-b border-slate-700">
                <th className="text-left px-2 py-2">Method</th>
                <th className="text-right px-2 py-2">MAE</th>
                <th className="text-right px-2 py-2">Dir%</th>
              </tr>
            </thead>
            <tbody>
              {resultRows.map((item) => {
                const method = item.method
                const directionPercent = item.avg_directional_accuracy == null
                  ? null
                  : item.avg_directional_accuracy * 100
                return (
                <tr key={method} className="border-b border-slate-700/50">
                  <td className="px-2 py-1.5 text-slate-200">{method}</td>
                  <td className="text-right px-2 py-1.5 text-slate-300 font-mono">
                    {item.avg_mae?.toFixed(4) ?? '-'}
                  </td>
                  <td
                    className={`text-right px-2 py-1.5 font-mono ${
                      (directionPercent ?? 0) >= 60
                        ? 'text-emerald-400'
                        : (directionPercent ?? 0) >= 50
                          ? 'text-amber-400'
                          : 'text-rose-400'
                    }`}
                  >
                    {directionPercent?.toFixed(0) ?? '-'}
                  </td>
                </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
