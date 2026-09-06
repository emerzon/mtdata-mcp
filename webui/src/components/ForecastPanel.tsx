import { useEffect, useMemo, useRef, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import {
  getVolatilityMethods,
  forecastVolatility,
  runBacktest,
  getErrorMessage,
} from '../api/client'
import { useForecast, useForecastMethods, useForecastSettings } from '../hooks/useForecast'
import type {
  BacktestResult,
  DenoiseSpecUI,
  ForecastPayload,
  ParamDef,
  VolatilityPayload,
} from '../types'
import {
  backtestDisplayRows,
  backtestMethodStatus,
  backtestResultFeedback,
  forecastResultFeedback,
  outputWarningMessage,
  type ResultFeedback,
} from '../lib/compactForecast'
import {
  scopedBacktestParams,
  sharedBacktestParamDefs,
  updateMethodParameter,
  updateParameterValue,
} from '../lib/forecastContracts'
import {
  buildVolatilityRequest,
  volatilityProxyForMethod,
  volatilityResultMetrics,
} from '../lib/volatilityContracts'
import { formatDateTime } from '../lib/utils'
import { createToolRunGate } from '../lib/toolRunState'
import type { LayoutBreakpoint } from '../lib/layout'
import { DenoiseModal } from './DenoiseModal'
import { ModelsBrowser } from './ModelsBrowser'
import { WorkspacePanelShell } from './WorkspacePanelShell'

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
  const [denoiseOpen, setDenoiseOpen] = useState(false)

  useEffect(() => {
    if (!open) setDenoiseOpen(false)
  }, [open])

  return (
    <WorkspacePanelShell
      open={open}
      onClose={onClose}
      layoutBreakpoint={layoutBreakpoint}
      label="Forecast panel"
      dismissLabel="Dismiss forecast panel"
      closeLabel="Close forecast panel"
      dismissEnabled={!denoiseOpen}
      header={
        <div className="flex gap-1 flex-wrap">
          {(['forecast', 'volatility', 'backtest'] as Tab[]).map((item) => (
            <button
              key={item}
              className={`px-3 py-1.5 min-h-9 text-xs font-medium rounded ${
                tab === item ? 'bg-sky-600 text-white' : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800'
              }`}
              onClick={() => {
                setTab(item)
                setDenoiseOpen(false)
              }}
            >
              {item === 'forecast' ? 'Price' : item === 'volatility' ? 'Volatility' : 'Backtest'}
            </button>
          ))}
        </div>
      }
    >
      {tab === 'forecast' && (
        <ForecastTab
          key={`${symbol}:${timeframe}`}
          symbol={symbol}
          timeframe={timeframe}
          anchor={anchor}
          onResult={onResult}
          denoiseOpen={denoiseOpen}
          onDenoiseOpenChange={setDenoiseOpen}
        />
      )}
      {tab === 'volatility' && <VolatilityTab symbol={symbol} timeframe={timeframe} anchor={anchor} />}
      {tab === 'backtest' && (
        <BacktestTab
          symbol={symbol}
          timeframe={timeframe}
          denoiseOpen={denoiseOpen}
          onDenoiseOpenChange={setDenoiseOpen}
        />
      )}
    </WorkspacePanelShell>
  )
}

function ForecastTab({
  symbol,
  timeframe,
  anchor,
  onResult,
  denoiseOpen,
  onDenoiseOpenChange,
}: {
  symbol: string
  timeframe: string
  anchor?: number
  onResult: (res: ForecastPayload | null) => void
  denoiseOpen: boolean
  onDenoiseOpenChange: (open: boolean) => void
}) {
  const { methods, error: methodsError } = useForecastMethods()
  const { settings, setSettings } = useForecastSettings(symbol, timeframe)
  const [showAdvanced, setShowAdvanced] = useState(false)
  const selectedMeta = useMemo(
    () => methods.find((method) => method.method === settings.method),
    [methods, settings.method]
  )
  const { run, isLoading, error, result } = useForecast(
    symbol,
    timeframe,
    settings,
    onResult,
    anchor,
    selectedMeta?.params ?? []
  )

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
            <button className="text-xs text-sky-400 hover:text-sky-300" onClick={() => onDenoiseOpenChange(true)}>
              Configure
            </button>
          </div>

          <ModelsBrowser methodFilter={settings.method} compact />

          {selectedMeta?.params && selectedMeta.params.length > 0 && (
            <div>
              <div className="text-xs text-slate-400 mb-2">Method Parameters</div>
              <div className="grid grid-cols-2 gap-2">
                {selectedMeta.params.map((param) => (
                  <div key={param.name}>
                    <label className="text-xs text-slate-500 mb-0.5 block">{param.name}</label>
                    <input
                      className="w-full bg-slate-800 text-slate-200 text-xs rounded px-2 py-1.5 border border-slate-700"
                      value={String(settings.paramsByMethod[settings.method]?.[param.name] ?? '')}
                      onChange={(event) =>
                        setSettings((previous) => ({
                          ...previous,
                          paramsByMethod: updateMethodParameter(
                            previous.paramsByMethod,
                            previous.method,
                            param.name,
                            event.target.value,
                            param.type
                          ),
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

      {result && <ResultFeedbackPanel feedback={forecastResultFeedback(result)} />}

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
        open={denoiseOpen}
        title="Forecast Denoising"
        value={settings.denoise}
        onClose={() => onDenoiseOpenChange(false)}
        onApply={(denoise) => {
          setSettings((previous) => ({
            ...previous,
            denoise,
          }))
          onDenoiseOpenChange(false)
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
  const runGateRef = useRef(createToolRunGate())
  const selectedMethod = useMemo(
    () => methods?.methods?.find((item) => item.method === method),
    [method, methods?.methods]
  )
  const effectiveProxy = volatilityProxyForMethod(selectedMethod, proxy)
  const requestBody = buildVolatilityRequest({
    symbol,
    timeframe,
    method,
    horizon,
    proxy,
    asOf: anchor ? formatDateTime(anchor) : undefined,
    methodInfo: selectedMethod,
  })
  const requestContract = JSON.stringify(requestBody)
  const requestContractRef = useRef(requestContract)
  requestContractRef.current = requestContract
  const resultMetrics = useMemo(
    () => result ? volatilityResultMetrics(result, horizon) : [],
    [horizon, result]
  )

  useEffect(() => {
    runGateRef.current.invalidate()
    setIsLoading(false)
    setError(null)
    setResult(null)
    return () => runGateRef.current.invalidate()
  }, [requestContract])

  const run = async () => {
    if (!symbol) return
    const runIdentity = runGateRef.current.begin(requestContract)
    setIsLoading(true)
    setError(null)
    setResult(null)
    try {
      const response = await forecastVolatility(requestBody)
      if (runGateRef.current.isCurrent(runIdentity, requestContractRef.current)) {
        setResult(response)
      }
    } catch (err) {
      if (runGateRef.current.isCurrent(runIdentity, requestContractRef.current)) {
        setError(getErrorMessage(err))
      }
    } finally {
      if (runGateRef.current.isCurrent(runIdentity, requestContractRef.current)) {
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

      <div className={`grid gap-3 ${selectedMethod?.requires_proxy ? 'grid-cols-2' : 'grid-cols-1'}`}>
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
        {selectedMethod?.requires_proxy && (
          <div>
            <label className="text-xs text-slate-400 mb-1 block">Proxy</label>
            <select
              className="w-full bg-slate-800 text-slate-200 text-sm rounded-lg px-3 py-2 border border-slate-700"
              value={effectiveProxy ?? ''}
              onChange={(event) => setProxy(event.target.value)}
            >
              {(selectedMethod.valid_proxies ?? []).map((value) => (
                <option key={value} value={value}>{value.replace(/_/g, ' ')}</option>
              ))}
            </select>
          </div>
        )}
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
        disabled={!symbol || !selectedMethod?.available || isLoading || (selectedMethod.requires_proxy && !effectiveProxy)}
      >
        {isLoading ? 'Running...' : 'Run Volatility Forecast'}
      </button>

      {result && (
        <div className="bg-slate-800/50 rounded-lg p-3 text-sm space-y-2">
          <div className="text-slate-400 text-xs mb-2">Result</div>
          {resultMetrics.map((metric) => (
            <div key={metric.label} className={metric.primary ? 'text-slate-100' : 'text-slate-400 text-xs'}>
              {metric.label}:{' '}
              <span className={`font-mono ${metric.primary ? 'text-sky-300 text-base' : 'text-slate-300'}`}>
                {metric.percent.toFixed(2)}%
              </span>
            </div>
          ))}
          {!resultMetrics.length && <div className="text-slate-400">Volatility unavailable</div>}
        </div>
      )}
    </div>
  )
}

function BacktestTab({
  symbol,
  timeframe,
  denoiseOpen,
  onDenoiseOpenChange,
}: {
  symbol: string
  timeframe: string
  denoiseOpen: boolean
  onDenoiseOpenChange: (open: boolean) => void
}) {
  const { methods, error: methodsError } = useForecastMethods()

  const [selectedMethods, setSelectedMethods] = useState<string[]>(['theta'])
  const [horizon, setHorizon] = useState(12)
  const [steps, setSteps] = useState(5)
  const [spacing, setSpacing] = useState(20)
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [denoise, setDenoise] = useState<DenoiseSpecUI | undefined>()
  const [sharedParams, setSharedParams] = useState<Record<string, unknown>>({})
  const [paramsByMethod, setParamsByMethod] = useState<
    Record<string, Record<string, unknown>>
  >({})
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<BacktestResult | null>(null)
  const runGateRef = useRef(createToolRunGate())
  const scopedParams = useMemo(
    () => scopedBacktestParams(selectedMethods, methods, sharedParams, paramsByMethod),
    [methods, paramsByMethod, selectedMethods, sharedParams]
  )
  const requestContract = JSON.stringify({
    symbol,
    timeframe,
    selectedMethods,
    horizon,
    steps,
    spacing,
    denoise,
    scopedParams,
  })
  const requestContractRef = useRef(requestContract)
  requestContractRef.current = requestContract

  const availableMethods = useMemo(() => methods.filter((method) => method.available), [methods])
  const resultRows = useMemo(() => backtestDisplayRows(result), [result])
  const sharedParamDefs = useMemo(
    () => sharedBacktestParamDefs(selectedMethods, methods),
    [methods, selectedMethods]
  )
  const configuredMethodOverrides = Object.values(scopedParams.params_per_method ?? {})
    .reduce((total, values) => total + Object.keys(values).length, 0)

  useEffect(() => {
    runGateRef.current.invalidate()
    setIsLoading(false)
    setError(null)
    setResult(null)
    return () => runGateRef.current.invalidate()
  }, [requestContract])

  const toggleMethod = (method: string) => {
    setSelectedMethods((previous) =>
      previous.includes(method) ? previous.filter((item) => item !== method) : [...previous, method]
    )
  }

  const run = async () => {
    if (!symbol || !selectedMethods.length) return
    const runIdentity = runGateRef.current.begin(requestContract)
    setIsLoading(true)
    setError(null)
    setResult(null)
    try {
      const response = await runBacktest({
        symbol,
        timeframe,
        horizon,
        steps,
        spacing,
        methods: selectedMethods,
        denoise,
        ...scopedParams,
      })
      if (runGateRef.current.isCurrent(runIdentity, requestContractRef.current)) {
        setResult(response)
      }
    } catch (err) {
      if (runGateRef.current.isCurrent(runIdentity, requestContractRef.current)) {
        setError(getErrorMessage(err))
      }
    } finally {
      if (runGateRef.current.isCurrent(runIdentity, requestContractRef.current)) {
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

      <button
        type="button"
        className="w-full text-left text-xs text-slate-400 hover:text-slate-300 flex items-center justify-between py-2 border-t border-slate-800"
        onClick={() => setShowAdvanced((value) => !value)}
      >
        <span>Advanced Options</span>
        <span>{showAdvanced ? '−' : '+'}</span>
      </button>

      <div className="space-y-1 rounded-lg border border-slate-800 bg-slate-900/40 px-3 py-2 text-xs text-slate-400">
        <div>
          Request configuration · {Object.keys(scopedParams.params ?? {}).length} shared ·{' '}
          {configuredMethodOverrides} override(s)
        </div>
        <div className="break-all font-mono text-[10px] text-slate-300">
          denoise={denoise ? JSON.stringify(denoise) : 'off'}
        </div>
        <div className="break-all font-mono text-[10px] text-slate-300">
          params={JSON.stringify(scopedParams.params ?? {})}
        </div>
        <div className="break-all font-mono text-[10px] text-slate-300">
          params_per_method={JSON.stringify(scopedParams.params_per_method ?? {})}
        </div>
      </div>

      {showAdvanced && (
        <div className="space-y-4 pb-3 border-b border-slate-800">
          <div className="flex items-center justify-between">
            <span className="text-xs text-slate-400">
              Backtest Denoise: <span className="text-slate-300">{denoise?.method || 'None'}</span>
            </span>
            <button
              type="button"
              className="text-xs text-sky-400 hover:text-sky-300"
              onClick={() => onDenoiseOpenChange(true)}
            >
              Configure
            </button>
          </div>

          <div>
            <div className="text-xs text-slate-300 mb-1">Shared Parameters</div>
            <p className="text-xs text-slate-500 mb-2">
              Applied to every selected method. Per-method values below take precedence.
            </p>
            {sharedParamDefs.length ? (
              <ParameterInputs
                definitions={sharedParamDefs}
                values={sharedParams}
                onChange={(definition, rawValue) =>
                  setSharedParams((previous) =>
                    updateParameterValue(
                      previous,
                      definition.name,
                      rawValue,
                      definition.type
                    )
                  )
                }
              />
            ) : (
              <p className="text-xs text-slate-500">
                No parameter is supported by every selected method.
              </p>
            )}
          </div>

          <div className="space-y-3">
            <div className="text-xs text-slate-300">Per-method Parameters</div>
            {selectedMethods.map((methodName) => {
              const method = methods.find((item) => item.method === methodName)
              if (!method?.params.length) return null
              return (
                <div key={methodName} className="rounded-lg border border-slate-800 p-2">
                  <div className="text-xs font-medium text-slate-300 mb-2">{methodName}</div>
                  <ParameterInputs
                    definitions={method.params}
                    values={paramsByMethod[methodName] ?? {}}
                    onChange={(definition, rawValue) =>
                      setParamsByMethod((previous) =>
                        updateMethodParameter(
                          previous,
                          methodName,
                          definition.name,
                          rawValue,
                          definition.type
                        )
                      )
                    }
                  />
                </div>
              )
            })}
            {!selectedMethods.some(
              (methodName) => methods.find((item) => item.method === methodName)?.params.length
            ) && <p className="text-xs text-slate-500">Selected methods have no parameters.</p>}
          </div>
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

      <button
        className="w-full bg-sky-600 hover:bg-sky-500 text-white text-sm font-medium py-2 rounded-lg disabled:opacity-50"
        onClick={run}
        disabled={!symbol || !selectedMethods.length || isLoading}
      >
        {isLoading ? 'Running Backtest...' : 'Run Backtest'}
      </button>

      {result && <ResultFeedbackPanel feedback={backtestResultFeedback(result)} />}

      {result && (
        <div className="bg-slate-800/50 rounded-lg overflow-hidden">
          <table className="w-full text-xs">
            <thead>
              <tr className="text-slate-400 border-b border-slate-700">
                <th className="text-left px-2 py-2">Method</th>
                <th className="text-right px-2 py-2">Anchors</th>
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
                const status = backtestMethodStatus(item)
                const warningMessages = (item.warnings ?? []).map(outputWarningMessage)
                return (
                <tr key={method} className="border-b border-slate-700/50">
                  <td className="px-2 py-1.5 text-slate-200 align-top">
                    <div className="flex flex-wrap items-center gap-1">
                      <span>{method}</span>
                      <span
                        className={`rounded px-1 py-0.5 text-[10px] uppercase tracking-wide ${
                          status === 'complete'
                            ? 'bg-emerald-950 text-emerald-300'
                            : status === 'partial'
                              ? 'bg-amber-950 text-amber-300'
                              : 'bg-rose-950 text-rose-300'
                        }`}
                      >
                        {status}
                      </span>
                      {item.ranking_status && item.ranking_status !== 'ranked' && (
                        <span className="text-[10px] text-slate-500">
                          {item.ranking_status.replace(/_/g, ' ')}
                        </span>
                      )}
                    </div>
                    {status === 'partial' && (
                      <div className="mt-0.5 text-[10px] text-amber-400">Metrics use an incomplete sample.</div>
                    )}
                    {item.error && (
                      <div className="mt-0.5 max-w-48 truncate text-[10px] text-rose-300" title={item.error}>
                        {item.error}
                      </div>
                    )}
                    {warningMessages.map((message) => (
                      <div key={message} className="mt-0.5 max-w-48 truncate text-[10px] text-amber-300" title={message}>
                        {message}
                      </div>
                    ))}
                  </td>
                  <td className="text-right px-2 py-1.5 text-slate-400 font-mono align-top">
                    {item.successful_tests ?? '-'}
                    {item.num_tests != null ? `/${item.num_tests}` : ''}
                    {(item.failed_tests ?? 0) > 0 && (
                      <div className="text-[10px] text-rose-400">{item.failed_tests} failed</div>
                    )}
                  </td>
                  <td className="text-right px-2 py-1.5 text-slate-300 font-mono">
                    {item.avg_mae?.toFixed(4) ?? '-'}
                  </td>
                  <td
                    className={`text-right px-2 py-1.5 font-mono ${
                      directionPercent == null
                        ? 'text-slate-500'
                        : directionPercent >= 60
                          ? 'text-emerald-400'
                          : directionPercent >= 50
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

      <DenoiseModal
        open={denoiseOpen}
        title="Backtest Denoising"
        value={denoise}
        onClose={() => onDenoiseOpenChange(false)}
        onApply={(value) => {
          setDenoise(value)
          onDenoiseOpenChange(false)
        }}
      />
    </div>
  )
}

function ResultFeedbackPanel({ feedback }: { feedback: ResultFeedback }) {
  const colors = {
    success: 'border-emerald-800 bg-emerald-950/50 text-emerald-200',
    warning: 'border-amber-800 bg-amber-950/50 text-amber-100',
    error: 'border-rose-800 bg-rose-950/50 text-rose-200',
  }
  return (
    <div
      className={`rounded-lg border px-3 py-2 text-xs ${colors[feedback.tone]}`}
      role={feedback.tone === 'error' ? 'alert' : 'status'}
    >
      <div className="font-medium">{feedback.summary}</div>
      {feedback.details.length > 0 && (
        <ul className="mt-1 list-disc space-y-0.5 pl-4">
          {feedback.details.map((detail) => <li key={detail}>{detail}</li>)}
        </ul>
      )}
    </div>
  )
}

function ParameterInputs({
  definitions,
  values,
  onChange,
}: {
  definitions: ParamDef[]
  values: Record<string, unknown>
  onChange: (definition: ParamDef, rawValue: string) => void
}) {
  return (
    <div className="grid grid-cols-2 gap-2">
      {definitions.map((definition) => (
        <label key={definition.name} className="block" title={definition.description || ''}>
          <span className="text-xs text-slate-500 mb-0.5 block">{definition.name}</span>
          <input
            className="w-full bg-slate-800 text-slate-200 text-xs rounded px-2 py-1.5 border border-slate-700"
            value={String(values[definition.name] ?? '')}
            onChange={(event) => onChange(definition, event.target.value)}
            placeholder={String(definition.default ?? '')}
          />
        </label>
      ))}
    </div>
  )
}
