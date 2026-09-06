import { useState, useEffect, useCallback, useMemo, useRef } from 'react'
import { useQuery } from '@tanstack/react-query'
import {
  getMethods,
  getPivots,
  getSupportResistance,
  forecastPrice,
  getErrorMessage,
} from '../api/client'
import type {
  HistoryBar,
  ForecastPayload,
  PivotLevel,
  SupportResistanceLevel,
  ChartOverlay,
  ForecastPriceBody,
  ParamDef,
} from '../types'
import { mapCompactForecastToSeries } from '../lib/compactForecast'
import { loadJSON, saveJSON } from '../lib/storage'
import { createToolRunGate } from '../lib/toolRunState'
import { formatDateTime } from '../lib/utils'
import {
  forecastMethodParams,
  normalizeForecastSettings,
  type ForecastSettings,
  type StoredForecastSettings,
} from '../lib/forecastContracts'
import {
  DEFAULT_PIVOT_METHOD,
  DEFAULT_SR_CONTROLS,
  buildPivotQuery,
  buildSupportResistanceQuery,
  normalizePivotMethod,
  normalizeSrControls,
  type PivotMethod,
  type SupportResistanceControls,
} from '../lib/overlayParams'
import { useToggleResource } from './useToggleResource'

type PivotResource = {
  levels: PivotLevel[]
  meta: { method: string; period?: { start?: string; end?: string } }
}

type SupportResistanceResource = {
  levels: SupportResistanceLevel[]
  meta: {
    method: string
    tolerance_pct: number
    min_touches: number
    lookback?: number
    window?: { start?: string | null; end?: string | null }
  }
}

const hasNoPivotLevels = (data: PivotResource) => !data.levels.length
const hasNoSupportResistanceLevels = (data: SupportResistanceResource) => !data.levels.length

// ============================================================================
// Forecast Methods Hook
// ============================================================================

export function useForecastMethods() {
  const query = useQuery({
    queryKey: ['methods'],
    queryFn: getMethods,
    staleTime: 60000,
  })

  return {
    methods: query.data?.methods ?? [],
    isLoading: query.isLoading,
    error: query.error ? getErrorMessage(query.error) : null,
    refetch: query.refetch,
  }
}

// ============================================================================
// Pivot Levels Hook
// ============================================================================

export function usePivotLevels(symbol: string, timeframe: string) {
  const [method, setMethodState] = useState<PivotMethod>(DEFAULT_PIVOT_METHOD)
  const methodRef = useRef(method)
  methodRef.current = method

  const fetchResource = useCallback(
    async (): Promise<PivotResource> => {
      const nextMethod = methodRef.current
      const data = await getPivots(buildPivotQuery(symbol, timeframe, nextMethod))
      const levels = (data.levels || [])
        .map((row) => ({ level: String(row.level), value: Number(row.value) }))
        .filter((row) => Number.isFinite(row.value))
      return { levels, meta: { method: data.method ?? nextMethod, period: data.period } }
    },
    [symbol, timeframe]
  )
  const resource = useToggleResource({
    enabled: Boolean(symbol),
    fetchResource,
    isEmpty: hasNoPivotLevels,
    emptyMessage: 'No pivot levels returned',
  })

  const setMethod = useCallback(
    async (next: string) => {
      const normalized = normalizePivotMethod(next)
      methodRef.current = normalized
      setMethodState(normalized)
      if (resource.data) {
        await resource.fetch()
      }
    },
    [resource]
  )

  return {
    levels: resource.data?.levels ?? null,
    meta: resource.data?.meta ?? null,
    method,
    setMethod,
    isLoading: resource.isLoading,
    error: resource.error,
    toggle: resource.toggle,
    reset: resource.reset,
  }
}

// ============================================================================
// Support/Resistance Hook
// ============================================================================

export function useSupportResistance(symbol: string, timeframe: string) {
  const [controls, setControlsState] = useState<SupportResistanceControls>(DEFAULT_SR_CONTROLS)
  const controlsRef = useRef(controls)
  controlsRef.current = controls

  const fetchResource = useCallback(
    async (): Promise<SupportResistanceResource> => {
      const nextControls = controlsRef.current
      const data = await getSupportResistance(
        buildSupportResistanceQuery(symbol, timeframe, nextControls)
      )
      const levels = [...(data.supports || []), ...(data.resistances || [])].filter((row) =>
        Number.isFinite(row?.value)
      )
      return {
        levels,
        meta: {
          method: data.method ?? 'swing',
          tolerance_pct: data.tolerance_pct ?? nextControls.tolerance_pct,
          min_touches: data.min_touches ?? nextControls.min_touches,
          lookback: data.lookback ?? nextControls.lookback,
          window: data.scan_window,
        },
      }
    },
    [symbol, timeframe]
  )
  const resource = useToggleResource({
    enabled: Boolean(symbol),
    fetchResource,
    isEmpty: hasNoSupportResistanceLevels,
    emptyMessage: 'No support/resistance levels detected',
  })

  const setControls = useCallback(
    async (partial: Partial<SupportResistanceControls>) => {
      const next = normalizeSrControls({ ...controlsRef.current, ...partial })
      controlsRef.current = next
      setControlsState(next)
      if (resource.data) {
        await resource.fetch()
      }
    },
    [resource]
  )

  return {
    levels: resource.data?.levels ?? null,
    meta: resource.data?.meta ?? null,
    controls,
    setControls,
    isLoading: resource.isLoading,
    error: resource.error,
    toggle: resource.toggle,
    reset: resource.reset,
  }
}

// ============================================================================
// Forecast State Hook
// ============================================================================

export type { ForecastSettings } from '../lib/forecastContracts'

export function loadForecastSettings(symbol: string, timeframe: string): ForecastSettings {
  if (!symbol || !timeframe) return normalizeForecastSettings()
  const storageKey = `fc:${symbol}:${timeframe}`
  const legacyStorageKey = `fc2:${symbol}:${timeframe}`
  const saved =
    loadJSON<StoredForecastSettings>(storageKey) ??
    loadJSON<StoredForecastSettings>(legacyStorageKey)
  return normalizeForecastSettings(saved)
}

export function useForecastSettings(symbol: string, timeframe: string) {
  const [settings, setSettings] = useState<ForecastSettings>(() =>
    loadForecastSettings(symbol, timeframe)
  )

  const storageKey = symbol && timeframe ? `fc:${symbol}:${timeframe}` : null

  useEffect(() => {
    if (!storageKey) return
    saveJSON(storageKey, settings)
  }, [storageKey, settings])

  return { settings, setSettings }
}

// ============================================================================
// Forecast Execution Hook
// ============================================================================

export function useForecast(
  symbol: string,
  timeframe: string,
  settings: ForecastSettings,
  onResult: (payload: ForecastPayload | null) => void,
  anchor?: number,
  methodParamDefs: ParamDef[] = []
) {
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<ForecastPayload | null>(null)
  const runGateRef = useRef(createToolRunGate())
  const onResultRef = useRef(onResult)
  const requestKey = JSON.stringify({
    symbol,
    timeframe,
    settings,
    anchor,
    methodParamDefs: methodParamDefs.map(({ name, type }) => ({ name, type })),
  })
  const requestKeyRef = useRef(requestKey)
  requestKeyRef.current = requestKey

  useEffect(() => {
    onResultRef.current = onResult
  }, [onResult])

  useEffect(() => {
    runGateRef.current.invalidate()
    setIsLoading(false)
    setError(null)
    setResult(null)
    onResultRef.current(null)
    return () => runGateRef.current.invalidate()
  }, [requestKey])

  const run = useCallback(
    async (kind: 'full' | 'partial', anchor?: number) => {
      if (!symbol) return

      const runIdentity = runGateRef.current.begin(requestKey)
      setIsLoading(true)
      setError(null)
      setResult(null)
      onResultRef.current(null)

      try {
        const body: ForecastPriceBody = {
          symbol,
          timeframe,
          method: settings.method,
          horizon: settings.horizon,
          lookback: settings.lookback === '' ? undefined : Number(settings.lookback),
          ci_alpha: settings.ci_alpha,
          quantity: settings.quantity,
          as_of: kind === 'full' ? undefined : anchor ? formatDateTime(anchor) : undefined,
          params: forecastMethodParams(settings, methodParamDefs),
          denoise: settings.denoise,
        }

        const res = await forecastPrice(body)
        const payload: ForecastPayload = {
          ...res,
          __anchor: kind === 'full' ? undefined : anchor,
          __kind: kind,
        }

        if (!runGateRef.current.isCurrent(runIdentity, requestKeyRef.current)) return
        try {
          const series = mapCompactForecastToSeries(payload.forecast ?? [])
          if (!series.values.length) {
            throw new Error('compact forecast has no chart points')
          }
        } catch {
          setError(
            settings.quantity === 'return'
              ? 'The return forecast did not include a reconstructed price path for the price chart.'
              : 'The forecast did not include a finite price path for the chart.'
          )
          return
        }
        setResult(payload)
        onResultRef.current(payload)
      } catch (err) {
        if (runGateRef.current.isCurrent(runIdentity, requestKeyRef.current)) {
          setError(getErrorMessage(err))
        }
      } finally {
        if (runGateRef.current.isCurrent(runIdentity, requestKeyRef.current)) {
          setIsLoading(false)
        }
      }
    },
    [methodParamDefs, requestKey, settings, symbol, timeframe]
  )

  return { run, isLoading, error, result }
}

// ============================================================================
// Chart Overlays Builder
// ============================================================================

export function useChartOverlays(
  bars: HistoryBar[],
  forecastOverlays: ChartOverlay[],
): ChartOverlay[] {
  return useMemo(() => {
    const map = new Map<string, ChartOverlay>()

    const addOverlay = (ov: ChartOverlay) => {
      if (!ov?.name || !Array.isArray(ov.points)) return
      map.set(ov.name, ov)
    }

    forecastOverlays.forEach(addOverlay)

    if (bars.some((bar) => Number.isFinite(bar.close_dn))) {
      const dnPoints = bars
        .filter((bar): bar is HistoryBar & { close_dn: number } => 
          Number.isFinite(bar.time) && Number.isFinite(bar.close_dn)
        )
        .map(bar => ({ time: bar.time, value: bar.close_dn }))

      if (dnPoints.length) {
        addOverlay({
          name: 'denoise:close',
          points: dnPoints,
          color: '#f59e0b',
          lineWidth: 2,
          label: 'Close · filtered',
        })
      }
    }

    return Array.from(map.values())
  }, [bars, forecastOverlays])
}
