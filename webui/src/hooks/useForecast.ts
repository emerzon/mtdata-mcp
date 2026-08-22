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
  DenoiseSpecUI,
  PivotLevel,
  SupportResistanceLevel,
  ChartOverlay,
  ForecastPriceBody,
} from '../types'
import { loadJSON, saveJSON } from '../lib/storage'
import { tfSeconds } from '../lib/timeframes'
import { formatDateTime } from '../lib/utils'
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

const DIMRED_METHODS = new Set([
  'mlf_rf', 'mlf_lightgbm', 'nhits', 'nbeatsx', 'tft', 'patchtst',
  'chronos_bolt', 'timesfm', 'ensemble',
])

export type ForecastSettings = {
  method: string
  horizon: number
  lookback: number | ''
  quantity: 'price' | 'return'
  ci_alpha: number
  params: Record<string, unknown>
  denoise?: DenoiseSpecUI
  dimredMethod?: string
  dimredParams?: Record<string, unknown>
}

export function useForecastSettings(symbol: string, timeframe: string) {
  const [settings, setSettings] = useState<ForecastSettings>({
    method: 'theta',
    horizon: 12,
    lookback: '',
    quantity: 'price',
    ci_alpha: 0.1,
    params: {},
  })

  const storageKey = symbol && timeframe ? `fc:${symbol}:${timeframe}` : null
  const legacyStorageKey = symbol && timeframe ? `fc2:${symbol}:${timeframe}` : null

  // Load saved settings when symbol/timeframe changes
  useEffect(() => {
    if (!storageKey) return
    const saved =
      loadJSON<
        Partial<ForecastSettings> & {
          methodParams?: Record<string, unknown>
        }
      >(storageKey) ??
      (legacyStorageKey
        ? loadJSON<
            Partial<ForecastSettings> & {
              methodParams?: Record<string, unknown>
            }
          >(legacyStorageKey)
        : null)
    if (saved) {
      setSettings(prev => ({
        ...prev,
        method: saved.method ?? prev.method,
        horizon: saved.horizon ?? prev.horizon,
        lookback: saved.lookback ?? prev.lookback,
        quantity: saved.quantity ?? prev.quantity,
        ci_alpha: saved.ci_alpha ?? prev.ci_alpha,
        params: saved.params ?? saved.methodParams ?? prev.params,
        denoise: saved.denoise,
        dimredMethod: saved.dimredMethod,
        dimredParams: saved.dimredParams,
      }))
    }
  }, [legacyStorageKey, storageKey])

  // Save settings when they change
  useEffect(() => {
    if (!storageKey) return
    saveJSON(storageKey, settings)
  }, [storageKey, settings])

  const supportsDimred = DIMRED_METHODS.has(settings.method)

  // Clear dimred when method doesn't support it
  useEffect(() => {
    if (!supportsDimred && (settings.dimredMethod || settings.dimredParams)) {
      setSettings(prev => ({ ...prev, dimredMethod: undefined, dimredParams: undefined }))
    }
  }, [supportsDimred, settings.dimredMethod, settings.dimredParams])

  return { settings, setSettings, supportsDimred }
}

// ============================================================================
// Forecast Execution Hook
// ============================================================================

export function useForecast(
  symbol: string,
  timeframe: string,
  settings: ForecastSettings,
  onResult: (payload: ForecastPayload | null) => void,
  anchor?: number
) {
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const runId = useRef(0)
  const onResultRef = useRef(onResult)
  const requestKey = JSON.stringify({ symbol, timeframe, settings, anchor })
  const requestKeyRef = useRef(requestKey)
  requestKeyRef.current = requestKey

  useEffect(() => {
    onResultRef.current = onResult
  }, [onResult])

  useEffect(() => {
    runId.current += 1
    setIsLoading(false)
    setError(null)
    onResultRef.current(null)
  }, [requestKey])

  const run = useCallback(
    async (kind: 'full' | 'partial', anchor?: number) => {
      if (!symbol) return

      const runRequestKey = requestKey
      const currentRunId = ++runId.current
      setIsLoading(true)
      setError(null)
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
          params: Object.keys(settings.params).length ? settings.params : undefined,
          denoise: settings.denoise,
          dimred: settings.dimredMethod
            ? { method: settings.dimredMethod, params: settings.dimredParams }
            : undefined,
        }

        const res = await forecastPrice(body)
        const payload: ForecastPayload = {
          ...res,
          __anchor: kind === 'full' ? undefined : anchor,
          __kind: kind,
        }

        if (currentRunId !== runId.current || runRequestKey !== requestKeyRef.current) return
        const compactRows = payload.forecast ?? []
        const hasChartPrice =
          payload.forecast_price?.some(Number.isFinite) ||
          compactRows.some((row) => Number.isFinite(row.price ?? row.value))
        if (!hasChartPrice) {
          setError(
            settings.quantity === 'return'
              ? 'The return forecast did not include a reconstructed price path for the price chart.'
              : 'The forecast did not include a finite price path for the chart.'
          )
          return
        }
        onResultRef.current(payload)
      } catch (err) {
        if (currentRunId === runId.current && runRequestKey === requestKeyRef.current) {
          setError(getErrorMessage(err))
        }
      } finally {
        if (currentRunId === runId.current && runRequestKey === requestKeyRef.current) {
          setIsLoading(false)
        }
      }
    },
    [requestKey, settings, symbol, timeframe]
  )

  return { run, isLoading, error }
}

// ============================================================================
// Chart Overlays Builder
// ============================================================================

export function useChartOverlays(
  bars: HistoryBar[],
  forecastOverlays: ChartOverlay[],
  pivotLevels: PivotLevel[] | null,
  srLevels: SupportResistanceLevel[] | null,
  timeframe: string
): ChartOverlay[] {
  return useMemo(() => {
    const map = new Map<string, ChartOverlay>()

    const addOverlay = (ov: ChartOverlay) => {
      if (!ov?.name || !Array.isArray(ov.points)) return
      map.set(ov.name, ov)
    }

    // Add forecast overlays
    forecastOverlays.forEach(addOverlay)

    // Calculate time boundaries
    const startTime = bars.length ? bars[0].time : undefined
    const lastBarTime = bars.length ? bars[bars.length - 1].time : undefined
    const tfStep = tfSeconds(timeframe) || 0
    const fallbackStep = tfStep || (bars.length >= 2 ? Math.max(1, bars[1].time - bars[0].time) : 60)

    let maxTime = lastBarTime
    forecastOverlays.forEach(ov => {
      ov.points?.forEach(pt => {
        if (pt?.time !== undefined && Number.isFinite(pt.time)) {
          maxTime = maxTime === undefined ? pt.time : Math.max(maxTime, pt.time)
        }
      })
    })
    const lineEnd = maxTime !== undefined ? maxTime + fallbackStep : undefined

    // Add denoised line if present
    if (bars.some((bar) => Number.isFinite(bar.close_dn))) {
      const dnPoints = bars
        .filter((bar): bar is HistoryBar & { close_dn: number } => 
          Number.isFinite(bar.time) && Number.isFinite(bar.close_dn)
        )
        .map(bar => ({ time: bar.time, value: bar.close_dn }))

      if (dnPoints.length) {
        addOverlay({ name: 'denoise:close', points: dnPoints, color: '#f59e0b', lineWidth: 2 })
      }
    }

    // Add pivot levels
    if (pivotLevels?.length && startTime !== undefined && lineEnd !== undefined) {
      const colorForLevel = (level: string) => {
        if (level.startsWith('R')) return '#f97316'
        if (level.startsWith('S')) return '#38bdf8'
        return '#facc15'
      }

      pivotLevels.forEach(level => {
        if (!Number.isFinite(level.value)) return
        addOverlay({
          name: `pivot-${level.level}`,
          points: [
            { time: startTime, value: level.value },
            { time: lineEnd, value: level.value },
          ],
          color: colorForLevel(level.level),
          lineStyle: 'dashed',
          lineWidth: 1,
          label: level.level,
        })
      })
    }

    // Add support/resistance levels
    if (srLevels?.length && startTime !== undefined && lineEnd !== undefined) {
      srLevels.forEach((level, idx) => {
        if (!Number.isFinite(level?.value)) return
        const color = level.type === 'resistance' ? '#f87171' : '#34d399'
        addOverlay({
          name: `sr-${level.type}-${idx}`,
          points: [
            { time: startTime, value: level.value },
            { time: lineEnd, value: level.value },
          ],
          color,
          lineWidth: 2,
          lineStyle: 'dotted',
          label: `${level.type === 'resistance' ? 'Res' : 'Sup'} (${level.touches})`,
        })
      })
    }

    return Array.from(map.values())
  }, [bars, forecastOverlays, pivotLevels, srLevels, timeframe])
}
