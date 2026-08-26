import { useCallback, useMemo, useRef, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { getErrorMessage, getHistory, getTick } from '../../api/client'
import { useConfluenceLevels, useExposureOverlay, useVolumeProfileLevels } from '../../hooks/useGeometry'
import { useChartOverlays, usePivotLevels, useSupportResistance } from '../../hooks/useForecast'
import { ensureChartDenoiseCausality } from '../../lib/denoiseSpec'
import {
  buildIndicatorOverlays,
  buildIndicatorsQuery,
  chartIndicatorsActive,
  DEFAULT_CHART_INDICATORS,
  historyOhlcvForIndicators,
  missingIndicatorMessages,
  normalizeChartIndicators,
  type ChartIndicatorSelection,
} from '../../lib/indicatorSpec'
import { loadJSON, saveJSON } from '../../lib/storage'
import { mapCompactForecastToSeries } from '../../lib/compactForecast'
import { toUtcSec } from '../../lib/time'
import { chartWorkspaceLivePollMs } from '../../lib/timeframes'
import { formatDateTime } from '../../lib/utils'
import {
  confluencePriceLines,
  exposurePriceLines,
  ideaGeometryPriceLines,
  pivotPriceLines,
  supportResistancePriceLines,
  volumeProfilePriceLines,
} from '../../lib/geometryOverlays'
import type {
  AnchorMetrics,
  ChartOverlay,
  DenoiseSpecUI,
  ForecastPayload,
  HistoryBar,
  TradeIdeaPayload,
} from '../../types'
import type { PriceLineSpec } from '../../components/OHLCChart'

export type TimezoneMode = 'utc' | 'local' | 'server'

const QUERY_LIMIT = 1000

function loadChartDenoise(symbol: string, timeframe: string): DenoiseSpecUI | undefined {
  if (!symbol || !timeframe) return undefined
  const saved = loadJSON<DenoiseSpecUI | undefined>(`chart_dn:${symbol}:${timeframe}`)
  const normalized = ensureChartDenoiseCausality(saved || undefined)
  if (normalized && (!saved?.causality || saved.causality !== normalized.causality)) {
    saveJSON(`chart_dn:${symbol}:${timeframe}`, normalized)
  }
  return normalized
}

function loadChartIndicators(symbol: string, timeframe: string): ChartIndicatorSelection {
  if (!symbol || !timeframe) return DEFAULT_CHART_INDICATORS
  return normalizeChartIndicators(loadJSON(`chart_ti:${symbol}:${timeframe}`))
}

export function useChartWorkspace() {
  const [symbol, setSymbol] = useState(() => loadJSON<string>('last_symbol') || '')
  const [timeframe, setTimeframe] = useState('H1')
  const [extraHistory, setExtraHistory] = useState<HistoryBar[]>([])
  const [isLoadingMore, setIsLoadingMore] = useState(false)
  const [anchor, setAnchor] = useState<number | undefined>(undefined)
  const [showBid, setShowBid] = useState(false)
  const [showAsk, setShowAsk] = useState(false)
  const [showLast, setShowLast] = useState(true)
  const [isLive, setIsLive] = useState(true)
  const [timezoneMode, setTimezoneMode] = useState<TimezoneMode>('local')
  const [forecastOverlays, setForecastOverlays] = useState<ChartOverlay[]>([])
  const [chartDenoise, setChartDenoise] = useState<DenoiseSpecUI | undefined>(() =>
    loadChartDenoise(symbol, timeframe)
  )
  const [chartIndicators, setChartIndicators] = useState<ChartIndicatorSelection>(() =>
    loadChartIndicators(symbol, timeframe)
  )
  const [metrics, setMetrics] = useState<AnchorMetrics | null>(null)
  const [historyPageError, setHistoryPageError] = useState<string | null>(null)
  const [ideaGeometry, setIdeaGeometry] = useState<TradeIdeaPayload['geometry'] | null>(null)
  const indicatorsQuery = buildIndicatorsQuery(chartIndicators)
  const indicatorsOhlcv = historyOhlcvForIndicators(chartIndicators)
  const historyContract = JSON.stringify({
    symbol,
    timeframe,
    denoise: chartDenoise ?? null,
    indicators: indicatorsQuery ?? null,
    ohlcv: indicatorsOhlcv ?? null,
  })
  const historyContractRef = useRef(historyContract)
  historyContractRef.current = historyContract

  const pivotState = usePivotLevels(symbol, timeframe)
  const srState = useSupportResistance(symbol, timeframe)
  const confluenceState = useConfluenceLevels(symbol)
  const volumeProfileState = useVolumeProfileLevels(symbol, timeframe)
  const exposureState = useExposureOverlay(symbol)
  const livePollMs = chartWorkspaceLivePollMs(timeframe)

  const {
    data: histDataResponse,
    error: historyError,
    refetch,
    isFetching,
    isLoading: isHistoryLoading,
    isFetched: isHistoryFetched,
  } = useQuery({
    queryKey: ['hist', symbol, timeframe, QUERY_LIMIT, JSON.stringify(chartDenoise || {}), indicatorsQuery, indicatorsOhlcv, isLive],
    queryFn: ({ signal }) =>
      getHistory({
        symbol,
        timeframe,
        limit: QUERY_LIMIT,
        denoise: chartDenoise,
        include_incomplete: isLive,
        indicators: indicatorsQuery,
        ohlcv: indicatorsOhlcv,
      }, signal),
    enabled: !!symbol,
  })

  const { data: liveDataResponse, error: liveHistoryError } = useQuery({
    queryKey: ['hist-live', symbol, timeframe, JSON.stringify(chartDenoise || {}), indicatorsQuery, indicatorsOhlcv],
    queryFn: ({ signal }) => getHistory({
      symbol,
      timeframe,
      limit: 2,
      denoise: chartDenoise,
      include_incomplete: true,
      indicators: indicatorsQuery,
      ohlcv: indicatorsOhlcv,
    }, signal),
    enabled: isLive && !!symbol,
    refetchInterval: livePollMs,
  })

  const { data: tickData, error: tickError } = useQuery({
    queryKey: ['tick', symbol],
    queryFn: ({ signal }) => getTick(symbol, signal),
    enabled: !!symbol,
    refetchInterval: livePollMs,
  })

  const bars = useMemo(() => {
    const base = (histDataResponse?.data ?? []) as HistoryBar[]
    const live = (liveDataResponse?.data ?? []) as HistoryBar[]

    let combined = base

    if (extraHistory.length) {
      const mainStart = base.length ? base[0].time : Infinity
      const older = extraHistory.filter((bar) => bar.time < mainStart)
      combined = [...older, ...base]
    }

    if (!isLive || !live.length || !combined.length) return combined

    const merged = [...combined]
    live.forEach((bar) => {
      const lastIndex = merged.length - 1
      if (lastIndex < 0) {
        merged.push(bar)
        return
      }
      const last = merged[lastIndex]
      if (Math.abs(bar.time - last.time) < 0.1) {
        merged[lastIndex] = bar
      } else if (bar.time > last.time) {
        merged.push(bar)
      }
    })
    return merged
  }, [extraHistory, histDataResponse, isLive, liveDataResponse])

  const serverTimeZone = useMemo(() => {
    const candidate = histDataResponse?.server_timezone
      ?? histDataResponse?.meta?.runtime?.timezone?.server?.tz
    if (!candidate) return undefined
    try {
      new Intl.DateTimeFormat('en-US', { timeZone: candidate }).format(0)
      return candidate
    } catch {
      return undefined
    }
  }, [histDataResponse])
  const localTimeZone = Intl.DateTimeFormat().resolvedOptions().timeZone || 'UTC'
  const displayTimeZone = timezoneMode === 'server'
    ? serverTimeZone ?? 'UTC'
    : timezoneMode === 'local'
      ? localTimeZone
      : 'UTC'
  const handleAnchorSelect = useCallback(
    (utcTime: number) => {
      setAnchor(utcTime)
      setForecastOverlays([])
      setMetrics(null)
    },
    []
  )

  const resetWorkspaceView = useCallback(() => {
    setExtraHistory([])
    setForecastOverlays([])
    setAnchor(undefined)
    setMetrics(null)
    setHistoryPageError(null)
    pivotState.reset()
    srState.reset()
    confluenceState.reset()
    volumeProfileState.reset()
    exposureState.reset()
    setIdeaGeometry(null)
  }, [confluenceState, exposureState, pivotState, srState, volumeProfileState])

  const handleSymbolChange = useCallback(
    (newSymbol: string) => {
      setSymbol(newSymbol)
      setChartDenoise(loadChartDenoise(newSymbol, timeframe))
      setChartIndicators(loadChartIndicators(newSymbol, timeframe))
      resetWorkspaceView()
      saveJSON('last_symbol', newSymbol)

      if (!newSymbol) return

      const recent = loadJSON<string[]>('recent_symbols') || []
      const updated = [newSymbol, ...recent.filter((item) => item !== newSymbol)].slice(0, 10)
      saveJSON('recent_symbols', updated)
    },
    [resetWorkspaceView, timeframe]
  )

  const handleTimeframeChange = useCallback(
    (newTimeframe: string) => {
      setTimeframe(newTimeframe)
      setChartDenoise(loadChartDenoise(symbol, newTimeframe))
      setChartIndicators(loadChartIndicators(symbol, newTimeframe))
      resetWorkspaceView()
    },
    [resetWorkspaceView, symbol]
  )

  const handleNeedMoreLeft = useCallback(
    async (earliestDisplayTime: number) => {
      if (!symbol || isLoadingMore || isFetching) return
      const requestedContract = historyContract
      setIsLoadingMore(true)
      setHistoryPageError(null)
      try {
        const utcTime = earliestDisplayTime
      const before = formatDateTime(utcTime - 1)
        const older = await getHistory({
          symbol,
          timeframe,
          limit: QUERY_LIMIT,
          end: before,
          denoise: chartDenoise,
          indicators: indicatorsQuery,
          ohlcv: indicatorsOhlcv,
        })
        if (historyContractRef.current === requestedContract && older.data.length) {
          setExtraHistory((previous) => [...older.data, ...previous])
        }
      } catch (error) {
        if (historyContractRef.current === requestedContract) {
          setHistoryPageError(getErrorMessage(error))
        }
      } finally {
        setIsLoadingMore(false)
      }
    },
    [chartDenoise, historyContract, indicatorsOhlcv, indicatorsQuery, isFetching, isLoadingMore, symbol, timeframe]
  )

  const handleDenoiseChange = useCallback(
    (denoise?: DenoiseSpecUI) => {
      const normalized = ensureChartDenoiseCausality(denoise)
      setChartDenoise(normalized)
      setExtraHistory([])
      setHistoryPageError(null)
      setForecastOverlays([])
      setMetrics(null)
      if (symbol && timeframe) {
        saveJSON(`chart_dn:${symbol}:${timeframe}`, normalized)
      }
    },
    [symbol, timeframe]
  )

  const handleIndicatorsChange = useCallback(
    (next: ChartIndicatorSelection) => {
      const normalized = normalizeChartIndicators(next)
      setChartIndicators(normalized)
      setExtraHistory([])
      setHistoryPageError(null)
      if (symbol && timeframe) {
        saveJSON(`chart_ti:${symbol}:${timeframe}`, normalized)
      }
    },
    [symbol, timeframe]
  )

  const handleForecastResult = useCallback(
    (result: ForecastPayload | null) => {
      if (!result) {
        setForecastOverlays([])
        setMetrics(null)
        return
      }
      let series
      try {
        series = mapCompactForecastToSeries(result.forecast ?? [])
      } catch {
        setForecastOverlays([])
        setMetrics(null)
        return
      }
      const { times, values: main, lower, upper } = series
      if (!main.length) {
        setForecastOverlays([])
        setMetrics(null)
        return
      }

      const overlays: ChartOverlay[] = [
        {
          name: 'forecast',
          points: times.map((time, index) => ({ time, value: main[index] })),
          color: '#60a5fa',
          lineWidth: 2,
        },
      ]

      if (lower && upper) {
        overlays.push({
          name: 'lower',
          points: times.map((time, index) => ({ time, value: lower[index] })),
          color: '#64748b',
          lineStyle: 'dashed',
        })
        overlays.push({
          name: 'upper',
          points: times.map((time, index) => ({ time, value: upper[index] })),
          color: '#64748b',
          lineStyle: 'dashed',
        })
      }
      setForecastOverlays(overlays)

      if (result.__kind === 'partial' && result.__anchor !== undefined && bars.length) {
        const closeByTime = new Map<number, number>()
        for (const bar of bars) closeByTime.set(Math.floor(bar.time), bar.close)

        const yPred: number[] = []
        const yAct: number[] = []
        const alignedTimes: number[] = []
        for (let index = 0; index < times.length; index += 1) {
          const actual = closeByTime.get(Math.floor(times[index]))
          if (actual !== undefined && Number.isFinite(main[index])) {
            yPred.push(Number(main[index]))
            yAct.push(Number(actual))
            alignedTimes.push(times[index])
          }
        }

        if (yPred.length) {
          const n = yPred.length
          const diffs = yPred.map((prediction, index) => prediction - yAct[index])
          const mae = diffs.reduce((total, diff) => total + Math.abs(diff), 0) / n
          const mape =
            (yPred.reduce((total, _, index) => {
              const denom = Math.abs(yAct[index]) || 1
              return total + Math.abs((yPred[index] - yAct[index]) / denom)
            }, 0) /
              n) *
            100
          const rmse = Math.sqrt(diffs.reduce((total, diff) => total + diff * diff, 0) / n)
          const firstForecastTime = alignedTimes[0]
          const backendBaselineTime = result.data_window?.last_observation
          let baselineClose: number | undefined
          if (backendBaselineTime !== undefined) {
            const baselineEpoch = toUtcSec(backendBaselineTime)
            baselineClose = closeByTime.get(Math.floor(baselineEpoch))
          }
          if (baselineClose === undefined) {
            for (let index = bars.length - 1; index >= 0; index -= 1) {
              const bar = bars[index]
              if (bar.time < firstForecastTime && Number.isFinite(bar.close)) {
                baselineClose = bar.close
                break
              }
            }
          }
          if (baselineClose === undefined) {
            setMetrics(null)
            return
          }

          let correct = 0
          for (let index = 0; index < n; index += 1) {
            const previous = index === 0 ? baselineClose : yAct[index - 1]
            if (Math.sign(yPred[index] - previous) === Math.sign(yAct[index] - previous)) {
              correct += 1
            }
          }
          setMetrics({ overlap: n, mae, mape, rmse, dirAcc: (correct / n) * 100 })
        } else {
          setMetrics(null)
        }
      } else {
        setMetrics(null)
      }
    },
    [bars]
  )

  const indicatorOverlays = useMemo(
    () => buildIndicatorOverlays(bars, chartIndicators),
    [bars, chartIndicators]
  )
  const forecastAndIndicatorOverlays = useMemo(
    () => [...forecastOverlays, ...indicatorOverlays],
    [forecastOverlays, indicatorOverlays]
  )

  const chartOverlays = useChartOverlays(
    bars,
    forecastAndIndicatorOverlays,
  )

  const priceLines: PriceLineSpec[] = useMemo(() => {
    if (!tickData) return []

    const lines: PriceLineSpec[] = []
    if (showBid) lines.push({ price: tickData.bid, color: '#ef4444', title: 'Bid' })
    if (showAsk) lines.push({ price: tickData.ask, color: '#22c55e', title: 'Ask' })

    if (showLast) {
      let lastPrice = tickData.last
      if (!lastPrice && bars.length > 0) {
        lastPrice = bars[bars.length - 1].close
      }
      if (lastPrice && lastPrice > 0) {
        lines.push({ price: lastPrice, color: '#facc15', title: 'Last' })
      }
    }

    return [
      ...lines,
      ...pivotPriceLines(pivotState.levels),
      ...supportResistancePriceLines(srState.levels),
      ...confluencePriceLines(confluenceState.data?.levels),
      ...volumeProfilePriceLines(volumeProfileState.data),
      ...ideaGeometryPriceLines(ideaGeometry),
      ...exposurePriceLines(exposureState.data?.positions, exposureState.data?.pending),
    ]
  }, [
    bars,
    confluenceState.data,
    exposureState.data,
    ideaGeometry,
    pivotState.levels,
    showAsk,
    showBid,
    showLast,
    srState.levels,
    tickData,
    volumeProfileState.data,
  ])

  const earliest = bars.length ? bars[0].time : undefined
  const workspaceErrors = useMemo(() => {
    const errors = [
      historyError ? `History: ${getErrorMessage(historyError)}` : null,
      isLive && liveHistoryError
        ? `Live history: ${getErrorMessage(liveHistoryError)}`
        : null,
      tickError ? `Quote: ${getErrorMessage(tickError)}` : null,
      historyPageError ? `Older history: ${historyPageError}` : null,
      pivotState.error ? `Pivots: ${pivotState.error}` : null,
      srState.error ? `Support/resistance: ${srState.error}` : null,
      confluenceState.error ? `Confluence: ${confluenceState.error}` : null,
      volumeProfileState.error ? `Volume profile: ${volumeProfileState.error}` : null,
      exposureState.error ? `Exposure: ${exposureState.error}` : null,
      timezoneMode === 'server' && !serverTimeZone
        ? 'Exchange timezone unavailable; configure an IANA MT5_SERVER_TZ value.'
        : null,
      ...missingIndicatorMessages(
        chartIndicators,
        histDataResponse?.indicator_columns,
        bars
      ),
    ]
    return errors.filter((value): value is string => Boolean(value))
  }, [
    bars,
    chartIndicators,
    histDataResponse?.indicator_columns,
    historyError,
    historyPageError,
    isLive,
    liveHistoryError,
    confluenceState.error,
    exposureState.error,
    pivotState.error,
    serverTimeZone,
    srState.error,
    tickError,
    timezoneMode,
    volumeProfileState.error,
  ])

  return {
    symbol,
    timeframe,
    anchor,
    showBid,
    showAsk,
    showLast,
    isLive,
    timezoneMode,
    displayTimeZone,
    chartDenoise,
    chartIndicators,
    indicatorsActive: chartIndicatorsActive(chartIndicators),
    bars,
    chartOverlays,
    priceLines,
    metrics,
    pivotLevels: pivotState.levels,
    pivotMethod: pivotState.method,
    pivotsLoading: pivotState.isLoading,
    srLevels: srState.levels,
    srControls: srState.controls,
    srLoading: srState.isLoading,
    isFetching,
    isLoadingMore,
    /** True until the primary history query has settled at least once for the current key. */
    isInitialHistoryLoading: !!symbol && (isHistoryLoading || (!isHistoryFetched && isFetching)),
    historyErrorMessage: historyError ? getErrorMessage(historyError) : null,
    workspaceErrors,
    earliest,
    setTimezoneMode,
    handleAnchorSelect,
    handleSymbolChange,
    handleTimeframeChange,
    handleNeedMoreLeft,
    handleDenoiseChange,
    handleIndicatorsChange,
    handleForecastResult,
    handlePivotToggle: pivotState.toggle,
    handlePivotMethodChange: pivotState.setMethod,
    handleSRToggle: srState.toggle,
    handleSrControlsChange: srState.setControls,
    handleConfluenceToggle: confluenceState.toggle,
    handleVolumeProfileToggle: volumeProfileState.toggle,
    handleExposureToggle: exposureState.toggle,
    handleIdeaResult: (idea: TradeIdeaPayload | null) => {
      setIdeaGeometry(idea?.geometry ?? null)
    },
    confluenceLevels: confluenceState.data?.levels,
    confluenceLoading: confluenceState.isLoading,
    volumeProfile: volumeProfileState.data,
    volumeProfileLoading: volumeProfileState.isLoading,
    exposure: exposureState.data,
    exposureLoading: exposureState.isLoading,
    reload: () => {
      setExtraHistory([])
      void refetch()
    },
    toggleBid: () => setShowBid((value) => !value),
    toggleAsk: () => setShowAsk((value) => !value),
    toggleLast: () => setShowLast((value) => !value),
    toggleLive: () => setIsLive((value) => !value),
    clearAnchor: () => {
      setAnchor(undefined)
      setForecastOverlays([])
      setMetrics(null)
    },
  }
}
