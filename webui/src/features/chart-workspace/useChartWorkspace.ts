import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
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
import {
  computeAnchorForecastMetrics,
  forecastResultFeedback,
  mapCompactForecastToSeries,
} from '../../lib/compactForecast'
import { toUtcSec } from '../../lib/time'
import { chartWorkspaceLivePollMs } from '../../lib/timeframes'
import { liveQuotePriceLines } from '../../lib/chartPriceLines'
import { chartQueryActivity, mergeHistoryBars } from '../../lib/historyBars'
import {
  resolveChartDenoiseFeedback,
  responseWarningMessages,
} from '../../lib/historyFeedback'
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
  const [liveHistory, setLiveHistory] = useState<HistoryBar[]>([])
  const [isLoadingMore, setIsLoadingMore] = useState(false)
  const [anchor, setAnchor] = useState<number | undefined>(undefined)
  const [showBid, setShowBid] = useState(false)
  const [showAsk, setShowAsk] = useState(false)
  const [showLast, setShowLast] = useState(true)
  const [isLive, setIsLive] = useState(true)
  const [timezoneMode, setTimezoneMode] = useState<TimezoneMode>('local')
  const [forecastOverlays, setForecastOverlays] = useState<ChartOverlay[]>([])
  const [forecastWarnings, setForecastWarnings] = useState<string[]>([])
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
  const queryActivity = chartQueryActivity(symbol, isLive)

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
    queryKey: [
      'hist',
      symbol,
      timeframe,
      QUERY_LIMIT,
      JSON.stringify(chartDenoise || {}),
      indicatorsQuery,
      indicatorsOhlcv,
    ],
    queryFn: ({ signal }) =>
      getHistory({
        symbol,
        timeframe,
        limit: QUERY_LIMIT,
        denoise: chartDenoise,
        include_incomplete: false,
        indicators: indicatorsQuery,
        ohlcv: indicatorsOhlcv,
      }, signal),
    enabled: queryActivity.history,
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
    enabled: queryActivity.liveHistory,
    refetchInterval: livePollMs,
  })

  const { data: tickData, error: tickError } = useQuery({
    queryKey: ['tick', symbol],
    queryFn: ({ signal }) => getTick(symbol, signal),
    enabled: queryActivity.tick,
    refetchInterval: isLive ? livePollMs : false,
  })

  useEffect(() => {
    setLiveHistory([])
  }, [historyContract])

  useEffect(() => {
    if (!isLive || !liveDataResponse?.data?.length) return
    setLiveHistory((previous) => mergeHistoryBars(previous, liveDataResponse.data))
  }, [historyContract, isLive, liveDataResponse])

  const bars = useMemo(() => {
    const base = (histDataResponse?.data ?? []) as HistoryBar[]
    return mergeHistoryBars(extraHistory, base, liveHistory)
  }, [extraHistory, histDataResponse, liveHistory])

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
      setForecastWarnings([])
      setMetrics(null)
    },
    []
  )

  const resetWorkspaceView = useCallback(() => {
    setExtraHistory([])
    setLiveHistory([])
    setForecastOverlays([])
    setForecastWarnings([])
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
      setLiveHistory([])
      setHistoryPageError(null)
      setForecastOverlays([])
      setForecastWarnings([])
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
      setLiveHistory([])
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
        setForecastWarnings([])
        setMetrics(null)
        return
      }
      let series
      try {
        series = mapCompactForecastToSeries(result.forecast ?? [])
      } catch {
        setForecastOverlays([])
        setForecastWarnings([])
        setMetrics(null)
        return
      }
      const feedback = forecastResultFeedback(result)
      setForecastWarnings(
        feedback.tone === 'success'
          ? []
          : [`Forecast: ${feedback.summary}`, ...feedback.details.map((detail) => `Forecast: ${detail}`)]
      )
      const { times, values: main, lower, upper } = series
      if (!main.length) {
        setForecastOverlays([])
        setForecastWarnings([])
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
          setMetrics(computeAnchorForecastMetrics(yPred, yAct, baselineClose))
        } else {
          setMetrics(null)
        }
      } else {
        setMetrics(null)
      }
    },
    [bars]
  )

  const denoiseFeedback = useMemo(
    () => resolveChartDenoiseFeedback(
      chartDenoise,
      histDataResponse,
      isLive ? liveDataResponse : undefined
    ),
    [chartDenoise, histDataResponse, isLive, liveDataResponse]
  )
  const indicatorOverlays = useMemo(
    () => buildIndicatorOverlays(bars, chartIndicators, {
      spec: chartDenoise,
      status: histDataResponse?.denoise_status
        || (histDataResponse?.denoise_applied === true ? 'applied' : undefined),
    }),
    [
      bars,
      chartDenoise,
      chartIndicators,
      histDataResponse?.denoise_applied,
      histDataResponse?.denoise_status,
    ]
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
    const lines = isLive
      ? liveQuotePriceLines(tickData, { showBid, showAsk, showLast })
      : []

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
    confluenceState.data,
    exposureState.data,
    ideaGeometry,
    isLive,
    pivotState.levels,
    showAsk,
    showBid,
    showLast,
    srState.levels,
    tickData,
    volumeProfileState.data,
  ])

  const earliest = bars.length ? bars[0].time : undefined
  const workspaceWarnings = useMemo(() => {
    const warnings = [
      ...responseWarningMessages('History', histDataResponse),
      ...(isLive
        ? responseWarningMessages(
            'Live history',
            liveDataResponse,
            new Set(['forming_candle_included'])
          )
        : []),
      ...(isLive ? responseWarningMessages('Quote', tickData) : []),
      denoiseFeedback.warning ?? null,
      ...forecastWarnings,
      isLive && tickData && tickData.usable_for_live_trading !== true
        ? 'Quote: Live price lines are hidden because this snapshot is not verified as usable for live trading.'
        : null,
    ].filter((value): value is string => Boolean(value))
    return Array.from(new Set(warnings))
  }, [denoiseFeedback.warning, forecastWarnings, histDataResponse, isLive, liveDataResponse, tickData])

  const workspaceErrors = useMemo(() => {
    const errors = [
      historyError ? `History: ${getErrorMessage(historyError)}` : null,
      isLive && liveHistoryError
        ? `Live history: ${getErrorMessage(liveHistoryError)}`
        : null,
      isLive && tickError ? `Quote: ${getErrorMessage(tickError)}` : null,
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
    denoiseFeedback,
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
    workspaceWarnings,
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
    exposureUpdatedAt: exposureState.updatedAt,
    reload: () => {
      setExtraHistory([])
      setLiveHistory([])
      void refetch()
    },
    toggleBid: () => setShowBid((value) => !value),
    toggleAsk: () => setShowAsk((value) => !value),
    toggleLast: () => setShowLast((value) => !value),
    toggleLive: () => setIsLive((value) => !value),
    clearAnchor: () => {
      setAnchor(undefined)
      setForecastOverlays([])
      setForecastWarnings([])
      setMetrics(null)
    },
  }
}
