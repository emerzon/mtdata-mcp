import {
  CandlestickSeries,
  createChart,
  HistogramSeries,
  IChartApi,
  IPriceLine,
  ISeriesApi,
  LineSeries,
  LineStyle,
  Time,
} from 'lightweight-charts'
import { useEffect, useRef } from 'react'
import type { HistoryBar, ChartOverlay } from '../types'

export type PriceLineSpec = {
  price: number
  color: string
  title: string
  lineStyle?: LineStyle
  lineWidth?: 1 | 2 | 3 | 4
}

export type OHLCChartProps = {
  data: HistoryBar[]
  onAnchor?: (t: number) => void
  onNeedMoreLeft?: (earliestTime: number) => void
  overlays?: ChartOverlay[]
  priceLines?: PriceLineSpec[]
  anchorTime?: number
  timeZone?: string
}

const chartTimeFormatters = new Map<string, Intl.DateTimeFormat>()

function formatChartTime(time: Time, timeZone: string, detailed: boolean): string {
  const epoch = Number(time)
  if (!Number.isFinite(epoch)) return String(time)
  const key = `${timeZone}:${detailed ? 'detailed' : 'tick'}`
  let formatter = chartTimeFormatters.get(key)
  if (!formatter) {
    formatter = new Intl.DateTimeFormat(undefined, {
      timeZone,
      month: 'short',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
      second: detailed ? '2-digit' : undefined,
      hourCycle: 'h23',
    })
    chartTimeFormatters.set(key, formatter)
  }
  return formatter.format(new Date(epoch * 1000))
}

export function OHLCChart({
  data,
  onAnchor,
  onNeedMoreLeft,
  overlays,
  priceLines,
  anchorTime,
  timeZone = 'UTC',
}: OHLCChartProps) {
  const ref = useRef<HTMLDivElement | null>(null)
  const apiRef = useRef<IChartApi | null>(null)
  const candleRef = useRef<ISeriesApi<'Candlestick'> | null>(null)
  const anchorRef = useRef<ISeriesApi<'Candlestick'> | null>(null)
  const priceLinesRef = useRef<IPriceLine[]>([])
  const timeZoneRef = useRef(timeZone)
  timeZoneRef.current = timeZone

  // Keep refs up to date for event handlers
  const dataRef = useRef(data)
  const onNeedMoreLeftRef = useRef(onNeedMoreLeft)
  const onAnchorRef = useRef(onAnchor)

  useEffect(() => {
    dataRef.current = data
    onNeedMoreLeftRef.current = onNeedMoreLeft
    onAnchorRef.current = onAnchor
  }, [data, onNeedMoreLeft, onAnchor])

  useEffect(() => {
    if (!ref.current) return

    const chart = createChart(ref.current, {
      autoSize: true,
      layout: { background: { color: '#0f172a' }, textColor: '#a3b3c7' },
      localization: {
        timeFormatter: (time: Time) => formatChartTime(time, timeZoneRef.current, true),
      },
      grid: { vertLines: { color: '#1f2937' }, horzLines: { color: '#1f2937' } },
      crosshair: { mode: 1 },
      rightPriceScale: { borderColor: '#1f2937' },
      timeScale: {
        borderColor: '#1f2937',
        timeVisible: true,
        secondsVisible: true,
        tickMarkFormatter: (time: Time) => formatChartTime(time, timeZoneRef.current, false),
      },
    })

    const series = chart.addSeries(CandlestickSeries, {
      upColor: '#22c55e',
      downColor: '#ef4444',
      borderVisible: false,
      wickUpColor: '#22c55e',
      wickDownColor: '#ef4444',
      lastValueVisible: false,
      priceLineVisible: false,
    })

    apiRef.current = chart
    candleRef.current = series

    const clickHandler = (p: { time?: Time }) => {
      if (!p || p.time === undefined) return
      const t = Number(p.time)
      if (!Number.isFinite(t)) return
      onAnchorRef.current?.(t)
    }

    const rangeHandler = (r: { from: Time; to: Time } | null) => {
      const currentData = dataRef.current
      const needMore = onNeedMoreLeftRef.current
      
      if (!r || !needMore || !currentData.length) return
      
      const from = r.from as number | undefined
      if (from) {
        const earliest = currentData[0].time
        // Trigger if visible range start is at or before the earliest bar
        if (from <= earliest) {
          needMore(earliest)
        }
      }
    }

    chart.subscribeClick(clickHandler)
    chart.timeScale().subscribeVisibleTimeRangeChange(rangeHandler)

    return () => {
      chart.unsubscribeClick(clickHandler)
      chart.timeScale().unsubscribeVisibleTimeRangeChange(rangeHandler)
      chart.remove()
      apiRef.current = null
      candleRef.current = null
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    apiRef.current?.applyOptions({
      localization: {
        timeFormatter: (time: Time) => formatChartTime(time, timeZoneRef.current, true),
      },
      timeScale: {
        tickMarkFormatter: (time: Time) => formatChartTime(time, timeZoneRef.current, false),
      },
    })
  }, [timeZone])

  useEffect(() => {
    if (!candleRef.current) return
    const series = candleRef.current
    const points = data.map(b => ({
      time: b.time as Time,
      open: b.open,
      high: b.high,
      low: b.low,
      close: b.close,
    }))
    series.setData(points)

    // Adjust seconds visibility based on detected bar spacing
    if (apiRef.current && data.length >= 2) {
      const dt = Math.abs(Math.floor(data[1].time) - Math.floor(data[0].time))
      const secondsVisible = dt < 60
      apiRef.current.applyOptions({ timeScale: { timeVisible: true, secondsVisible } })
    }
  }, [data])

  useEffect(() => {
    if (!apiRef.current) return
    const chart = apiRef.current
    const overlaySeries: Array<ISeriesApi<'Line'> | ISeriesApi<'Histogram'>> = []

    const extraPanes = chart.panes()
    for (let index = extraPanes.length - 1; index > 0; index -= 1) {
      if (extraPanes[index].getSeries().length === 0) {
        chart.removePane(index)
      }
    }

    const usedPanes = new Set(
      (overlays ?? [])
        .map((overlay) => overlay.pane)
        .filter((pane): pane is NonNullable<typeof pane> => Boolean(pane) && pane !== 'price')
    )
    const paneIndexByName = new Map<string, number>()
    let nextPane = 1
    for (const paneName of ['rsi', 'macd', 'volume'] as const) {
      if (usedPanes.has(paneName)) {
        paneIndexByName.set(paneName, nextPane)
        nextPane += 1
      }
    }

    overlays?.forEach((ov) => {
      if (!ov?.points?.length) return

      const style =
        ov.lineStyle === 'dashed'
          ? LineStyle.Dashed
          : ov.lineStyle === 'dotted'
          ? LineStyle.Dotted
          : LineStyle.Solid
      const paneIndex = ov.pane && ov.pane !== 'price' ? paneIndexByName.get(ov.pane) ?? 0 : 0
      const points = ov.points.filter(
        (point) => Number.isFinite(point.time) && Number.isFinite(point.value)
      )
      if (!points.length) return

      if (ov.kind === 'histogram') {
        const series = chart.addSeries(
          HistogramSeries,
          {
            color: ov.color || '#22c55e',
            priceLineVisible: false,
            lastValueVisible: Boolean(ov.label),
            priceScaleId: ov.priceScaleId || 'right',
          },
          paneIndex
        )
        series.setData(
          points.map((point) => ({
            time: Math.floor(point.time) as unknown as Time,
            value: point.value,
            color: point.color,
          }))
        )
        if (ov.label && points.length > 0) {
          const last = points[points.length - 1]
          series.createPriceLine({
            price: last.value,
            color: ov.color || '#22c55e',
            lineWidth: 1,
            lineStyle: LineStyle.Dotted,
            axisLabelVisible: true,
            title: ov.label,
          })
        }
        overlaySeries.push(series)
        return
      }

      const series = chart.addSeries(
        LineSeries,
        {
          color: ov.color || '#60a5fa',
          lineWidth: (ov.lineWidth ?? 2) as 1 | 2 | 3 | 4,
          lineStyle: style,
          priceScaleId: ov.priceScaleId || 'right',
          lastValueVisible: Boolean(ov.label),
        },
        paneIndex
      )

      series.setData(
        points.map((point) => ({
          time: Math.floor(point.time) as unknown as Time,
          value: point.value,
        }))
      )

      if (ov.label && points.length > 0) {
        const last = points[points.length - 1]
        series.createPriceLine({
          price: last.value,
          color: ov.color || '#60a5fa',
          lineWidth: (ov.lineWidth ?? 2) as 1 | 2 | 3 | 4,
          lineStyle: style,
          axisLabelVisible: true,
          title: ov.label,
        })
      }
      for (const line of ov.referenceLines ?? []) {
        series.createPriceLine({
          price: line.price,
          color: line.color,
          lineWidth: 1,
          lineStyle: LineStyle.Dashed,
          axisLabelVisible: true,
          title: line.title,
        })
      }

      overlaySeries.push(series)
    })

    const panes = chart.panes()
    if (panes.length > 1) {
      panes[0].setStretchFactor(0.72)
      const extra = 0.28 / (panes.length - 1)
      for (let index = 1; index < panes.length; index += 1) {
        panes[index].setStretchFactor(extra)
      }
    }

    return () => {
      overlaySeries.forEach((series) => chart.removeSeries(series))
    }
  }, [overlays])

  // Manage Price Lines (Bid/Ask)
  useEffect(() => {
    if (!candleRef.current) return
    const series = candleRef.current
    
    // Clear old lines
    priceLinesRef.current.forEach(l => series.removePriceLine(l))
    priceLinesRef.current = []

    // Add new lines
    priceLines?.forEach(pl => {
      const line = series.createPriceLine({
        price: pl.price,
        color: pl.color,
        lineWidth: pl.lineWidth ?? 1,
        lineStyle: pl.lineStyle ?? LineStyle.Dotted,
        axisLabelVisible: true,
        title: pl.title,
      })
      priceLinesRef.current.push(line)
    })
    
    return () => {
      if (candleRef.current) {
        priceLinesRef.current.forEach(l => candleRef.current?.removePriceLine(l))
      }
      priceLinesRef.current = []
    }
  }, [priceLines])

  // Anchor vertical marker using a 1-candle series with long wick spanning min..max
  useEffect(() => {
    const chart = apiRef.current
    if (!chart) return

    if (anchorRef.current) {
      chart.removeSeries(anchorRef.current)
      anchorRef.current = null
    }

    if (!data?.length || !Number.isFinite(anchorTime as number)) return

    const minLow = Math.min(...data.map(d => d.low))
    const maxHigh = Math.max(...data.map(d => d.high))
    const center = (minLow + maxHigh) / 2

    const s = chart.addSeries(CandlestickSeries, {
      upColor: '#facc15',
      downColor: '#facc15',
      borderVisible: false,
      wickUpColor: '#facc15',
      wickDownColor: '#facc15',
      priceScaleId: 'right',
    })

    s.setData([
      {
        time: Math.floor(anchorTime as number) as unknown as Time,
        open: center,
        high: maxHigh,
        low: minLow,
        close: center,
      },
    ])

    anchorRef.current = s

    return () => {
      if (anchorRef.current && chart) {
        chart.removeSeries(anchorRef.current)
        anchorRef.current = null
      }
    }
  }, [anchorTime, data])

  return <div className="w-full h-full" ref={ref} />
}
