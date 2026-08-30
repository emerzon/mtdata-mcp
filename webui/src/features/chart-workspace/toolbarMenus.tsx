import { useMemo, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import {
  getErrorMessage,
  getTimeframes,
  searchInstruments,
} from '../../api/client'
import { DenoiseModal } from '../../components/DenoiseModal'
import {
  CHART_INDICATOR_IDS,
  DEFAULT_CHART_INDICATORS,
  SAMPLE_TRADE_INDICATORS,
  chartIndicatorsActive,
  type ChartIndicatorId,
  type ChartIndicatorSelection,
} from '../../lib/indicatorSpec'
import { loadJSON } from '../../lib/storage'
import { useDismissiblePanel } from '../../lib/useDismissiblePanel'
import type { DenoiseSpecUI } from '../../types'
import type { ChartDenoiseFeedback } from '../../lib/historyFeedback'
import { ChevronDown, IndicatorIcon } from './toolbarIcons'

export function SymbolSelector({
  value,
  onChange,
}: {
  value: string
  onChange: (value: string) => void
}) {
  const [open, setOpen] = useState(false)
  const [search, setSearch] = useState('')

  return (
    <div className="relative">
      <button className="toolbar-btn min-w-[120px] justify-between" onClick={() => setOpen((value) => !value)}>
        <span className={value ? 'text-slate-200' : 'text-slate-500'}>{value || 'Symbol'}</span>
        <ChevronDown />
      </button>
      {open && (
        <SymbolDropdown
          value={value}
          search={search}
          onSearchChange={setSearch}
          onSelect={(symbol) => {
            onChange(symbol)
            setOpen(false)
          }}
          onClose={() => setOpen(false)}
        />
      )}
    </div>
  )
}

function SymbolDropdown({
  value,
  search,
  onSearchChange,
  onSelect,
  onClose,
}: {
  value: string
  search: string
  onSearchChange: (value: string) => void
  onSelect: (value: string) => void
  onClose: () => void
}) {
  const ref = useDismissiblePanel(onClose)
  const { data: searchResults, error, isFetching } = useQuery({
    queryKey: ['instruments', search],
    queryFn: ({ signal }) => searchInstruments(search || undefined, 50, signal),
    enabled: !!search,
  })

  const items = useMemo(() => {
    if (search) return searchResults ?? []
    const recent = loadJSON<string[]>('recent_symbols') || []
    return recent.map((symbol) => ({ symbol, description: 'Recent' }))
  }, [search, searchResults])

  return (
    <div ref={ref} className="absolute top-full left-0 mt-1 w-72 bg-slate-900 border border-slate-700 rounded-lg shadow-xl overflow-hidden z-50">
      <input
        className="w-full px-3 py-2 bg-slate-800 text-sm text-slate-200 border-b border-slate-700 focus:outline-none"
        placeholder="Search instruments..."
        value={search}
        onChange={(event) => onSearchChange(event.target.value)}
        autoFocus
      />
      <div className="max-h-64 overflow-y-auto">
        {items.map((item) => (
          <button
            key={item.symbol}
            className={`w-full text-left px-3 py-2 text-sm hover:bg-slate-800 ${
              item.symbol === value ? 'bg-slate-800 text-sky-400' : 'text-slate-300'
            }`}
            onClick={() => onSelect(item.symbol)}
          >
            <span className="font-medium">{item.symbol}</span>
            {item.description && <span className="ml-2 text-slate-500 text-xs">{item.description}</span>}
          </button>
        ))}
        {error && (
          <div className="px-3 py-4 text-sm text-rose-400 text-center">
            Instrument search failed: {getErrorMessage(error)}
          </div>
        )}
        {!error && isFetching && (
          <div className="px-3 py-4 text-sm text-slate-500 text-center">Searching...</div>
        )}
        {!error && !isFetching && items.length === 0 && (
          <div className="px-3 py-4 text-sm text-slate-500 text-center">
            {search ? 'No instruments found' : 'No recent instruments'}
          </div>
        )}
      </div>
    </div>
  )
}

export function TimezoneSelector({
  value,
  onChange,
}: {
  value: 'utc' | 'local' | 'server'
  onChange: (value: 'utc' | 'local' | 'server') => void
}) {
  const [open, setOpen] = useState(false)

  return (
    <div className="relative">
      <button
        className="toolbar-btn min-w-[60px] justify-between text-slate-400"
        onClick={() => setOpen((value) => !value)}
        title="Timezone"
      >
        <span className="text-xs font-medium uppercase">{value === 'server' ? 'Exch' : value}</span>
        <ChevronDown />
      </button>
      {open && (
        <TimezoneDropdown
          value={value}
          onChange={(mode) => {
            onChange(mode)
            setOpen(false)
          }}
          onClose={() => setOpen(false)}
        />
      )}
    </div>
  )
}

function TimezoneDropdown({
  value,
  onChange,
  onClose,
}: {
  value: 'utc' | 'local' | 'server'
  onChange: (value: 'utc' | 'local' | 'server') => void
  onClose: () => void
}) {
  const ref = useDismissiblePanel(onClose)

  return (
    <div ref={ref} className="absolute top-full left-0 mt-1 w-32 bg-slate-900 border border-slate-700 rounded-lg shadow-xl overflow-hidden z-50">
      <div className="px-3 py-2 text-xs text-slate-400 border-b border-slate-700 font-medium">Timezone</div>
      {(['utc', 'server', 'local'] as const).map((mode) => (
        <button
          key={mode}
          className={`w-full text-left px-3 py-2 text-sm hover:bg-slate-800 ${
            value === mode ? 'text-sky-400' : 'text-slate-300'
          }`}
          onClick={() => onChange(mode)}
        >
          {mode === 'server' ? 'Exchange' : mode.toUpperCase()}
        </button>
      ))}
    </div>
  )
}

export function TimeframeSelector({
  value,
  onChange,
}: {
  value: string
  onChange: (value: string) => void
}) {
  const [open, setOpen] = useState(false)
  const ref = useDismissiblePanel(() => setOpen(false), open)
  const { data } = useQuery({ queryKey: ['timeframes'], queryFn: getTimeframes })
  const timeframes = data ?? []

  return (
    <div ref={ref} className="relative">
      <button className="toolbar-btn min-w-[50px] justify-between" onClick={() => setOpen((value) => !value)}>
        <span>{value}</span>
        <ChevronDown />
      </button>
      {open && (
        <div className="absolute top-full left-0 mt-1 w-24 bg-slate-900 border border-slate-700 rounded-lg shadow-xl overflow-hidden z-50 max-h-64 overflow-y-auto">
          {timeframes.map((timeframe) => (
            <button
              key={timeframe}
              className={`w-full text-left px-3 py-1.5 text-sm hover:bg-slate-800 ${
                timeframe === value ? 'bg-slate-800 text-sky-400' : 'text-slate-300'
              }`}
              onClick={() => {
                onChange(timeframe)
                setOpen(false)
              }}
            >
              {timeframe}
            </button>
          ))}
        </div>
      )}
    </div>
  )
}

export function DenoiseSelector({
  value,
  feedback,
  disabled,
  onChange,
}: {
  value?: DenoiseSpecUI
  feedback: ChartDenoiseFeedback
  disabled: boolean
  onChange: (value?: DenoiseSpecUI) => void
}) {
  const [open, setOpen] = useState(false)
  const statusClass = feedback.state === 'applied'
    ? 'text-emerald-400'
    : feedback.state === 'skipped' || feedback.state === 'failed'
      ? 'text-rose-400'
      : feedback.state === 'pending'
        ? 'text-amber-400'
        : ''
  const statusLabel = feedback.state === 'applied'
    ? 'Filter ✓'
    : feedback.state === 'skipped' || feedback.state === 'failed'
      ? 'Filter !'
      : 'Filter'

  return (
    <div className="relative">
      <button
        className={`toolbar-btn ${statusClass}`}
        onClick={() => setOpen(true)}
        disabled={disabled}
        title={feedback.title}
        aria-label="Chart denoising"
        aria-pressed={feedback.state === 'applied'}
      >
        <span>{statusLabel}</span>
        <ChevronDown />
      </button>
      <DenoiseModal
        open={open}
        title="Chart Denoising"
        value={value}
        onClose={() => setOpen(false)}
        onApply={(denoise) => {
          onChange(denoise)
          setOpen(false)
        }}
      />
    </div>
  )
}

const INDICATOR_LABELS: Record<ChartIndicatorId, string> = {
  ema20: 'EMA 20',
  ema50: 'EMA 50',
  rsi14: 'RSI 14',
  macd: 'MACD',
  volume: 'Volume',
}

export function IndicatorSelector({
  value,
  disabled,
  onChange,
}: {
  value: ChartIndicatorSelection
  disabled: boolean
  onChange: (value: ChartIndicatorSelection) => void
}) {
  const [open, setOpen] = useState(false)
  const active = chartIndicatorsActive(value)

  return (
    <div className="relative">
      <button
        type="button"
        className={`toolbar-btn ${active ? 'text-sky-400' : ''}`}
        onClick={() => setOpen((current) => !current)}
        disabled={disabled}
        title="Chart indicators"
        aria-expanded={open}
        aria-pressed={active}
      >
        <IndicatorIcon />
        <span>Indicators</span>
        <ChevronDown />
      </button>
      {open && (
        <IndicatorDropdown
          value={value}
          onChange={onChange}
          onClose={() => setOpen(false)}
        />
      )}
    </div>
  )
}

function IndicatorDropdown({
  value,
  onChange,
  onClose,
}: {
  value: ChartIndicatorSelection
  onChange: (value: ChartIndicatorSelection) => void
  onClose: () => void
}) {
  const ref = useDismissiblePanel(onClose)

  const toggle = (id: ChartIndicatorId) => {
    onChange({ ...value, [id]: !value[id] })
  }

  return (
    <div
      ref={ref}
      className="absolute top-full right-0 mt-1 w-60 bg-slate-900 border border-slate-700 rounded-lg shadow-xl overflow-hidden z-50"
      role="dialog"
      aria-label="Chart indicators"
    >
      <div className="px-3 py-2 text-xs text-slate-400 border-b border-slate-700 font-medium">
        Chart indicators
      </div>
      <div className="px-3 py-2 flex gap-2 border-b border-slate-800">
        <button
          type="button"
          className="flex-1 text-xs px-2 py-1.5 rounded border border-slate-700 text-slate-200 hover:bg-slate-800"
          onClick={() => onChange({ ...SAMPLE_TRADE_INDICATORS })}
        >
          Sample trade
        </button>
        <button
          type="button"
          className="flex-1 text-xs px-2 py-1.5 rounded border border-slate-700 text-slate-400 hover:bg-slate-800"
          onClick={() => onChange({ ...DEFAULT_CHART_INDICATORS })}
        >
          Clear
        </button>
      </div>
      <div className="py-1">
        {CHART_INDICATOR_IDS.map((id) => (
          <label
            key={id}
            className="flex items-center gap-2 px-3 py-1.5 text-sm text-slate-200 hover:bg-slate-800 cursor-pointer"
          >
            <input
              type="checkbox"
              className="accent-sky-500"
              checked={value[id]}
              onChange={() => toggle(id)}
            />
            {INDICATOR_LABELS[id]}
          </label>
        ))}
      </div>
      <p className="px-3 py-2 text-[11px] text-slate-500 border-t border-slate-800">
        Lines and panes are research overlays, not trade signals. Volume on FX is usually tick volume only.
      </p>
    </div>
  )
}

export function PriceLinesSelector({
  showBid,
  showAsk,
  showLast,
  isLive,
  disabled,
  onToggleBid,
  onToggleAsk,
  onToggleLast,
  onToggleLive,
}: {
  showBid: boolean
  showAsk: boolean
  showLast: boolean
  isLive: boolean
  disabled: boolean
  onToggleBid: () => void
  onToggleAsk: () => void
  onToggleLast: () => void
  onToggleLive: () => void
}) {
  const [open, setOpen] = useState(false)

  return (
    <div className="relative">
      <button
        className={`toolbar-btn ${showBid || showAsk || showLast ? 'text-sky-400' : ''}`}
        onClick={() => setOpen((value) => !value)}
        disabled={disabled}
        title="Price Lines"
      >
        <span>Lines</span>
        <ChevronDown />
      </button>
      {open && (
        <PriceLinesDropdown
          showBid={showBid}
          showAsk={showAsk}
          showLast={showLast}
          isLive={isLive}
          onToggleBid={onToggleBid}
          onToggleAsk={onToggleAsk}
          onToggleLast={onToggleLast}
          onToggleLive={onToggleLive}
          onClose={() => setOpen(false)}
        />
      )}
    </div>
  )
}

function PriceLinesDropdown({
  showBid,
  showAsk,
  showLast,
  isLive,
  onToggleBid,
  onToggleAsk,
  onToggleLast,
  onToggleLive,
  onClose,
}: {
  showBid: boolean
  showAsk: boolean
  showLast: boolean
  isLive: boolean
  onToggleBid: () => void
  onToggleAsk: () => void
  onToggleLast: () => void
  onToggleLive: () => void
  onClose: () => void
}) {
  const ref = useDismissiblePanel(onClose)

  return (
    <div ref={ref} className="absolute top-full right-0 mt-1 w-48 bg-slate-900 border border-slate-700 rounded-lg shadow-xl overflow-hidden z-50">
      <div className="px-3 py-2 text-xs text-slate-400 border-b border-slate-700 font-medium">Chart Settings</div>
      <div className="p-1">
        <ToolbarCheckbox checked={isLive} label="Live Chart" onChange={onToggleLive} />
        <div className="h-px bg-slate-700 my-1" />
        <ToolbarCheckbox checked={showLast} label="Show Last" onChange={onToggleLast} />
        <ToolbarCheckbox checked={showBid} label="Show Bid" onChange={onToggleBid} />
        <ToolbarCheckbox checked={showAsk} label="Show Ask" onChange={onToggleAsk} />
      </div>
    </div>
  )
}

function ToolbarCheckbox({
  checked,
  label,
  onChange,
}: {
  checked: boolean
  label: string
  onChange: () => void
}) {
  return (
    <label className="flex items-center gap-2 px-2 py-1.5 hover:bg-slate-800 rounded cursor-pointer">
      <input
        type="checkbox"
        className="rounded border-slate-600 bg-slate-700 text-sky-500 focus:ring-offset-slate-900"
        checked={checked}
        onChange={onChange}
      />
      <span className="text-sm text-slate-300">{label}</span>
    </label>
  )
}
