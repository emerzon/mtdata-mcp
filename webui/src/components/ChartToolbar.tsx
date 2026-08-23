import { useState } from 'react'
import type { DenoiseSpecUI } from '../types'
import type { PivotMethod, SupportResistanceControls } from '../lib/overlayParams'
import { toolbarUsesOverflowMenu, type LayoutBreakpoint } from '../lib/layout'
import { useDismissiblePanel } from '../lib/useDismissiblePanel'
import { ApiAuthControl } from './ApiAuthControl'
import { ConnectionStatus } from './ConnectionStatus'
import { OverlayControls } from './OverlayControls'
import { RefreshIcon } from '../features/chart-workspace/toolbarIcons'
import {
  DenoiseSelector,
  IndicatorSelector,
  PriceLinesSelector,
  SymbolSelector,
  TimeframeSelector,
  TimezoneSelector,
} from '../features/chart-workspace/toolbarMenus'
import type { ChartIndicatorSelection } from '../lib/indicatorSpec'
import { formatEpochTime } from '../lib/time'

type Props = {
  symbol: string
  timeframe: string
  displayAnchor?: number
  isLoading: boolean
  barsCount: number
  hasPivots: boolean
  hasSR: boolean
  denoise?: DenoiseSpecUI
  indicators: ChartIndicatorSelection
  onIndicatorsChange: (value: ChartIndicatorSelection) => void
  showBid: boolean
  showAsk: boolean
  showLast: boolean
  isLive: boolean
  onSymbolChange: (value: string) => void
  onTimeframeChange: (value: string) => void
  onClearAnchor: () => void
  onReload: () => void
  onTogglePivots: () => void
  onToggleSR: () => void
  onDenoiseChange: (value?: DenoiseSpecUI) => void
  onOpenForecast: () => void
  onOpenTools: () => void
  onOpenIdea: () => void
  onOpenRadar: () => void
  onToggleBid: () => void
  onToggleAsk: () => void
  onToggleLast: () => void
  onToggleLive: () => void
  timezoneMode: 'utc' | 'local' | 'server'
  displayTimeZone: string
  onTimezoneChange: (value: 'utc' | 'local' | 'server') => void
  onAuthChange: () => void
  layoutBreakpoint: LayoutBreakpoint
  pivotMethod: PivotMethod
  onPivotMethodChange: (method: string) => void
  pivotsLoading?: boolean
  srControls: SupportResistanceControls
  onSrControlsChange: (partial: Partial<SupportResistanceControls>) => void
  srLoading?: boolean
  hasConfluence?: boolean
  onToggleConfluence?: () => void
  confluenceLoading?: boolean
  hasVolumeProfile?: boolean
  onToggleVolumeProfile?: () => void
  volumeProfileLoading?: boolean
  hasExposure?: boolean
  onToggleExposure?: () => void
  exposureLoading?: boolean
}

export function ChartToolbar({
  symbol,
  timeframe,
  displayAnchor,
  isLoading,
  barsCount,
  hasPivots,
  hasSR,
  denoise,
  indicators,
  onIndicatorsChange,
  showBid,
  showAsk,
  showLast,
  isLive,
  onSymbolChange,
  onTimeframeChange,
  onClearAnchor,
  onReload,
  onTogglePivots,
  onToggleSR,
  onDenoiseChange,
  onOpenForecast,
  onOpenTools,
  onOpenIdea,
  onOpenRadar,
  onToggleBid,
  onToggleAsk,
  onToggleLast,
  onToggleLive,
  timezoneMode,
  displayTimeZone,
  onTimezoneChange,
  onAuthChange,
  layoutBreakpoint,
  pivotMethod,
  onPivotMethodChange,
  pivotsLoading,
  srControls,
  onSrControlsChange,
  srLoading,
  hasConfluence,
  onToggleConfluence,
  confluenceLoading,
  hasVolumeProfile,
  onToggleVolumeProfile,
  volumeProfileLoading,
  hasExposure,
  onToggleExposure,
  exposureLoading,
}: Props) {
  const overflow = toolbarUsesOverflowMenu(layoutBreakpoint)
  const [moreOpen, setMoreOpen] = useState(false)
  const moreRef = useDismissiblePanel(() => setMoreOpen(false), moreOpen)

  const analysisGroup = (
    <>
      <ApiAuthControl onChange={onAuthChange} />
      <div className="w-px h-5 bg-slate-700 hidden sm:block" />
      <OverlayControls
        disabled={!symbol}
        pivotMethod={pivotMethod}
        onPivotMethodChange={onPivotMethodChange}
        hasPivots={hasPivots}
        onTogglePivots={onTogglePivots}
        pivotsLoading={pivotsLoading}
        srControls={srControls}
        onSrControlsChange={onSrControlsChange}
        hasSR={hasSR}
        onToggleSR={onToggleSR}
        srLoading={srLoading}
        hasConfluence={hasConfluence}
        onToggleConfluence={onToggleConfluence}
        confluenceLoading={confluenceLoading}
        hasVolumeProfile={hasVolumeProfile}
        onToggleVolumeProfile={onToggleVolumeProfile}
        volumeProfileLoading={volumeProfileLoading}
        hasExposure={hasExposure}
        onToggleExposure={onToggleExposure}
        exposureLoading={exposureLoading}
      />
      <div className="w-px h-5 bg-slate-700 hidden sm:block" />
      <PriceLinesSelector
        showBid={showBid}
        showAsk={showAsk}
        showLast={showLast}
        isLive={isLive}
        disabled={!symbol}
        onToggleBid={onToggleBid}
        onToggleAsk={onToggleAsk}
        onToggleLast={onToggleLast}
        onToggleLive={onToggleLive}
      />
      <div className="w-px h-5 bg-slate-700 hidden sm:block" />
      <IndicatorSelector value={indicators} disabled={!symbol} onChange={onIndicatorsChange} />
      <DenoiseSelector value={denoise} disabled={!symbol} onChange={onDenoiseChange} />
    </>
  )

  return (
    <div className="absolute top-2 left-2 right-2 z-20 flex flex-wrap items-start gap-2 pointer-events-none [&>*]:pointer-events-auto">
      <div className="flex flex-wrap items-center gap-1 bg-slate-900/95 backdrop-blur-sm rounded-lg border border-slate-800 p-1 max-w-full">
        <SymbolSelector value={symbol} onChange={onSymbolChange} />
        <div className="w-px h-5 bg-slate-700" />
        <TimezoneSelector value={timezoneMode} onChange={onTimezoneChange} />
        <div className="w-px h-5 bg-slate-700" />
        <TimeframeSelector value={timeframe} onChange={onTimeframeChange} />
        <div className="w-px h-5 bg-slate-700" />
        <button
          className="toolbar-btn min-h-9 min-w-9 justify-center"
          onClick={onReload}
          disabled={!symbol || isLoading}
          title="Reload data"
        >
          <RefreshIcon className={isLoading ? 'animate-spin' : ''} />
        </button>
      </div>

      <div className="flex-1 min-w-[0.5rem]" />

      {symbol && layoutBreakpoint === 'desktop' && (
        <div className="bg-slate-900/95 backdrop-blur-sm rounded-lg border border-slate-800 px-3 py-1.5 text-xs text-slate-400">
          {barsCount} bars
          {displayAnchor !== undefined && (
            <span className="ml-2 text-amber-400">
              Anchor: {formatEpochTime(displayAnchor, displayTimeZone)}
              <button className="ml-1 text-slate-500 hover:text-slate-300" onClick={onClearAnchor}>
                ×
              </button>
            </span>
          )}
        </div>
      )}

      <div className="flex flex-wrap items-center gap-2 ml-auto">
        <ConnectionStatus />

        {overflow ? (
          <div className="relative" ref={moreRef}>
            <button
              type="button"
              className="toolbar-btn bg-slate-900/95 border border-slate-800 rounded-lg min-h-9 px-3"
              onClick={() => setMoreOpen((value) => !value)}
              aria-expanded={moreOpen}
            >
              More
            </button>
            {moreOpen && (
              <div className="absolute top-full right-0 mt-1 flex flex-wrap items-center gap-1 bg-slate-900 border border-slate-700 rounded-lg shadow-xl p-2 z-50 max-w-[min(24rem,calc(100vw-1rem))]">
                {analysisGroup}
              </div>
            )}
          </div>
        ) : (
          <div className="flex flex-wrap items-center gap-1 bg-slate-900/95 backdrop-blur-sm rounded-lg border border-slate-800 p-1">
            {analysisGroup}
          </div>
        )}

        <button
          type="button"
          className="bg-slate-800 hover:bg-slate-700 text-slate-100 text-sm font-medium px-3 py-2 min-h-9 rounded-lg border border-slate-700 transition-colors"
          onClick={onOpenRadar}
          title="Open the watchlist radar"
        >
          Watch
        </button>
        <button
          type="button"
          className="bg-slate-800 hover:bg-slate-700 text-slate-100 text-sm font-medium px-3 py-2 min-h-9 rounded-lg border border-slate-700 transition-colors"
          onClick={onOpenTools}
          title="Browse and run all backend tools"
        >
          Tools
        </button>
        <button
          type="button"
          className="bg-slate-800 hover:bg-slate-700 text-slate-100 text-sm font-medium px-3 py-2 min-h-9 rounded-lg border border-slate-700 transition-colors disabled:opacity-50"
          onClick={onOpenIdea}
          disabled={!symbol}
          title="Compose a preview-only trade idea"
        >
          Idea
        </button>
        <button
          className="bg-sky-600 hover:bg-sky-500 text-white text-sm font-medium px-3 sm:px-4 py-2 min-h-9 rounded-lg disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          onClick={onOpenForecast}
          disabled={!symbol}
        >
          Forecast
        </button>
      </div>

      {symbol && layoutBreakpoint !== 'desktop' && displayAnchor !== undefined && (
        <div className="w-full bg-slate-900/95 backdrop-blur-sm rounded-lg border border-slate-800 px-3 py-1.5 text-xs text-amber-400">
          Anchor: {formatEpochTime(displayAnchor, displayTimeZone)}
          <button className="ml-2 text-slate-500 hover:text-slate-300" onClick={onClearAnchor}>
            Clear
          </button>
        </div>
      )}
    </div>
  )
}
