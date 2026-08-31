import { useMemo, useState } from 'react'
import { OHLCChart } from './components/OHLCChart'
import { ChartToolbar } from './components/ChartToolbar'
import { ChartWorkspaceStatusView } from './components/ChartWorkspaceStatus'
import { ForecastPanel } from './components/ForecastPanel'
import { IdeaPanel } from './components/IdeaPanel'
import { RadarPanel } from './components/RadarPanel'
import { ToolsRunnerPanel } from './components/ToolsRunnerPanel'
import { useChartWorkspace } from './features/chart-workspace/useChartWorkspace'
import { useViewportBreakpoint } from './hooks/useViewportBreakpoint'
import { resolveChartWorkspaceStatus } from './lib/workspaceStatus'

type ActivePanel = 'forecast' | 'tools' | 'idea' | 'radar' | null

export default function App() {
  const [activePanel, setActivePanel] = useState<ActivePanel>(null)
  const [ideaAutoKey, setIdeaAutoKey] = useState(0)
  const workspace = useChartWorkspace()
  const layoutBreakpoint = useViewportBreakpoint()

  const chartStatus = useMemo(
    () =>
      resolveChartWorkspaceStatus({
        symbol: workspace.symbol,
        isLoading: workspace.isFetching || workspace.isLoadingMore,
        isInitialLoading: workspace.isInitialHistoryLoading,
        barsCount: workspace.bars.length,
        historyError: workspace.historyErrorMessage,
      }),
    [
      workspace.bars.length,
      workspace.historyErrorMessage,
      workspace.isFetching,
      workspace.isInitialHistoryLoading,
      workspace.isLoadingMore,
      workspace.symbol,
    ]
  )

  return (
    <div className="h-full min-h-screen flex flex-col bg-slate-950 text-slate-100 overflow-x-hidden">
      <main className="flex-1 relative min-h-0 overflow-hidden">
        <ChartToolbar
          symbol={workspace.symbol}
          timeframe={workspace.timeframe}
          displayAnchor={workspace.anchor}
          isLoading={workspace.isFetching || workspace.isLoadingMore}
          onSymbolChange={workspace.handleSymbolChange}
          onTimeframeChange={workspace.handleTimeframeChange}
          onClearAnchor={workspace.clearAnchor}
          onReload={workspace.reload}
          onTogglePivots={workspace.handlePivotToggle}
          onToggleSR={workspace.handleSRToggle}
          onDenoiseChange={workspace.handleDenoiseChange}
          onOpenForecast={() => setActivePanel('forecast')}
          onOpenTools={() => setActivePanel('tools')}
          onOpenIdea={() => setActivePanel('idea')}
          onOpenRadar={() => setActivePanel('radar')}
          hasPivots={!!workspace.pivotLevels}
          hasSR={!!workspace.srLevels}
          denoise={workspace.chartDenoise}
          denoiseFeedback={workspace.denoiseFeedback}
          indicators={workspace.chartIndicators}
          onIndicatorsChange={workspace.handleIndicatorsChange}
          barsCount={workspace.bars.length}
          showBid={workspace.showBid}
          showAsk={workspace.showAsk}
          showLast={workspace.showLast}
          isLive={workspace.isLive}
          timezoneMode={workspace.timezoneMode}
          displayTimeZone={workspace.displayTimeZone}
          onToggleBid={workspace.toggleBid}
          onToggleAsk={workspace.toggleAsk}
          onToggleLast={workspace.toggleLast}
          onToggleLive={workspace.toggleLive}
          onTimezoneChange={workspace.setTimezoneMode}
          layoutBreakpoint={layoutBreakpoint}
          pivotMethod={workspace.pivotMethod}
          onPivotMethodChange={workspace.handlePivotMethodChange}
          pivotsLoading={workspace.pivotsLoading}
          srControls={workspace.srControls}
          onSrControlsChange={workspace.handleSrControlsChange}
          srLoading={workspace.srLoading}
          hasConfluence={!!workspace.confluenceLevels?.length}
          onToggleConfluence={workspace.handleConfluenceToggle}
          confluenceLoading={workspace.confluenceLoading}
          hasVolumeProfile={!!workspace.volumeProfile}
          onToggleVolumeProfile={workspace.handleVolumeProfileToggle}
          volumeProfileLoading={workspace.volumeProfileLoading}
          hasExposure={!!workspace.exposure}
          onToggleExposure={workspace.handleExposureToggle}
          exposureLoading={workspace.exposureLoading}
          exposureUpdatedAt={workspace.exposureUpdatedAt}
        />

        <div className="absolute inset-0" data-chart-surface>
          <OHLCChart
            data={workspace.bars}
            onAnchor={workspace.handleAnchorSelect}
            onNeedMoreLeft={workspace.earliest ? workspace.handleNeedMoreLeft : undefined}
            anchorTime={workspace.anchor}
            overlays={workspace.chartOverlays}
            priceLines={workspace.priceLines}
            timeZone={workspace.displayTimeZone}
          />
        </div>

        <ChartWorkspaceStatusView status={chartStatus} onReload={workspace.reload} />

        {chartStatus.kind === 'ready'
          && (workspace.workspaceErrors.length > 0 || workspace.workspaceWarnings.length > 0)
          && (
          <div className="absolute top-16 sm:top-14 left-2 right-2 sm:right-auto z-20 max-w-xl space-y-2 text-xs shadow-lg">
            {workspace.workspaceErrors.length > 0 && (
              <div
                className="rounded-lg border border-rose-800 bg-rose-950/95 px-3 py-2 text-rose-200"
                role="alert"
              >
                {workspace.workspaceErrors.map((message) => (
                  <div key={message}>{message}</div>
                ))}
              </div>
            )}
            {workspace.workspaceWarnings.length > 0 && (
              <div
                className="rounded-lg border border-amber-800 bg-amber-950/95 px-3 py-2 text-amber-100"
                role="status"
              >
                {workspace.workspaceWarnings.map((message) => (
                  <div key={message}>{message}</div>
                ))}
              </div>
            )}
          </div>
        )}

        {workspace.metrics && (
          <div className="absolute bottom-4 left-2 right-2 sm:right-auto sm:left-4 flex flex-wrap gap-2 z-20 pointer-events-none">
            <MetricBadge label="n" value={String(workspace.metrics.overlap)} />
            <MetricBadge label="MAE" value={workspace.metrics.mae.toFixed(4)} />
            <MetricBadge label="MAPE" value={`${workspace.metrics.mape.toFixed(1)}%`} />
            <MetricBadge label="RMSE" value={workspace.metrics.rmse.toFixed(4)} />
            <MetricBadge
              label="Dir"
              value={workspace.metrics.dirAcc == null ? '—' : `${workspace.metrics.dirAcc.toFixed(0)}%`}
              variant={
                workspace.metrics.dirAcc == null
                  ? 'default'
                  : workspace.metrics.dirAcc >= 60
                    ? 'success'
                    : workspace.metrics.dirAcc >= 50
                      ? 'warning'
                      : 'error'
              }
            />
          </div>
        )}


        <ForecastPanel
          open={activePanel === 'forecast'}
          onClose={() => setActivePanel(null)}
          symbol={workspace.symbol}
          timeframe={workspace.timeframe}
          anchor={workspace.anchor}
          onResult={workspace.handleForecastResult}
          layoutBreakpoint={layoutBreakpoint}
        />

        <RadarPanel
          open={activePanel === 'radar'}
          onClose={() => setActivePanel(null)}
          symbol={workspace.symbol}
          timeframe={workspace.timeframe}
          onSelectSymbol={workspace.handleSymbolChange}
          onComposeIdea={(name) => {
            workspace.handleSymbolChange(name)
            setIdeaAutoKey((value) => value + 1)
            setActivePanel('idea')
          }}
          layoutBreakpoint={layoutBreakpoint}
        />

        <IdeaPanel
          open={activePanel === 'idea'}
          onClose={() => setActivePanel(null)}
          symbol={workspace.symbol}
          timeframe={workspace.timeframe}
          onIdea={workspace.handleIdeaResult}
          layoutBreakpoint={layoutBreakpoint}
          autoComposeKey={ideaAutoKey || undefined}
        />

        <ToolsRunnerPanel
          open={activePanel === 'tools'}
          onClose={() => setActivePanel(null)}
          layoutBreakpoint={layoutBreakpoint}
          symbol={workspace.symbol}
          timeframe={workspace.timeframe}
        />
      </main>
    </div>
  )
}

function MetricBadge({
  label,
  value,
  variant = 'default',
}: {
  label: string
  value: string
  variant?: 'default' | 'success' | 'warning' | 'error'
}) {
  const colors = {
    default: 'bg-slate-800/90 text-slate-300 border-slate-700',
    success: 'bg-emerald-950/90 text-emerald-300 border-emerald-800',
    warning: 'bg-amber-950/90 text-amber-300 border-amber-800',
    error: 'bg-rose-950/90 text-rose-300 border-rose-800',
  }
  return (
    <div className={`px-2 py-1 rounded border text-xs font-medium backdrop-blur-sm ${colors[variant]}`}>
      <span className="text-slate-500 mr-1">{label}</span>
      {value}
    </div>
  )
}
