import { useState } from 'react'
import {
  PIVOT_METHODS,
  type PivotMethod,
  type SupportResistanceControls,
} from '../lib/overlayParams'
import { useEscapeKey } from '../lib/useEscapeKey'

type Props = {
  disabled?: boolean
  pivotMethod: PivotMethod
  onPivotMethodChange: (method: string) => void
  hasPivots: boolean
  onTogglePivots: () => void
  pivotsLoading?: boolean
  srControls: SupportResistanceControls
  onSrControlsChange: (partial: Partial<SupportResistanceControls>) => void
  hasSR: boolean
  onToggleSR: () => void
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

/**
 * Pivot method + S/R parameter controls with progressive disclosure.
 */
export function OverlayControls({
  disabled,
  pivotMethod,
  onPivotMethodChange,
  hasPivots,
  onTogglePivots,
  pivotsLoading,
  srControls,
  onSrControlsChange,
  hasSR,
  onToggleSR,
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
  const [open, setOpen] = useState(false)
  useEscapeKey(open, () => setOpen(false))

  return (
    <div className="relative">
      <button
        type="button"
        className={`toolbar-btn text-xs ${open || hasPivots || hasSR || hasConfluence || hasVolumeProfile || hasExposure ? 'text-sky-300' : ''}`}
        disabled={disabled}
        onClick={() => setOpen((value) => !value)}
        title="Pivot method and support/resistance parameters"
        aria-expanded={open}
        aria-haspopup="dialog"
      >
        Levels
      </button>

      {open && (
        <div
          className="absolute top-full right-0 mt-1 w-72 max-w-[min(18rem,calc(100vw-1.5rem))] rounded-lg border border-slate-700 bg-slate-900 shadow-xl z-50 p-3 space-y-3"
          role="dialog"
          aria-label="Level overlay controls"
        >
          <div className="space-y-1.5">
            <div className="flex items-center justify-between">
              <span className="text-xs font-medium text-slate-300">Pivots</span>
              <button
                type="button"
                className={`text-xs px-2 py-1 rounded border ${
                  hasPivots
                    ? 'border-amber-700 text-amber-300 bg-amber-950/40'
                    : 'border-slate-700 text-slate-300 hover:bg-slate-800'
                }`}
                disabled={disabled || pivotsLoading}
                onClick={() => void onTogglePivots()}
              >
                {pivotsLoading ? '…' : hasPivots ? 'On' : 'Off'}
              </button>
            </div>
            <label className="block text-[11px] text-slate-500">
              Method
              <select
                className="mt-0.5 w-full bg-slate-800 text-slate-200 text-xs rounded px-2 py-1.5 border border-slate-700"
                value={pivotMethod}
                disabled={disabled}
                onChange={(event) => void onPivotMethodChange(event.target.value)}
              >
                {PIVOT_METHODS.map((method) => (
                  <option key={method} value={method}>
                    {method}
                  </option>
                ))}
              </select>
            </label>
          </div>

          <div className="border-t border-slate-800 pt-2 space-y-1.5">
            <div className="flex items-center justify-between">
              <span className="text-xs font-medium text-slate-300">Support / Resistance</span>
              <button
                type="button"
                className={`text-xs px-2 py-1 rounded border ${
                  hasSR
                    ? 'border-emerald-700 text-emerald-300 bg-emerald-950/40'
                    : 'border-slate-700 text-slate-300 hover:bg-slate-800'
                }`}
                disabled={disabled || srLoading}
                onClick={() => void onToggleSR()}
              >
                {srLoading ? '…' : hasSR ? 'On' : 'Off'}
              </button>
            </div>
            <div className="grid grid-cols-2 gap-2">
              <label className="block text-[11px] text-slate-500">
                Lookback
                <input
                  type="number"
                  min={100}
                  max={20000}
                  className="mt-0.5 w-full bg-slate-800 text-slate-200 text-xs rounded px-2 py-1.5 border border-slate-700"
                  value={srControls.lookback}
                  disabled={disabled}
                  onChange={(event) =>
                    void onSrControlsChange({ lookback: Number(event.target.value) })
                  }
                />
              </label>
              <label className="block text-[11px] text-slate-500">
                Min touches
                <input
                  type="number"
                  min={1}
                  max={50}
                  className="mt-0.5 w-full bg-slate-800 text-slate-200 text-xs rounded px-2 py-1.5 border border-slate-700"
                  value={srControls.min_touches}
                  disabled={disabled}
                  onChange={(event) =>
                    void onSrControlsChange({ min_touches: Number(event.target.value) })
                  }
                />
              </label>
              <label className="block text-[11px] text-slate-500">
                Max levels
                <input
                  type="number"
                  min={1}
                  max={20}
                  className="mt-0.5 w-full bg-slate-800 text-slate-200 text-xs rounded px-2 py-1.5 border border-slate-700"
                  value={srControls.max_levels}
                  disabled={disabled}
                  onChange={(event) =>
                    void onSrControlsChange({ max_levels: Number(event.target.value) })
                  }
                />
              </label>
              <label className="block text-[11px] text-slate-500">
                Tolerance %
                <input
                  type="number"
                  min={0}
                  max={5}
                  step={0.01}
                  className="mt-0.5 w-full bg-slate-800 text-slate-200 text-xs rounded px-2 py-1.5 border border-slate-700"
                  value={srControls.tolerance_pct}
                  disabled={disabled}
                  onChange={(event) =>
                    void onSrControlsChange({
                      tolerance_pct: Number(event.target.value),
                    })
                  }
                />
              </label>
            </div>
            <p className="text-[10px] text-slate-600">
              Changing parameters refreshes levels when the overlay is on.
            </p>
          </div>

          <div className="border-t border-slate-800 pt-2 space-y-1.5">
            <div className="flex items-center justify-between">
              <span className="text-xs font-medium text-slate-300">Confluence</span>
              <button
                type="button"
                className={`text-xs px-2 py-1 rounded border ${
                  hasConfluence
                    ? 'border-fuchsia-700 text-fuchsia-300 bg-fuchsia-950/40'
                    : 'border-slate-700 text-slate-300 hover:bg-slate-800'
                }`}
                disabled={disabled || confluenceLoading || !onToggleConfluence}
                onClick={() => void onToggleConfluence?.()}
              >
                {confluenceLoading ? '…' : hasConfluence ? 'On' : 'Off'}
              </button>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-xs font-medium text-slate-300">Volume profile</span>
              <button
                type="button"
                className={`text-xs px-2 py-1 rounded border ${
                  hasVolumeProfile
                    ? 'border-violet-700 text-violet-300 bg-violet-950/40'
                    : 'border-slate-700 text-slate-300 hover:bg-slate-800'
                }`}
                disabled={disabled || volumeProfileLoading || !onToggleVolumeProfile}
                onClick={() => void onToggleVolumeProfile?.()}
              >
                {volumeProfileLoading ? '…' : hasVolumeProfile ? 'On' : 'Off'}
              </button>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-xs font-medium text-slate-300">Open exposure</span>
              <button
                type="button"
                className={`text-xs px-2 py-1 rounded border ${
                  hasExposure
                    ? 'border-amber-700 text-amber-300 bg-amber-950/40'
                    : 'border-slate-700 text-slate-300 hover:bg-slate-800'
                }`}
                disabled={disabled || exposureLoading || !onToggleExposure}
                onClick={() => void onToggleExposure?.()}
              >
                {exposureLoading ? '…' : hasExposure ? 'On' : 'Off'}
              </button>
            </div>
            <p className="text-[11px] text-slate-500">Exposure lines are read-only.</p>
          </div>

          <button
            type="button"
            className="w-full text-xs text-slate-400 hover:text-slate-200 py-1"
            onClick={() => setOpen(false)}
          >
            Close
          </button>
        </div>
      )}
    </div>
  )
}
