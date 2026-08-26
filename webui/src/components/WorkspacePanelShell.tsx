import type { ReactNode } from 'react'
import { forecastPanelPlacementClass, radarPanelPlacementClass, type LayoutBreakpoint } from '../lib/layout'
import { useEscapeKey } from '../lib/useEscapeKey'

type Props = {
  open: boolean
  onClose: () => void
  layoutBreakpoint?: LayoutBreakpoint
  side?: 'left' | 'right'
  label: string
  dismissLabel: string
  closeLabel: string
  header: ReactNode
  children: ReactNode
  /** When false, Escape and the mobile backdrop do not close the panel. */
  dismissEnabled?: boolean
  bodyClassName?: string
  dialogData?: Record<string, string>
}

function CloseIcon() {
  return (
    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
    </svg>
  )
}

export function WorkspacePanelShell({
  open,
  onClose,
  layoutBreakpoint = 'desktop',
  side = 'right',
  label,
  dismissLabel,
  closeLabel,
  header,
  children,
  dismissEnabled = true,
  bodyClassName = 'flex-1 overflow-y-auto overscroll-contain p-4 min-h-0',
  dialogData,
}: Props) {
  const canDismiss = open && dismissEnabled
  useEscapeKey(canDismiss, onClose)

  if (!open) return null

  const panelClass =
    side === 'left'
      ? radarPanelPlacementClass(layoutBreakpoint)
      : forecastPanelPlacementClass(layoutBreakpoint)

  return (
    <>
      {layoutBreakpoint === 'mobile' && (
        <button
          type="button"
          className="fixed inset-0 z-20 bg-slate-950/50 backdrop-blur-[1px]"
          aria-label={dismissLabel}
          onClick={canDismiss ? onClose : undefined}
        />
      )}
      <div
        className={`${panelClass} animate-slide-in-right`}
        role="dialog"
        aria-modal="true"
        aria-label={label}
        {...dialogData}
      >
        {layoutBreakpoint === 'mobile' && (
          <div className="flex justify-center pt-2 pb-1" aria-hidden>
            <div className="h-1 w-10 rounded-full bg-slate-700" />
          </div>
        )}
        <div className="flex items-center justify-between px-4 py-3 border-b border-slate-800 shrink-0">
          {header}
          <button
            className="text-slate-400 hover:text-slate-200 p-2 min-h-9 min-w-9"
            onClick={onClose}
            aria-label={closeLabel}
          >
            <CloseIcon />
          </button>
        </div>
        <div className={bodyClassName}>{children}</div>
      </div>
    </>
  )
}
