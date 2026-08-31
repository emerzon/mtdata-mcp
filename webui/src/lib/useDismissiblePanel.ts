import { useEffect, useRef } from 'react'
import { useEscapeKey } from './useEscapeKey'

const DISMISSIBLE_PORTAL_SELECTOR = '[data-dismissible-panel-portal]'

export function isDismissiblePortalTarget(target: EventTarget | null): boolean {
  const candidate = target as {
    closest?: (selector: string) => unknown
    parentElement?: { closest?: (selector: string) => unknown } | null
  } | null
  return Boolean(
    candidate?.closest?.(DISMISSIBLE_PORTAL_SELECTOR)
      ?? candidate?.parentElement?.closest?.(DISMISSIBLE_PORTAL_SELECTOR)
  )
}

export function useDismissiblePanel(onClose: () => void, enabled = true) {
  const ref = useRef<HTMLDivElement>(null)
  useEscapeKey(enabled, onClose)

  useEffect(() => {
    if (!enabled) return
    const handleClick = (event: MouseEvent) => {
      if (isDismissiblePortalTarget(event.target)) return
      if (ref.current && !ref.current.contains(event.target as Node)) onClose()
    }
    document.addEventListener('mousedown', handleClick)
    return () => document.removeEventListener('mousedown', handleClick)
  }, [enabled, onClose])

  return ref
}
