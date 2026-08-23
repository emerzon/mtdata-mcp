import { useEffect, useRef } from 'react'
import { useEscapeKey } from './useEscapeKey'

export function useDismissiblePanel(onClose: () => void, enabled = true) {
  const ref = useRef<HTMLDivElement>(null)
  useEscapeKey(enabled, onClose)

  useEffect(() => {
    if (!enabled) return
    const handleClick = (event: MouseEvent) => {
      if (ref.current && !ref.current.contains(event.target as Node)) onClose()
    }
    document.addEventListener('mousedown', handleClick)
    return () => document.removeEventListener('mousedown', handleClick)
  }, [enabled, onClose])

  return ref
}
