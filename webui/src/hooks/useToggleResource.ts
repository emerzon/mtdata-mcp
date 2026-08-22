import { useCallback, useRef, useState } from 'react'
import { getErrorMessage } from '../api/client'

type ToggleResourceOptions<T> = {
  enabled: boolean
  fetchResource: () => Promise<T>
  isEmpty: (data: T) => boolean
  emptyMessage: string
}

export function useToggleResource<T>({
  enabled,
  fetchResource,
  isEmpty,
  emptyMessage,
}: ToggleResourceOptions<T>) {
  const [data, setData] = useState<T | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const dataRef = useRef(data)
  dataRef.current = data

  const fetch = useCallback(async () => {
    if (!enabled) return
    try {
      setIsLoading(true)
      setError(null)
      const next = await fetchResource()
      if (isEmpty(next)) {
        setError(emptyMessage)
        setData(null)
        return
      }
      setData(next)
    } catch (err) {
      setError(getErrorMessage(err))
      setData(null)
    } finally {
      setIsLoading(false)
    }
  }, [emptyMessage, enabled, fetchResource, isEmpty])

  const toggle = useCallback(async () => {
    if (!enabled) return
    if (dataRef.current !== null) {
      setData(null)
      setError(null)
      return
    }
    await fetch()
  }, [enabled, fetch])

  const reset = useCallback(() => {
    setData(null)
    setError(null)
  }, [])

  return { data, isLoading, error, fetch, toggle, reset }
}
