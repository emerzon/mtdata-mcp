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
  const fetchGeneration = useRef(0)

  const fetch = useCallback(async () => {
    if (!enabled) return
    const generation = ++fetchGeneration.current
    try {
      setIsLoading(true)
      setError(null)
      const next = await fetchResource()
      if (generation !== fetchGeneration.current) return
      if (isEmpty(next)) {
        setError(emptyMessage)
        setData(null)
        return
      }
      setData(next)
    } catch (err) {
      if (generation !== fetchGeneration.current) return
      setError(getErrorMessage(err))
      setData(null)
    } finally {
      if (generation === fetchGeneration.current) {
        setIsLoading(false)
      }
    }
  }, [emptyMessage, enabled, fetchResource, isEmpty])

  const toggle = useCallback(async () => {
    if (!enabled) return
    if (dataRef.current !== null) {
      fetchGeneration.current += 1
      setData(null)
      setError(null)
      setIsLoading(false)
      return
    }
    await fetch()
  }, [enabled, fetch])

  const reset = useCallback(() => {
    fetchGeneration.current += 1
    setData(null)
    setError(null)
    setIsLoading(false)
  }, [])

  return { data, isLoading, error, fetch, toggle, reset }
}
