import { useCallback, useEffect, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { getConfluence, getErrorMessage, getExposure, getVolumeProfile } from '../api/client'
import type { ConfluenceResponse, VolumeProfileResponse } from '../types'
import { useToggleResource } from './useToggleResource'

const hasNoConfluenceLevels = (data: ConfluenceResponse) => !data.levels?.length
const hasNoVolumeProfileLevels = (data: VolumeProfileResponse) =>
  data.poc == null && data.vah == null && data.val == null
const EXPOSURE_POLL_MS = 15_000

export function useConfluenceLevels(symbol: string) {
  const fetchResource = useCallback(() => getConfluence({ symbol }), [symbol])
  return useToggleResource({
    enabled: Boolean(symbol),
    fetchResource,
    isEmpty: hasNoConfluenceLevels,
    emptyMessage: 'No confluence levels returned',
  })
}

export function useVolumeProfileLevels(symbol: string, timeframe: string) {
  const fetchResource = useCallback(
    () => getVolumeProfile({ symbol, timeframe }),
    [symbol, timeframe]
  )
  return useToggleResource({
    enabled: Boolean(symbol),
    fetchResource,
    isEmpty: hasNoVolumeProfileLevels,
    emptyMessage: 'No volume-profile levels returned',
  })
}

export function useExposureOverlay(symbol: string) {
  const [active, setActive] = useState(false)
  const query = useQuery({
    queryKey: ['exposure', symbol],
    queryFn: () => getExposure(symbol),
    enabled: Boolean(symbol) && active,
    refetchInterval: active ? EXPOSURE_POLL_MS : false,
  })

  useEffect(() => setActive(false), [symbol])

  const fetch = useCallback(async () => {
    if (!symbol) return
    setActive(true)
    await query.refetch()
  }, [query.refetch, symbol])
  const toggle = useCallback(() => {
    if (!symbol) return
    setActive((value) => !value)
  }, [symbol])
  const reset = useCallback(() => setActive(false), [])

  return {
    data: active ? query.data ?? null : null,
    isLoading: active && query.isFetching && query.data == null,
    error: active && query.error ? getErrorMessage(query.error) : null,
    updatedAt: active && query.dataUpdatedAt ? query.dataUpdatedAt : undefined,
    fetch,
    toggle,
    reset,
  }
}
