import { useCallback } from 'react'
import { getConfluence, getExposure, getVolumeProfile } from '../api/client'
import type { ConfluenceResponse, ExposureResponse, VolumeProfileResponse } from '../types'
import { useToggleResource } from './useToggleResource'

const hasNoConfluenceLevels = (data: ConfluenceResponse) => !data.levels?.length
const hasNoVolumeProfileLevels = (data: VolumeProfileResponse) =>
  data.poc == null && data.vah == null && data.val == null
const exposureIsNeverEmpty = (_data: ExposureResponse) => false

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
  const fetchResource = useCallback(() => getExposure(symbol), [symbol])
  return useToggleResource({
    enabled: Boolean(symbol),
    fetchResource,
    isEmpty: exposureIsNeverEmpty,
    emptyMessage: '',
  })
}
