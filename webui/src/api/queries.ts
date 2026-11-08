import { queryOptions } from '@tanstack/react-query'
import { getDenoiseMethods, getTimeframes } from './client'

export const timeframeCatalogQuery = queryOptions({
  queryKey: ['timeframes'],
  queryFn: getTimeframes,
})

export const denoiseMethodsQuery = queryOptions({
  queryKey: ['denoise_methods'],
  queryFn: getDenoiseMethods,
})
