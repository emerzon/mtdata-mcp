import type {
  ForecastVolBody,
  VolatilityMethodInfo,
  VolatilityPayload,
} from '../types'

type VolatilityRequestInput = {
  symbol: string
  timeframe: string
  method: string
  horizon: number
  proxy: string
  asOf?: string
  methodInfo?: VolatilityMethodInfo
}

export type VolatilityResultMetric = {
  label: string
  percent: number
  primary: boolean
}

export function volatilityProxyForMethod(
  methodInfo: VolatilityMethodInfo | undefined,
  selectedProxy: string
): string | undefined {
  if (methodInfo?.requires_proxy !== true) return undefined
  const valid = methodInfo.valid_proxies ?? []
  if (!valid.length || valid.includes(selectedProxy)) return selectedProxy
  return valid[0]
}

export function buildVolatilityRequest({
  symbol,
  timeframe,
  method,
  horizon,
  proxy,
  asOf,
  methodInfo,
}: VolatilityRequestInput): ForecastVolBody {
  const request: ForecastVolBody = {
    symbol,
    timeframe,
    method,
    horizon,
    as_of: asOf,
  }
  const effectiveProxy = volatilityProxyForMethod(methodInfo, proxy)
  if (effectiveProxy) request.proxy = effectiveProxy
  return request
}

export function volatilityResultMetrics(
  result: VolatilityPayload,
  fallbackHorizon: number
): VolatilityResultMetric[] {
  const metrics: VolatilityResultMetric[] = []
  if (typeof result.volatility_horizon === 'number' && Number.isFinite(result.volatility_horizon)) {
    metrics.push({
      label: `${result.horizon || fallbackHorizon}-bar volatility`,
      percent: result.volatility_horizon * 100,
      primary: true,
    })
  }
  if (typeof result.volatility_annualized === 'number' && Number.isFinite(result.volatility_annualized)) {
    metrics.push({
      label: 'Annualized volatility',
      percent: result.volatility_annualized * 100,
      primary: false,
    })
  }
  return metrics
}
