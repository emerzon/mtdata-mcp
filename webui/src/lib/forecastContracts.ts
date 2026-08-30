import type { DenoiseSpecUI, MethodInfo, ParamDef } from '../types'
import { coerceParamValue } from './toolCatalog'

export type ForecastSettings = {
  method: string
  horizon: number
  lookback: number | ''
  quantity: 'price' | 'return'
  ci_alpha: number
  paramsByMethod: Record<string, Record<string, unknown>>
  denoise?: DenoiseSpecUI
}

export type StoredForecastSettings = Partial<ForecastSettings> & {
  /** Pre-method-scoping storage fields retained only for one-time migration. */
  params?: Record<string, unknown>
  methodParams?: Record<string, unknown>
}

export const DEFAULT_FORECAST_SETTINGS: ForecastSettings = {
  method: 'theta',
  horizon: 12,
  lookback: '',
  quantity: 'price',
  ci_alpha: 0.1,
  paramsByMethod: {},
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function normalizeParamsByMethod(value: unknown): Record<string, Record<string, unknown>> {
  if (!isRecord(value)) return {}
  const normalized: Record<string, Record<string, unknown>> = {}
  for (const [method, params] of Object.entries(value)) {
    if (isRecord(params) && Object.keys(params).length) normalized[method] = { ...params }
  }
  return normalized
}

/** Load the current settings shape and scope legacy flat parameters to their saved method. */
export function normalizeForecastSettings(saved?: StoredForecastSettings): ForecastSettings {
  if (!saved) return { ...DEFAULT_FORECAST_SETTINGS, paramsByMethod: {} }
  const method = saved.method ?? DEFAULT_FORECAST_SETTINGS.method
  const paramsByMethod = normalizeParamsByMethod(saved.paramsByMethod)
  const legacyParams = isRecord(saved.params)
    ? saved.params
    : isRecord(saved.methodParams)
      ? saved.methodParams
      : undefined
  if (legacyParams && Object.keys(legacyParams).length && !paramsByMethod[method]) {
    paramsByMethod[method] = { ...legacyParams }
  }
  return {
    ...DEFAULT_FORECAST_SETTINGS,
    method,
    horizon: saved.horizon ?? DEFAULT_FORECAST_SETTINGS.horizon,
    lookback: saved.lookback ?? DEFAULT_FORECAST_SETTINGS.lookback,
    quantity: saved.quantity ?? DEFAULT_FORECAST_SETTINGS.quantity,
    ci_alpha: saved.ci_alpha ?? DEFAULT_FORECAST_SETTINGS.ci_alpha,
    paramsByMethod,
    denoise: saved.denoise,
  }
}

export function forecastMethodParams(
  settings: ForecastSettings,
  definitions?: ParamDef[]
): Record<string, unknown> | undefined {
  const params = settings.paramsByMethod[settings.method]
  if (!params || !Object.keys(params).length) return undefined
  if (definitions === undefined) return params
  return filterParamRecord(params, definitions)
}

/** Update one method-scoped form value. Empty input removes the parameter entirely. */
export function updateParameterValue(
  params: Record<string, unknown>,
  name: string,
  rawValue: string,
  type?: string
): Record<string, unknown> {
  const next = { ...params }
  if (!rawValue.trim()) delete next[name]
  else next[name] = coerceParamValue(rawValue, type)
  return next
}

export function updateMethodParameter(
  paramsByMethod: Record<string, Record<string, unknown>>,
  method: string,
  name: string,
  rawValue: string,
  type?: string
): Record<string, Record<string, unknown>> {
  const next = { ...paramsByMethod }
  const methodParams = updateParameterValue(next[method] ?? {}, name, rawValue, type)
  if (Object.keys(methodParams).length) next[method] = methodParams
  else delete next[method]
  return next
}

/** Parameters that every selected method declares with the same input type. */
export function sharedBacktestParamDefs(
  selectedMethods: string[],
  catalog: MethodInfo[]
): ParamDef[] {
  if (!selectedMethods.length) return []
  const selected = selectedMethods.map((name) => catalog.find((method) => method.method === name))
  if (selected.some((method) => !method)) return []
  const [first, ...rest] = selected as MethodInfo[]
  return first.params.filter((definition) =>
    rest.every((method) =>
      method.params.some(
        (candidate) => candidate.name === definition.name && candidate.type === definition.type
      )
    )
  )
}

function filterParamRecord(
  params: Record<string, unknown>,
  definitions: ParamDef[]
): Record<string, unknown> | undefined {
  const allowed = new Set(definitions.map((definition) => definition.name))
  const filtered = Object.fromEntries(
    Object.entries(params).filter(([name]) => allowed.has(name))
  )
  return Object.keys(filtered).length ? filtered : undefined
}

/** Keep only visible metadata-backed values for the currently selected methods. */
export function scopedBacktestParams(
  selectedMethods: string[],
  catalog: MethodInfo[],
  sharedParams: Record<string, unknown>,
  paramsByMethod: Record<string, Record<string, unknown>>
): {
  params?: Record<string, unknown>
  params_per_method?: Record<string, Record<string, unknown>>
} {
  const params = filterParamRecord(sharedParams, sharedBacktestParamDefs(selectedMethods, catalog))
  const paramsPerMethod: Record<string, Record<string, unknown>> = {}
  for (const methodName of selectedMethods) {
    const method = catalog.find((item) => item.method === methodName)
    if (!method) continue
    const filtered = filterParamRecord(paramsByMethod[methodName] ?? {}, method.params)
    if (filtered) paramsPerMethod[methodName] = filtered
  }
  return {
    ...(params ? { params } : {}),
    ...(Object.keys(paramsPerMethod).length ? { params_per_method: paramsPerMethod } : {}),
  }
}
