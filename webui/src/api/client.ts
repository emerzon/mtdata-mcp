import axios from 'axios'
import { setTimeframeSeconds } from '../lib/timeframes'
import type {
  HistoryBar,
  HistoryResponse,
  Instrument,
  Tick,
  MethodsMeta,
  VolatilityMethodsMeta,
  DenoiseMethodsMeta,
  WaveletsResponse,
  ModelsResponse,
  ReadyResponse,
  ForecastPayload,
  VolatilityPayload,
  PivotResponse,
  SupportResistanceResponse,
  DenoiseSpecUI,
  ForecastPriceBody,
  ForecastVolBody,
  BacktestBody,
  BacktestResult,
  ConfluenceResponse,
  ExposureResponse,
  RadarResponse,
  SessionStripResponse,
  TradeIdeaPayload,
  VolumeProfileResponse,
} from '../types'
import type { PivotMethod, SrQueryParams } from '../lib/overlayParams'
import type { ToolCatalogEntry } from '../lib/toolCatalog'

const apiOrigin = String(import.meta.env.VITE_API_BASE ?? '').replace(/\/$/, '')

export const api = axios.create({
  baseURL: `${apiOrigin}/api/v1`,
})

type ApiTokenListener = () => void

let activeApiToken = ''
const apiTokenListeners = new Set<ApiTokenListener>()

/** Return whether this tab currently sends an API bearer token. */
export function getApiTokenConfigured(): boolean {
  return Boolean(activeApiToken)
}

/** Subscribe React controls to the in-memory API credential state. */
export function subscribeApiToken(listener: ApiTokenListener): () => void {
  apiTokenListeners.add(listener)
  return () => apiTokenListeners.delete(listener)
}

export function setApiToken(token: string): void {
  const value = token.trim()
  if (value) {
    api.defaults.headers.common.Authorization = `Bearer ${value}`
  } else {
    delete api.defaults.headers.common.Authorization
  }
  if (value === activeApiToken) return
  activeApiToken = value
  apiTokenListeners.forEach((listener) => listener())
}

/**
 * Standardized error extraction from axios errors.
 */
export function getErrorMessage(error: unknown): string {
  if (axios.isAxiosError(error)) {
    const data: unknown = error.response?.data
    const message = extractErrorText(data)
    return message ?? error.message ?? 'The API request failed'
  }
  if (error instanceof Error) {
    return error.message
  }
  return extractErrorText(error) ?? 'An unknown error occurred'
}

function extractErrorText(value: unknown): string | null {
  if (typeof value === 'string') return value.trim() || null
  if (typeof value === 'number' || typeof value === 'boolean') return String(value)
  if (Array.isArray(value)) {
    const messages = value
      .map((item) => extractErrorText(item))
      .filter((item): item is string => Boolean(item))
    return messages.length ? messages.join('; ') : null
  }
  if (!value || typeof value !== 'object') return null

  const record = value as Record<string, unknown>
  for (const key of ['detail', 'error', 'message', 'msg']) {
    const message = extractErrorText(record[key])
    if (message) return message
  }
  try {
    return JSON.stringify(value)
  } catch {
    return null
  }
}

// ============================================================================
// Timeframes & Instruments
// ============================================================================

export async function getTimeframes(): Promise<string[]> {
  const { data } = await api.get<{
    timeframes: string[]
    seconds?: Record<string, number>
  }>('timeframes')
  setTimeframeSeconds(data.seconds)
  return data.timeframes ?? []
}

export async function searchInstruments(search?: string, limit?: number, signal?: AbortSignal): Promise<Instrument[]> {
  const { data } = await api.get<{ items: Instrument[] }>('instruments', {
    params: { search, limit },
    signal,
  })
  return data.items ?? []
}

// ============================================================================
// History Data
// ============================================================================

export type HistoryParams = {
  symbol: string
  timeframe: string
  limit: number
  start?: string
  end?: string
  denoise?: DenoiseSpecUI
  include_incomplete?: boolean
  indicators?: string
  ohlcv?: string
}

export async function getHistory(params: HistoryParams, signal?: AbortSignal): Promise<HistoryResponse> {
  const query: Record<string, unknown> = {
    symbol: params.symbol,
    timeframe: params.timeframe,
    limit: params.limit,
    start: params.start,
    end: params.end,
    include_incomplete: params.include_incomplete,
    timestamp_format: 'epoch',
    indicators: params.indicators,
    ohlcv: params.ohlcv,
  }

  const dn = params.denoise
  if (dn?.method) {
    query.denoise_method = dn.method
    const extras: Record<string, unknown> = {}
    if (dn.params) extras.params = dn.params
    if (dn.columns) extras.columns = dn.columns
    if (dn.when) extras.when = dn.when
    // Always forward causality when present so non-causal methods (l1_trend, etc.)
    // can opt into zero_phase; omit only when unset (server chooses causal default).
    if (dn.causality === 'zero_phase' || dn.causality === 'causal') {
      extras.causality = dn.causality
    }
    if (typeof dn.keep_original === 'boolean') extras.keep_original = dn.keep_original
    if (Object.keys(extras).length) {
      query.denoise_params = JSON.stringify(extras)
    }
  }

  const { data } = await api.get<HistoryResponse>('history', { params: query, signal })
  return {
    ...data,
    data: data.data ?? [],
  }
}

export async function getTick(symbol: string, signal?: AbortSignal): Promise<Tick> {
  const { data } = await api.get<Tick>('tick', { params: { symbol }, signal })
  return data
}

// ============================================================================
// Forecast Methods Metadata
// ============================================================================

export async function getMethods(): Promise<MethodsMeta> {
  const { data } = await api.get<MethodsMeta>('methods')
  return data
}

export async function getVolatilityMethods(): Promise<VolatilityMethodsMeta> {
  const { data } = await api.get<VolatilityMethodsMeta>('volatility/methods')
  return data
}

export async function getDenoiseMethods(): Promise<DenoiseMethodsMeta> {
  const { data } = await api.get<DenoiseMethodsMeta>('denoise/methods')
  return data
}

export async function getWavelets(): Promise<WaveletsResponse> {
  const { data } = await api.get<WaveletsResponse>('denoise/wavelets')
  return data
}

export async function getModels(method?: string, signal?: AbortSignal): Promise<ModelsResponse> {
  const { data } = await api.get<ModelsResponse>('models', {
    params: method ? { method } : undefined,
    signal,
  })
  return {
    ...data,
    models: Array.isArray(data?.models) ? data.models : [],
    count: typeof data?.count === 'number' ? data.count : Array.isArray(data?.models) ? data.models.length : 0,
  }
}

// ============================================================================
// Forecasting
// ============================================================================

export async function forecastPrice(body: ForecastPriceBody): Promise<ForecastPayload> {
  const { data } = await api.post<ForecastPayload>('forecast/price', body)
  return data
}

export async function forecastVolatility(body: ForecastVolBody): Promise<VolatilityPayload> {
  const { data } = await api.post<VolatilityPayload>('forecast/volatility', body)
  return data
}

export async function runBacktest(body: BacktestBody): Promise<BacktestResult> {
  const { data } = await api.post<BacktestResult>('backtest', body)
  return data
}

// ============================================================================
// Technical Analysis
// ============================================================================

export type PivotParams = {
  symbol: string
  timeframe: string
  method?: PivotMethod
}

export async function getPivots(params: PivotParams): Promise<PivotResponse> {
  const { data } = await api.get<PivotResponse>('pivots', { params })
  return data
}

export type SupportResistanceParams = Pick<SrQueryParams, 'symbol'> &
  Partial<Omit<SrQueryParams, 'symbol'>> & {
  max_distance_pct?: number
  volume_weighting?: 'off' | 'auto'
  reaction_bars?: number
  adx_period?: number
  decay_half_life_bars?: number
  }

export async function getSupportResistance(
  params: SupportResistanceParams
): Promise<SupportResistanceResponse> {
  const { data } = await api.get<SupportResistanceResponse>('support-resistance', { params })
  return data
}

export async function getConfluence(params: {
  symbol: string
  pivot_timeframe?: string
  sr_timeframe?: string
}): Promise<ConfluenceResponse> {
  const { data } = await api.get<ConfluenceResponse>('confluence', { params })
  return data
}

export async function getVolumeProfile(params: {
  symbol: string
  timeframe?: string
}): Promise<VolumeProfileResponse> {
  const { data } = await api.get<VolumeProfileResponse>('volume-profile', { params })
  return data
}

export async function getExposure(symbol: string): Promise<ExposureResponse> {
  const { data } = await api.get<ExposureResponse>('exposure', { params: { symbol } })
  return data
}

export async function getRadar(params: {
  symbols?: string
  timeframe?: string
  rank_by?: string
  limit?: number
}): Promise<RadarResponse> {
  const { data } = await api.get<RadarResponse>('radar', { params })
  return {
    ...data,
    rows: Array.isArray(data?.rows) ? data.rows : [],
  }
}

export async function getSessionStrip(symbol?: string): Promise<SessionStripResponse> {
  const { data } = await api.get<SessionStripResponse>('session-strip', {
    params: symbol ? { symbol } : undefined,
  })
  return data
}

export async function composeTradeIdea(body: {
  symbol: string
  timeframe?: string
  horizon?: number
  direction?: 'auto' | 'long' | 'short'
  template?: 'quick' | 'standard'
  risk_pct?: number
  as_of?: string
  detail?: string
}): Promise<TradeIdeaPayload> {
  const { data } = await api.post<TradeIdeaPayload>('trade-ideas', body)
  return data
}

// ============================================================================
// Readiness
// ============================================================================

/**
 * MT5 readiness probe. Resolves with the JSON body on HTTP 200 and the
 * intentional 503 (MT5 not ready). Transport errors, 401/403, and unexpected
 * 5xx reject so the UI can treat those as API down.
 */
export async function readyCheck(signal?: AbortSignal): Promise<{ ok: boolean; payload: ReadyResponse }> {
  const { data, status } = await api.get<ReadyResponse>('ready', {
    signal,
    validateStatus: (next) => (next >= 200 && next < 300) || next === 503,
  })
  const payload = data && typeof data === 'object' ? data : {}
  const ok = status >= 200 && status < 300
  return { ok, payload }
}

// ============================================================================
// MCP tool catalog + generic invoke
// ============================================================================

export type ToolsListPagination = {
  total?: number
  returned?: number
  offset?: number
  limit?: number
  has_more?: boolean
  more_available?: number
}

export type ToolsListResponse = {
  success?: boolean
  count?: number
  detail?: string
  categories?: Record<string, string[]>
  surfaces?: Record<string, number>
  pagination?: ToolsListPagination
  tools: ToolCatalogEntry[]
}

/** Compact catalog page size used by the Tools runner index fetch. */
export const TOOL_CATALOG_INDEX_LIMIT = 500

export type ToolDetailResponse = {
  success?: boolean
  tool: ToolCatalogEntry
}

export type ToolInvokeResponse = {
  success?: boolean
  tool?: string
  surface?: string
  result?: unknown
}

export async function listTools(
  params?: {
    category?: string
    search?: string
    include_fields?: boolean
    detail?: 'compact' | 'standard' | 'full'
    limit?: number
    offset?: number
  },
  signal?: AbortSignal
): Promise<ToolsListResponse> {
  const { data } = await api.get<ToolsListResponse>('tools', {
    params: {
      detail: params?.detail ?? 'compact',
      limit: params?.limit ?? TOOL_CATALOG_INDEX_LIMIT,
      offset: params?.offset,
      category: params?.category || undefined,
      search: params?.search || undefined,
      include_fields: params?.include_fields || undefined,
    },
    signal,
  })
  const tools = Array.isArray(data?.tools) ? data.tools : []
  return {
    ...data,
    tools,
    count: typeof data?.count === 'number' ? data.count : tools.length,
  }
}

function isNamedToolEntry(value: unknown): value is ToolCatalogEntry {
  if (!value || typeof value !== 'object') return false
  const name = (value as { name?: unknown }).name
  return typeof name === 'string' && Boolean(name.trim())
}

export async function getTool(toolName: string, signal?: AbortSignal): Promise<ToolDetailResponse> {
  const { data } = await api.get<ToolDetailResponse>(`tools/${encodeURIComponent(toolName)}`, {
    signal,
  })
  const tool = data?.tool
  if (!isNamedToolEntry(tool)) {
    throw new Error(`Tool detail for '${toolName}' omitted the required tool collection`)
  }
  return {
    ...data,
    tool,
  }
}

export async function invokeTool(
  toolName: string,
  body: { arguments?: Record<string, unknown>; confirm?: boolean }
): Promise<ToolInvokeResponse> {
  const { data } = await api.post<ToolInvokeResponse>(
    `tools/${encodeURIComponent(toolName)}/invoke`,
    {
      arguments: body.arguments ?? {},
      confirm: Boolean(body.confirm),
    }
  )
  return data
}
