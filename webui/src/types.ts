// ============================================================================
// Core Data Types
// ============================================================================

export type Instrument = {
  symbol: string
  group?: string
  description?: string
}

export type HistoryBar = {
  time: number // epoch seconds UTC
  open: number
  high: number
  low: number
  close: number
  tick_volume?: number
  real_volume?: number
  volume?: number
  close_dn?: number // denoised close (when denoising applied)
  /** Indicator / derived columns from GET /history?indicators= */
  [column: string]: number | string | boolean | undefined
}

export type RuntimeTimezoneMeta = {
  utc?: {
    tz?: string | null
    now?: string
  }
  server?: {
    source?: string
    tz?: string | null
    offset_seconds?: number
    now?: string
  }
  client?: {
    tz?: string | null
    now?: string
  }
}

export type OutputWarning = {
  code: string
  message: string
  scope?: string
  [key: string]: unknown
}

export type OutputWarningLike = OutputWarning | string

export type HistoryResponse = {
  data: HistoryBar[]
  count: number
  timeframe?: string
  symbol?: string
  success?: boolean
  /** Display-normalized indicator columns present on each row. */
  indicator_columns?: string[]
  /** Normalized indicator spec that produced those columns. */
  indicators_spec?: string
  forming_candle_status?: 'included' | 'skipped' | 'detected' | 'none'
  forming_candle_index?: number
  data_as_of?: string
  data_as_of_basis?: string
  timestamp_format?: 'iso_utc' | 'iso_offset' | 'epoch_seconds'
  server_utc_offset_seconds?: number
  server_timezone?: string
  source?: { provider: string }
  warnings?: OutputWarningLike[]
  denoise_status?: string
  denoise_status_reason?: string
  denoise_applied?: boolean
  denoise_method?: string | string[]
  denoise_overwrote_columns?: string[]
  denoise_live_safe?: boolean
  meta?: {
    runtime?: {
      timezone?: RuntimeTimezoneMeta
    }
  }
}

// ============================================================================
// Support/Resistance & Pivot Types
// ============================================================================

export type SupportResistanceLevel = {
  type: 'support' | 'resistance'
  value: number
  touches: number
  episodes?: number
  score?: number
  distance?: number | null
  distance_pct?: number | null
  zone_low?: number | null
  zone_high?: number | null
  zone_width?: number | null
  zone_width_atr?: number | null
  first_touch?: string | null
  last_touch?: string | null
  dominant_source?: 'support' | 'resistance' | 'mixed'
  status?: string
  source_tests?: {
    support: number
    resistance: number
  }
  source_episodes?: {
    support: number
    resistance: number
  }
  avg_bounce_atr?: number | null
  avg_pretest_adx?: number | null
  breakout_analysis?: {
    decisive_break_count: number
    avg_breach_atr?: number | null
    last_break_time?: string | null
    role_reversal_count: number
  }
  score_breakdown?: {
    base?: number
    retests?: number
    bounce?: number
    adx?: number
    breakout_penalty?: number
    role_reversal_bonus?: number
    mtf_confirmation_bonus?: number
    total?: number
  }
  source_timeframes?: string[]
  merge_details?: {
    cross_timeframe_dedupe_count: number
    deduped_timeframes: string[]
  }
  episode_details?: Array<{
    type: 'support' | 'resistance' | 'mixed'
    touches: number
    first_touch?: string | null
    last_touch?: string | null
  }>
  timeframe_contributions?: Array<{
    timeframe: string
    weight: number
    raw_score: number
    weighted_score: number
    touches: number
    episodes?: number
    merge_mode?: 'full' | 'deduped'
  }>
}

export type PivotLevel = {
  level: string
  value: number
}

export type PivotResponse = {
  levels: PivotLevel[]
  period?: { start?: string; end?: string }
  method: string
  symbol: string
  timeframe: string
}

export type SupportResistanceResponse = {
  symbol: string
  timeframe: string
  mode?: string
  timeframes_analyzed?: string[]
  timeframe_weights?: Record<string, number>
  per_timeframe?: Array<{
    timeframe: string
    supports: number
    resistances: number
    current_price?: number | null
    window?: { start?: string | null; end?: string | null }
    effective_tolerance_pct?: number
    effective_reaction_bars?: number
    volatility_ratio?: number
    current_atr_pct?: number | null
    baseline_atr_pct?: number | null
  }>
  lookback?: number
  limit?: number
  method?: string
  tolerance_pct?: number
  effective_tolerance_pct?: number
  min_touches?: number
  qualification_basis?: 'episodes'
  max_levels?: number
  reaction_bars?: number
  effective_reaction_bars?: number
  adx_period?: number
  adaptive_mode?: 'atr_regime'
  volatility_ratio?: number
  current_atr_pct?: number | null
  baseline_atr_pct?: number | null
  current_price?: number | null
  window?: { start?: string | null; end?: string | null }
  scan_window?: { start?: string | null; end?: string | null }
  levels?: SupportResistanceLevel[]
  supports?: SupportResistanceLevel[]
  resistances?: SupportResistanceLevel[]
}

// ============================================================================
// Forecast Types
// ============================================================================

/** Compact dedicated-panel forecast row from POST /forecast/price (detail defaults to compact). */
export type CompactForecastRow = {
  time?: string
  value?: number
  price?: number
  return?: number
  lower_price?: number
  upper_price?: number
  lower?: number
  upper?: number
}

/** Compact dedicated-panel forecast payload. Tools runner uses generic invoke JSON, not this type. */
export type ForecastPayload = {
  forecast?: CompactForecastRow[]
  quantity?: 'price' | 'return' | 'volatility'
  forecast_status?: string
  signal_status?: string
  trust_level?: string
  trust_blockers?: string[]
  ci_status?: string
  forecast_mode?: string
  warnings?: OutputWarningLike[]
  uncertainty?: {
    status?: string
    mode?: string
    reason?: string
    recommended_tool?: string
    [key: string]: unknown
  }
  data_window?: {
    last_observation?: string | number
    forecast_start?: string | number
  }
  // client-only context
  __anchor?: number
  __kind?: 'full' | 'partial' | 'backtest'
}

export type VolatilityPayload = {
  symbol: string
  timeframe: string
  method: string
  horizon: number
  forecast_epoch?: number[]
  forecast_time?: string[]
  forecast_vol?: number[]
  volatility_per_bar?: number
  volatility_annualized?: number
  volatility_horizon?: number
}

// ============================================================================
// Method Metadata Types
// ============================================================================

export type ParamDef = {
  name: string
  type: string
  default?: unknown
  description?: string
}

export type MethodInfo = {
  method: string
  available: boolean
  requires: string[]
  description: string
  params: ParamDef[]
  supports?: { price?: boolean; return?: boolean; ci?: boolean }
}

export type MethodsMeta = {
  methods: MethodInfo[]
}

export type VolatilityMethodInfo = {
  method: string
  available: boolean
  requires: string[]
  description?: string
  params: ParamDef[]
  requires_proxy?: boolean
  valid_proxies?: string[]
}

export type VolatilityMethodsMeta = {
  methods: VolatilityMethodInfo[]
}

export type DenoiseMethodInfo = {
  method: string
  available: boolean
  requires?: string
  description: string
  params: ParamDef[]
  supports_causal?: boolean
  requires_causality_opt_in?: boolean
  supports?: { causality?: string[] }
  defaults?: {
    causality?: 'zero_phase' | 'causal'
    when?: 'pre_ti' | 'post_ti'
    keep_original?: boolean
  }
}

export type DenoiseMethodsMeta = {
  methods: DenoiseMethodInfo[]
}

export type WaveletsResponse = {
  available: boolean
  families: string[]
  wavelets: string[]
  by_family: Record<string, string[]>
}

export type StoredModelInfo = {
  id?: string
  model_id?: string
  method?: string
  symbol?: string
  timeframe?: string
  created_at?: string
  updated_at?: string
  path?: string
  [key: string]: unknown
}

export type ModelsResponse = {
  success?: boolean
  detail?: string
  count?: number
  models: StoredModelInfo[]
}

export type ReadyResponse = {
  status?: string
  ready?: boolean
  service?: string
  detail?: string
  message?: string
  mt5?: { connected?: boolean; error?: string }
  [key: string]: unknown
}

// ============================================================================
// Denoise Spec (for UI forms)
// ============================================================================

export type DenoiseSpecUI = {
  method?: string
  params?: Record<string, unknown>
  columns?: string | string[]
  when?: 'pre_ti' | 'post_ti'
  causality?: 'zero_phase' | 'causal'
  keep_original?: boolean
}

export type DimensionalityReductionSpecUI = {
  method: string
  params?: Record<string, unknown>
}

// ============================================================================
// API Request Bodies
// ============================================================================

export type ForecastPriceBody = {
  symbol: string
  timeframe?: string
  library?: 'native' | 'statsforecast' | 'sktime' | 'mlforecast' | 'pretrained'
  method?: string
  horizon?: number
  lookback?: number
  as_of?: string
  params?: Record<string, unknown>
  ci_alpha?: number
  quantity?: 'price' | 'return' | 'volatility'
  denoise?: DenoiseSpecUI
  features?: Record<string, unknown>
  dimred?: DimensionalityReductionSpecUI
  target_spec?: Record<string, unknown>
}

export type ForecastVolBody = {
  symbol: string
  timeframe?: string
  horizon?: number
  method?: string
  proxy?: string
  params?: Record<string, unknown>
  as_of?: string
  denoise?: DenoiseSpecUI
}

export type BacktestBody = {
  symbol: string
  timeframe?: string
  horizon?: number
  steps?: number
  spacing?: number
  methods?: string[]
  params_per_method?: Record<string, Record<string, unknown>>
  quantity?: 'price' | 'return' | 'volatility'
  denoise?: DenoiseSpecUI
  params?: Record<string, unknown>
  features?: Record<string, unknown>
  dimred?: DimensionalityReductionSpecUI
  slippage_bps?: number
  trade_threshold?: number
  detail?: 'compact' | 'full'
}

export type BacktestResult = {
  symbol: string
  timeframe: string
  horizon: number
  steps: number
  spacing: number
  success?: boolean
  complete_success?: boolean
  status?: string
  methods_total?: number
  methods_succeeded?: number
  methods_complete?: number
  methods_partial?: number
  methods_failed?: number
  complete_methods?: string[]
  partial_methods?: string[]
  failed_methods?: string[]
  anchor_tests_planned?: number
  anchor_tests_succeeded?: number
  anchor_tests_failed?: number
  warnings?: OutputWarningLike[]
  results?: Record<string, BacktestMethodResult>
  ranked_methods?: Array<BacktestMethodResult & { method: string }>
}

export type BacktestMethodResult = {
  success?: boolean
  complete_success?: boolean
  status?: string
  ranking_status?: string
  rank?: number
  avg_mae?: number
  avg_rmse?: number
  avg_directional_accuracy?: number
  successful_tests?: number
  failed_tests?: number
  num_tests?: number
  error?: string
  error_code?: string
  warnings?: OutputWarningLike[]
}

// ============================================================================
// Chart Overlay Types
// ============================================================================

export type TradeIdeaGate = {
  status: 'pass' | 'fail' | 'skip'
  reason?: string
}

export type TradeIdeaPayload = {
  success?: boolean
  symbol?: string
  timeframe?: string
  horizon?: number
  template?: string
  direction?: string
  suggested_direction?: string
  actionability?: 'preview_only' | 'research'
  narrative?: string
  geometry?: {
    entry?: number
    take_profit?: number
    stop_loss?: number
    direction?: string
  }
  sizing?: { suggested_volume?: number; candidate_valid?: boolean }
  preview?: {
    dry_run?: boolean
    preview_ok?: boolean
    would_send_order?: boolean
    skipped?: boolean
    blockers?: unknown[]
  }
  gates?: Record<string, TradeIdeaGate>
  partial_failure?: boolean
  failed_sections?: string[]
  section_errors?: Record<string, {
    reason?: string
    error_code?: string
    remediation?: string
  }>
  error?: string
}

export type ConfluenceResponse = {
  success?: boolean
  symbol?: string
  levels?: Array<{
    price: number
    type?: string
    role?: string
    score?: number
    source_families?: string[]
    source_count?: number
    record_count?: number
    range?: { low?: number; high?: number }
  }>
}

export type VolumeProfileResponse = {
  success?: boolean
  symbol?: string
  poc?: number
  vah?: number
  val?: number
}

export type ExposureResponse = {
  success?: boolean
  symbol?: string
  positions?: Array<{
    ticket?: number | string
    ticket_exact?: string
    identifier_encoding?: string
    type?: string
    volume?: number
    price?: number
    sl?: number
    tp?: number
  }>
  pending?: Array<{
    ticket?: number | string
    ticket_exact?: string
    identifier_encoding?: string
    type?: string
    volume?: number
    price?: number
    sl?: number
    tp?: number
  }>
}

export type RadarRow = {
  symbol: string
  bid?: number
  ask?: number
  mid?: number
  last?: number
  bar_close?: number
  spread?: number
  spread_pips?: number
  spread_pct?: number
  spread_quality?: string
  quote_usable_for_live_trading?: boolean
  quote_not_live_ready?: boolean
  data_stale?: boolean
  price_change_pct?: number
  live_price_change_pct?: number
  rsi?: number
  tick_volume?: number
}

export type RadarResponse = {
  success?: boolean
  timeframe?: string
  rank_by?: string
  rows?: RadarRow[]
  missing_symbols?: string[]
  seeded?: boolean
  partial_failure?: boolean
  warnings?: OutputWarningLike[]
  count?: number
  error?: string
}

export type SessionStripResponse = {
  success?: boolean
  account?: {
    login?: number | string
    server?: string
    company?: string
    equity?: number
    balance?: number
    currency?: string
    is_demo?: boolean
  }
  account_error?: string
  news?: Array<{ title: string; time?: string; source?: string; bucket?: string }>
  exposure_count?: number
  market_status?: {
    status?: string
    is_tradable?: boolean
    can_open_new_positions?: boolean
    reason?: string
  }
  partial_failure?: boolean
  failed_sections?: string[]
}

export type ChartOverlayPane = 'price' | 'rsi' | 'macd' | 'volume'

export type ChartOverlay = {
  name: string
  points: { time: number; value: number; color?: string }[]
  color?: string
  lineWidth?: number
  lineStyle?: 'solid' | 'dashed' | 'dotted'
  priceScaleId?: string
  label?: string
  pane?: ChartOverlayPane
  kind?: 'line' | 'histogram'
  referenceLines?: Array<{ price: number; color: string; title: string }>
}

// ============================================================================
// Metrics
// ============================================================================

export type AnchorMetrics = {
  overlap: number
  mae: number
  mape: number
  rmse: number
  dirAcc: number | null
}

export type Tick = {
  success: boolean
  symbol: string
  time: string
  time_epoch: number
  timezone?: string
  bid: number
  ask: number
  mid?: number
  last?: number | null
  spread?: number
  spread_pips?: number
  spread_points?: number
  usable_for_live_trading?: boolean
  source?: { provider: string }
  warnings?: OutputWarningLike[]
}
