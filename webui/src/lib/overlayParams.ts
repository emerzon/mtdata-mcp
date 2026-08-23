/**
 * Pure builders / clamps for pivot and support-resistance Web API query params.
 */

export const PIVOT_METHODS = ['classic', 'fibonacci', 'woodie', 'camarilla', 'demark'] as const
export type PivotMethod = (typeof PIVOT_METHODS)[number]

export const DEFAULT_PIVOT_METHOD: PivotMethod = 'classic'

export type SupportResistanceControls = {
  lookback: number
  min_touches: number
  max_levels: number
  /** Percentage points, e.g. 0.15 = 0.15% */
  tolerance_pct: number
}

export const DEFAULT_SR_CONTROLS: SupportResistanceControls = {
  lookback: 200,
  min_touches: 2,
  max_levels: 4,
  tolerance_pct: 0.15,
}

/** API allows lookback ge=100, le=20000 when provided. */
export function clampSrLookback(value: number): number {
  if (!Number.isFinite(value)) return DEFAULT_SR_CONTROLS.lookback
  return Math.min(20000, Math.max(100, Math.round(value)))
}

export function clampMinTouches(value: number): number {
  if (!Number.isFinite(value)) return DEFAULT_SR_CONTROLS.min_touches
  return Math.min(50, Math.max(1, Math.round(value)))
}

export function clampMaxLevels(value: number): number {
  if (!Number.isFinite(value)) return DEFAULT_SR_CONTROLS.max_levels
  return Math.min(20, Math.max(1, Math.round(value)))
}

export function clampTolerancePct(value: number): number {
  if (!Number.isFinite(value)) return DEFAULT_SR_CONTROLS.tolerance_pct
  return Math.min(5, Math.max(0, value))
}

export function normalizePivotMethod(value: string | undefined | null): PivotMethod {
  const v = (value ?? '').trim().toLowerCase()
  if ((PIVOT_METHODS as readonly string[]).includes(v)) return v as PivotMethod
  return DEFAULT_PIVOT_METHOD
}

export function normalizeSrControls(
  partial?: Partial<SupportResistanceControls> | null
): SupportResistanceControls {
  const base = { ...DEFAULT_SR_CONTROLS, ...(partial ?? {}) }
  return {
    lookback: clampSrLookback(base.lookback),
    min_touches: clampMinTouches(base.min_touches),
    max_levels: clampMaxLevels(base.max_levels),
    tolerance_pct: clampTolerancePct(base.tolerance_pct),
  }
}

export type PivotQueryParams = {
  symbol: string
  timeframe: string
  method: PivotMethod
}

export function buildPivotQuery(
  symbol: string,
  timeframe: string,
  method?: string | null
): PivotQueryParams {
  return {
    symbol,
    timeframe,
    method: normalizePivotMethod(method),
  }
}

export type SrQueryParams = {
  symbol: string
  timeframe: string
  lookback: number
  min_touches: number
  max_levels: number
  tolerance_pct: number
}

export function buildSupportResistanceQuery(
  symbol: string,
  timeframe: string,
  controls?: Partial<SupportResistanceControls> | null
): SrQueryParams {
  const c = normalizeSrControls(controls)
  return {
    symbol,
    timeframe,
    lookback: c.lookback,
    min_touches: c.min_touches,
    max_levels: c.max_levels,
    tolerance_pct: c.tolerance_pct,
  }
}
