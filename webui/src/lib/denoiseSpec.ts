/**
 * Pure helpers for chart-workspace denoise specs (history API query shape).
 */

import type { DenoiseMethodInfo, DenoiseSpecUI } from '../types'

export function defaultDenoiseCausality(
  methodMeta?: DenoiseMethodInfo | null
): 'causal' | 'zero_phase' | undefined {
  if (!methodMeta) return undefined
  if (
    methodMeta?.requires_causality_opt_in === true ||
    methodMeta?.supports_causal === false ||
    (Array.isArray(methodMeta?.supports?.causality) &&
      !methodMeta.supports.causality.includes('causal'))
  ) {
    return 'zero_phase'
  }
  if (methodMeta?.defaults?.causality === 'causal' || methodMeta?.defaults?.causality === 'zero_phase') {
    return methodMeta.defaults.causality
  }
  return 'causal'
}

export function denoiseNeedsMetadata(spec?: DenoiseSpecUI | null): boolean {
  return Boolean(spec?.method && spec.causality !== 'causal' && spec.causality !== 'zero_phase')
}

/**
 * Ensure a stored/loaded denoise spec includes causality before history requests.
 * Older UI saves only sent `{ method, params }` which 400s for non-causal methods.
 */
export function ensureChartDenoiseCausality(
  spec?: DenoiseSpecUI | null,
  methods: DenoiseMethodInfo[] = []
): DenoiseSpecUI | undefined {
  const name = spec?.method?.trim()
  if (!name || name.toLowerCase() === 'none') return undefined
  if (!denoiseNeedsMetadata(spec)) return spec ?? undefined
  const meta = methods.find((item) => item.method.toLowerCase() === name.toLowerCase())
  if (!meta) return spec ?? undefined
  return {
    ...spec,
    method: name,
    params: spec?.params ?? {},
    when: spec?.when ?? 'post_ti',
    causality: defaultDenoiseCausality(meta),
    keep_original: spec?.keep_original ?? true,
  }
}
