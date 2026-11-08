import { describe, expect, it } from 'vitest'
import type { DenoiseMethodInfo, DenoiseSpecUI } from '../types'
import { defaultDenoiseCausality, denoiseNeedsMetadata, ensureChartDenoiseCausality } from './denoiseSpec'

const method: DenoiseMethodInfo = {
  method: 'new_server_filter',
  available: true,
  description: 'A method introduced by the server',
  params: [],
  supports: { causality: ['zero_phase'] },
}

describe('denoise causality migration', () => {
  it('resolves legacy saved specs using server capabilities without a client method registry', () => {
    const saved: DenoiseSpecUI = { method: method.method, params: { strength: 3 } }
    const migrated = ensureChartDenoiseCausality(saved, [method])
    expect(migrated).toEqual({
      method: method.method,
      params: { strength: 3 },
      causality: 'zero_phase',
      when: 'post_ti',
      keep_original: true,
    })
    expect(saved.causality).toBeUndefined()
    expect(denoiseNeedsMetadata(migrated)).toBe(false)
  })

  it('leaves an unresolved saved spec intact until its metadata is available', () => {
    const saved: DenoiseSpecUI = { method: method.method, params: { strength: 3 } }
    expect(ensureChartDenoiseCausality(saved)).toBe(saved)
    expect(denoiseNeedsMetadata(ensureChartDenoiseCausality(saved))).toBe(true)
    expect(ensureChartDenoiseCausality(saved, [{ ...method, method: 'another_filter' }])).toBe(saved)
  })

  it('preserves explicit causality and advanced settings without requiring a catalog', () => {
    const saved: DenoiseSpecUI = {
      method: method.method,
      causality: 'causal',
      params: { strength: 3 },
      columns: ['high', 'low'],
      when: 'pre_ti',
      keep_original: false,
    }
    expect(ensureChartDenoiseCausality(saved)).toBe(saved)
    expect(ensureChartDenoiseCausality(saved, [method])).toBe(saved)
    expect(denoiseNeedsMetadata(saved)).toBe(false)
  })

  it('matches catalog names without changing existing advanced settings', () => {
    const migrated = ensureChartDenoiseCausality({
      method: ' NEW_SERVER_FILTER ',
      columns: ['high'],
      when: 'pre_ti',
      keep_original: false,
    }, [method])
    expect(migrated?.causality).toBe('zero_phase')
    expect(migrated?.columns).toEqual(['high'])
    expect(migrated?.when).toBe('pre_ti')
    expect(migrated?.keep_original).toBe(false)
  })

  it('does not require metadata when denoising is disabled', () => {
    expect(ensureChartDenoiseCausality()).toBeUndefined()
    expect(ensureChartDenoiseCausality({ method: 'none' })).toBeUndefined()
    expect(denoiseNeedsMetadata(undefined)).toBe(false)
  })
})

describe('defaultDenoiseCausality', () => {
  it('does not invent a default before metadata arrives', () => {
    expect(defaultDenoiseCausality()).toBeUndefined()
  })

  it.each([
    { supports_causal: false },
    { requires_causality_opt_in: true },
    { supports: { causality: ['zero_phase'] } },
  ])('honors server non-causal capability metadata: %j', (capability) => {
    expect(defaultDenoiseCausality({ ...method, supports: undefined, ...capability })).toBe('zero_phase')
  })

  it('honors defaults for dual-mode methods', () => {
    const dualMode = { ...method, supports: { causality: ['causal', 'zero_phase'] } }
    expect(defaultDenoiseCausality(dualMode)).toBe('causal')
    expect(defaultDenoiseCausality({ ...dualMode, defaults: { causality: 'zero_phase' } })).toBe('zero_phase')
  })
})
