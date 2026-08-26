import { describe, expect, it } from 'vitest'
import { resolveConnectionStatus } from './connectionStatus'

describe('resolveConnectionStatus', () => {
  it('reports checking while readiness is pending', () => {
    const status = resolveConnectionStatus({ readyState: 'pending' })
    expect(status.kind).toBe('checking')
    expect(status.label).toMatch(/connect/i)
  })

  it('reports api-down on transport, auth, or unexpected failure', () => {
    const status = resolveConnectionStatus({
      readyState: 'error',
      readyMessage: 'Network Error',
    })
    expect(status.kind).toBe('api-down')
    expect(status.label).toMatch(/API/i)
    expect(status.hint).toContain('Network Error')
  })

  it('reports mt5-not-ready on the intentional 503', () => {
    const status = resolveConnectionStatus({
      readyState: 'mt5-not-ready',
      readyMessage: 'MT5 connection failed',
    })
    expect(status.kind).toBe('mt5-not-ready')
    expect(status.hint).toContain('MT5 connection failed')
  })

  it('reports ok when readiness returns 200', () => {
    const status = resolveConnectionStatus({ readyState: 'ok' })
    expect(status.kind).toBe('ok')
    expect(status.label).toMatch(/connected/i)
  })
})
