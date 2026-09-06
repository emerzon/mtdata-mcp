import { describe, expect, it } from 'vitest'
import { createToolRunGate } from './toolRunState'

describe('tool run request gate', () => {
  it('rejects a changed request before an effect invalidates its generation', () => {
    const gate = createToolRunGate()
    const first = gate.begin('EURUSD:H1:12')
    expect(gate.isCurrent(first, 'EURUSD:H1:24')).toBe(false)
    expect(gate.isCurrent(first, null)).toBe(false)
  })

  it('keeps a newer run current when the older run finishes for identical inputs', async () => {
    const gate = createToolRunGate()
    let resolve!: (value: string) => void
    const pending = new Promise<string>((complete) => { resolve = complete })
    const first = gate.begin('EURUSD:H1:12')
    let result: string | null = null
    let loading = true
    const completion = pending.then((value) => {
      if (gate.isCurrent(first, 'EURUSD:H1:12')) {
        result = value
        loading = false
      }
    })
    const second = gate.begin('EURUSD:H1:12')
    resolve('stale forecast')
    await completion
    expect(result).toBeNull()
    expect(loading).toBe(true)
    expect(gate.isCurrent(second, 'EURUSD:H1:12')).toBe(true)
  })

  it('rejects errors after disposal even if the same request is selected again', async () => {
    const gate = createToolRunGate()
    let reject!: (reason: Error) => void
    const pending = new Promise<void>((_complete, fail) => { reject = fail })
    const first = gate.begin('EURUSD:H1:12')
    let error: unknown = null
    const completion = pending.catch((reason) => {
      if (gate.isCurrent(first, 'EURUSD:H1:12')) error = reason
    })
    gate.invalidate()
    const second = gate.begin('EURUSD:H1:12')
    reject(new Error('stale error'))
    await completion
    expect(error).toBeNull()
    expect(gate.isCurrent(second, 'EURUSD:H1:12')).toBe(true)
  })
})
