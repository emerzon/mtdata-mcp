import { afterEach, describe, expect, it } from 'vitest'
import { chartWorkspaceLivePollMs, setTimeframeSeconds, tfSeconds } from './timeframes'

describe('tfSeconds', () => {
  it('maps known timeframes and normalizes case', () => {
    expect(tfSeconds('M1')).toBe(60)
    expect(tfSeconds('h1')).toBe(3600)
    expect(tfSeconds(' D1 ')).toBe(86400)
  })

  it('falls back to H1 seconds for unknown frames', () => {
    expect(tfSeconds('UNKNOWN')).toBe(3600)
  })

  it('prefers live catalog seconds when set', () => {
    setTimeframeSeconds({ m5: 301 })
    expect(tfSeconds('M5')).toBe(301)
    setTimeframeSeconds(null)
    expect(tfSeconds('M5')).toBe(300)
  })
})

afterEach(() => {
  setTimeframeSeconds(null)
})

describe('chartWorkspaceLivePollMs', () => {
  it('polls faster on lower timeframes', () => {
    expect(chartWorkspaceLivePollMs('M1')).toBe(2000)
    expect(chartWorkspaceLivePollMs('M15')).toBe(2000)
    expect(chartWorkspaceLivePollMs('H1')).toBe(5000)
    expect(chartWorkspaceLivePollMs('H4')).toBe(10000)
    expect(chartWorkspaceLivePollMs('D1')).toBe(15000)
    expect(chartWorkspaceLivePollMs('W1')).toBe(30000)
  })

  it('uses a short default for unknown frames', () => {
    expect(chartWorkspaceLivePollMs('XYZ')).toBe(2000)
  })
})
