import { describe, expect, it } from 'vitest'
import { chartWorkspaceLivePollMs } from './timeframes'

describe('chartWorkspaceLivePollMs', () => {
  it('polls faster on lower timeframes without a catalog', () => {
    expect(chartWorkspaceLivePollMs('M1')).toBe(2000)
    expect(chartWorkspaceLivePollMs('M15')).toBe(2000)
    expect(chartWorkspaceLivePollMs('h1')).toBe(5000)
    expect(chartWorkspaceLivePollMs('H4')).toBe(10000)
    expect(chartWorkspaceLivePollMs(' D1 ')).toBe(15000)
    expect(chartWorkspaceLivePollMs('W1')).toBe(30000)
  })

  it('uses the supplied catalog for new and existing timeframes without retaining global state', () => {
    expect(chartWorkspaceLivePollMs('CUSTOM', { custom: 1800 })).toBe(5000)
    expect(chartWorkspaceLivePollMs('M5', { m5: 3600 })).toBe(5000)
    expect(chartWorkspaceLivePollMs('M5')).toBe(2000)
    expect(chartWorkspaceLivePollMs('CUSTOM')).toBe(2000)
  })

  it('uses a short default for unknown frames', () => {
    expect(chartWorkspaceLivePollMs('XYZ')).toBe(2000)
  })
})
