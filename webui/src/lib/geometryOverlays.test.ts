import { describe, expect, it } from 'vitest'
import { LineStyle } from 'lightweight-charts'
import {
  confluencePriceLines,
  exposurePriceLines,
  ideaGeometryPriceLines,
  pivotPriceLines,
  supportResistancePriceLines,
  volumeProfilePriceLines,
} from './geometryOverlays'

describe('geometryOverlays', () => {
  it('maps pivot and support/resistance levels to native price lines', () => {
    const pivots = pivotPriceLines([
      { level: 'R1', value: 1.2 },
      { level: 'PP', value: 1.1 },
    ])
    expect(pivots).toEqual([
      expect.objectContaining({ title: 'R1', lineStyle: LineStyle.Dashed, lineWidth: 1 }),
      expect.objectContaining({ title: 'PP', color: '#facc15' }),
    ])
    const sr = supportResistancePriceLines([
      { type: 'resistance', value: 1.25, touches: 3 },
      { type: 'support', value: 1.05, touches: 2 },
    ])
    expect(sr.map((row) => row.title)).toEqual(['Res (3)', 'Sup (2)'])
    expect(sr[0].lineStyle).toBe(LineStyle.Dotted)
  })

  it('maps confluence types to colored lines', () => {
    const lines = confluencePriceLines([
      { price: 1.1, type: 'support' },
      { price: 1.2, type: 'resistance' },
    ])
    expect(lines).toHaveLength(2)
    expect(lines[0].title).toBe('Conf support')
    expect(lines[1].color).toBe('#fb7185')
  })

  it('maps volume profile poc/vah/val', () => {
    const lines = volumeProfilePriceLines({ poc: 1.1, vah: 1.12, val: 1.08 })
    expect(lines.map((row) => row.title)).toEqual(['POC', 'VAH', 'VAL'])
  })

  it('maps idea geometry and ignores invalid prices', () => {
    const lines = ideaGeometryPriceLines({
      entry: 1.1,
      take_profit: 1.12,
      stop_loss: 0,
    })
    expect(lines.map((row) => row.title)).toEqual(['Idea entry', 'Idea TP'])
  })

  it('maps read-only exposure without inventing prices', () => {
    const lines = exposurePriceLines(
      [{ ticket: 11, price: 1.1, sl: 1.09, tp: 1.12 }],
      [{ ticket: 22, price: 1.08 }]
    )
    expect(lines).toHaveLength(4)
    expect(lines.some((row) => row.title === 'Pend 22')).toBe(true)
  })
})
