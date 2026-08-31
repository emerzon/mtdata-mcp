import { describe, expect, it } from 'vitest'
import { tradeIdeaSectionFailures } from './ideaFeedback'

describe('trade idea partial-result feedback', () => {
  it('preserves failed section reasons and remediation', () => {
    expect(tradeIdeaSectionFailures({
      partial_failure: true,
      failed_sections: ['volatility'],
      section_errors: {
        volatility: {
          reason: 'ewma unavailable',
          remediation: 'Check volatility inputs.',
        },
      },
    })).toEqual([{
      section: 'volatility',
      reason: 'ewma unavailable',
      remediation: 'Check volatility inputs.',
    }])
  })

  it('does not warn for a complete result', () => {
    expect(tradeIdeaSectionFailures({ partial_failure: false })).toEqual([])
  })
})
