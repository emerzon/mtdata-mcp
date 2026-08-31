import type { TradeIdeaPayload } from '../types'

export type TradeIdeaSectionFailure = {
  section: string
  reason: string
  remediation?: string
}

export function tradeIdeaSectionFailures(
  idea: TradeIdeaPayload | null | undefined
): TradeIdeaSectionFailure[] {
  if (!idea?.partial_failure) return []
  return (idea.failed_sections ?? []).map((section) => {
    const details = idea.section_errors?.[section]
    return {
      section,
      reason: details?.reason?.trim() || 'Section unavailable',
      ...(details?.remediation?.trim() ? { remediation: details.remediation.trim() } : {}),
    }
  })
}
