import type { DenoiseSpecUI, HistoryResponse, OutputWarningLike } from '../types'

export type ChartDenoiseFeedbackState = 'off' | 'pending' | 'applied' | 'skipped' | 'failed'

export type ChartDenoiseFeedback = {
  state: ChartDenoiseFeedbackState
  title: string
  warning?: string
}

function warningText(warning: OutputWarningLike): string | null {
  if (typeof warning === 'string') return warning.trim() || null
  return warning.message?.trim() || null
}

/** Normalize successful-response warnings for a visible chart status surface. */
export function responseWarningMessages(
  prefix: string,
  response: Pick<HistoryResponse, 'warnings'> | undefined,
  ignoredCodes: ReadonlySet<string> = new Set()
): string[] {
  const messages: string[] = []
  for (const warning of response?.warnings ?? []) {
    if (typeof warning !== 'string' && ignoredCodes.has(warning.code)) continue
    const message = warningText(warning)
    if (message) messages.push(`${prefix}: ${message}`)
  }
  return messages
}

function responseDenoiseStatus(response: HistoryResponse | undefined): string {
  const explicit = String(response?.denoise_status || '').trim().toLowerCase()
  if (explicit) return explicit
  if (response?.denoise_applied === true) return 'applied'
  if (response?.denoise_applied === false) return 'skipped'
  return ''
}

function effectiveMethod(response: HistoryResponse | undefined): string {
  const value = response?.denoise_method
  if (Array.isArray(value)) return value.map(String).filter(Boolean).join(', ')
  return String(value || '').trim()
}

/** Resolve requested versus effective denoise state across base and live history. */
export function resolveChartDenoiseFeedback(
  spec: DenoiseSpecUI | undefined,
  primary: HistoryResponse | undefined,
  live?: HistoryResponse
): ChartDenoiseFeedback {
  const requested = String(spec?.method || '').trim()
  if (!requested) return { state: 'off', title: 'Chart denoising' }

  const responses = [primary, live].filter((item): item is HistoryResponse => Boolean(item))
  const degraded = responses.find((item) => {
    const status = responseDenoiseStatus(item)
    return status === 'skipped' || status === 'failed'
  })
  if (degraded) {
    const status = responseDenoiseStatus(degraded) as 'skipped' | 'failed'
    const reason = String(degraded.denoise_status_reason || '').trim()
    const detail = reason || `The server reported denoise status '${status}'.`
    return {
      state: status,
      title: `Chart denoising ${status}: ${detail}`,
      warning: `Denoise: ${requested} was ${status}. ${detail}`,
    }
  }

  const applied = responses.find((item) => responseDenoiseStatus(item) === 'applied')
  if (!applied) {
    return {
      state: 'pending',
      title: `Denoising requested: ${requested}. Waiting for the history response.`,
    }
  }

  const effective = effectiveMethod(applied) || requested
  const differs = effective.toLowerCase() !== requested.toLowerCase()
  return {
    state: 'applied',
    title: `Chart denoising applied: ${effective}.`,
    ...(differs
      ? { warning: `Denoise: requested ${requested}, but the server applied ${effective}.` }
      : {}),
  }
}
