/**
 * Pure connection / readiness status for the chart workspace chrome.
 * Driven by a single /api/v1/ready query (or fetch failure).
 */

export type ConnectionStatusKind = 'checking' | 'ok' | 'api-down' | 'mt5-not-ready'

export type ConnectionStatus = {
  kind: ConnectionStatusKind
  /** Short toolbar label */
  label: string
  /** Longer hint for title / tooltip */
  hint?: string
}

export type ConnectionReadyState = 'pending' | 'ok' | 'mt5-not-ready' | 'error'

export type ConnectionStatusInput = {
  readyState: ConnectionReadyState
  readyMessage?: string | null
}

/**
 * Resolve one readiness query into a non-blocking chrome status.
 * pending=checking, 200=ok, 503=mt5-not-ready, transport/auth/unexpected=api-down.
 */
export function resolveConnectionStatus(input: ConnectionStatusInput): ConnectionStatus {
  if (input.readyState === 'pending') {
    return {
      kind: 'checking',
      label: 'Connecting…',
      hint: 'Checking API and MT5 readiness.',
    }
  }

  if (input.readyState === 'error') {
    return {
      kind: 'api-down',
      label: 'API down',
      hint: input.readyMessage?.trim() || 'Cannot reach /api/v1/ready. Is mtdata-webapi running?',
    }
  }

  if (input.readyState === 'mt5-not-ready') {
    return {
      kind: 'mt5-not-ready',
      label: 'MT5 not ready',
      hint:
        input.readyMessage?.trim() ||
        'API is up but MT5 readiness failed. Terminal may be closed or credentials missing.',
    }
  }

  return {
    kind: 'ok',
    label: 'Connected',
    hint: 'API reachable and MT5 readiness reported OK.',
  }
}
