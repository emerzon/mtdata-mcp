import { useQuery } from '@tanstack/react-query'
import { getErrorMessage, readyCheck } from '../api/client'
import { resolveConnectionStatus, type ConnectionReadyState } from '../lib/connectionStatus'

const POLL_MS = 30_000

export function ConnectionStatus() {
  const ready = useQuery({
    queryKey: ['api-ready'],
    queryFn: ({ signal }) => readyCheck(signal),
    refetchInterval: POLL_MS,
    retry: 1,
  })

  const readyState: ConnectionReadyState =
    ready.isPending && !ready.data && !ready.error
      ? 'pending'
      : ready.isError
        ? 'error'
        : ready.isSuccess && ready.data.ok
          ? 'ok'
          : ready.isSuccess
            ? 'mt5-not-ready'
            : 'error'

  const status = resolveConnectionStatus({
    readyState,
    readyMessage:
      ready.data?.payload?.message ||
      ready.data?.payload?.detail ||
      (typeof ready.data?.payload?.status === 'string' ? ready.data.payload.status : null) ||
      (ready.error ? getErrorMessage(ready.error) : null),
  })

  const colors: Record<string, string> = {
    checking: 'text-slate-400 border-slate-700 bg-slate-900/90',
    ok: 'text-emerald-300 border-emerald-800 bg-emerald-950/80',
    'api-down': 'text-rose-300 border-rose-800 bg-rose-950/80',
    'mt5-not-ready': 'text-amber-300 border-amber-800 bg-amber-950/80',
  }

  const dot: Record<string, string> = {
    checking: 'bg-slate-500 animate-pulse',
    ok: 'bg-emerald-400',
    'api-down': 'bg-rose-400',
    'mt5-not-ready': 'bg-amber-400',
  }

  return (
    <div
      className={`inline-flex items-center gap-1.5 rounded-lg border px-2 py-1 text-[11px] font-medium backdrop-blur-sm ${colors[status.kind]}`}
      title={status.hint}
      role="status"
      aria-live="polite"
      data-connection-kind={status.kind}
    >
      <span className={`h-1.5 w-1.5 rounded-full shrink-0 ${dot[status.kind]}`} aria-hidden />
      <span className="hidden sm:inline whitespace-nowrap">{status.label}</span>
      <span className="sm:hidden whitespace-nowrap">
        {status.kind === 'ok' ? 'OK' : status.kind === 'checking' ? '…' : '!'}
      </span>
    </div>
  )
}
