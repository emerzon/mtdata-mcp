import { FormEvent, useState, useSyncExternalStore } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { getApiTokenConfigured, subscribeApiToken } from '../api/client'
import { replaceApiToken } from '../lib/authSession'

export function ApiAuthControl() {
  const queryClient = useQueryClient()
  const [open, setOpen] = useState(false)
  const [token, setToken] = useState('')
  const configured = useSyncExternalStore(
    subscribeApiToken,
    getApiTokenConfigured,
    getApiTokenConfigured
  )

  const apply = async (event: FormEvent) => {
    event.preventDefault()
    const nextToken = token
    setToken('')
    setOpen(false)
    await replaceApiToken(queryClient, nextToken)
  }

  const clear = async () => {
    setToken('')
    setOpen(false)
    await replaceApiToken(queryClient, '')
  }

  if (!open) {
    return (
      <button
        className={`toolbar-btn px-2 text-xs ${configured ? 'text-emerald-400' : 'text-slate-300'}`}
        onClick={() => setOpen(true)}
        title={configured ? 'API token configured for this tab' : 'Configure API token'}
      >
        {configured ? 'Auth ✓' : 'Auth'}
      </button>
    )
  }

  return (
    <form
      className="flex items-center gap-1"
      onSubmit={apply}
      aria-label="API authentication"
    >
      <input
        autoFocus
        type="password"
        value={token}
        onChange={(event) => setToken(event.target.value)}
        placeholder="API token"
        autoComplete="off"
        className="w-36 rounded border border-slate-700 bg-slate-950 px-2 py-1 text-xs text-slate-100 outline-none focus:border-sky-500"
      />
      <button type="submit" className="toolbar-btn px-2 text-xs" disabled={!token.trim()}>
        Apply
      </button>
      {configured && (
        <button type="button" className="toolbar-btn px-2 text-xs text-rose-400" onClick={clear}>
          Clear
        </button>
      )}
      <button type="button" className="toolbar-btn px-2 text-xs" onClick={() => setOpen(false)}>
        ×
      </button>
    </form>
  )
}
