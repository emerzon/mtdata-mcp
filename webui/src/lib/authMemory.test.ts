import { QueryClient, QueryObserver } from '@tanstack/react-query'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

/**
 * Auth token must stay in process memory only — never browser storage.
 * This test drives the shipped setApiToken module and asserts storage is untouched.
 */
describe('API auth token memory-only contract', () => {
  beforeEach(() => {
    vi.resetModules()
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('setApiToken does not write to localStorage or sessionStorage', async () => {
    const localSet = vi.fn()
    const sessionSet = vi.fn()
    const localStorageMock = {
      getItem: vi.fn(),
      setItem: localSet,
      removeItem: vi.fn(),
      clear: vi.fn(),
    }
    const sessionStorageMock = {
      getItem: vi.fn(),
      setItem: sessionSet,
      removeItem: vi.fn(),
      clear: vi.fn(),
    }
    vi.stubGlobal('localStorage', localStorageMock)
    vi.stubGlobal('sessionStorage', sessionStorageMock)

    const { setApiToken } = await import('../api/client')
    setApiToken('secret-token-value')
    setApiToken('')

    expect(localSet).not.toHaveBeenCalled()
    expect(sessionSet).not.toHaveBeenCalled()

    // Ensure client source does not reference storage keys for tokens either
    const fs = await import('node:fs')
    const path = await import('node:path')
    const clientSrc = fs.readFileSync(path.join(__dirname, '../api/client.ts'), 'utf8')
    const authSrc = fs.readFileSync(path.join(__dirname, '../components/ApiAuthControl.tsx'), 'utf8')
    expect(clientSrc).not.toMatch(/localStorage|sessionStorage/)
    expect(authSrc).not.toMatch(/localStorage|sessionStorage/)
  })

  it('retries an active failed query after a token is applied', async () => {
    const { getApiTokenConfigured, setApiToken } = await import('../api/client')
    const { replaceApiToken } = await import('./authSession')
    setApiToken('')

    let calls = 0
    const queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    })
    const observer = new QueryObserver(queryClient, {
      queryKey: ['protected-workflow'],
      queryFn: async () => {
        calls += 1
        if (!getApiTokenConfigured()) throw new Error('unauthorized')
        return 'authenticated'
      },
    })
    let resolveInitialError: (() => void) | undefined
    const initialError = new Promise<void>((resolve) => {
      resolveInitialError = resolve
    })
    const unsubscribe = observer.subscribe((result) => {
      if (result.isError) resolveInitialError?.()
    })

    await initialError
    expect(calls).toBe(1)
    await replaceApiToken(queryClient, 'secret-token-value')

    expect(calls).toBe(2)
    expect(observer.getCurrentResult().data).toBe('authenticated')
    unsubscribe()
    queryClient.clear()
    setApiToken('')
  })
})
