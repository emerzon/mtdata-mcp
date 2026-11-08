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

    const { api, getApiTokenConfigured, setApiToken, subscribeApiToken } = await import('../api/client')
    const listener = vi.fn()
    const unsubscribe = subscribeApiToken(listener)
    setApiToken('secret-token-value')
    expect(api.defaults.headers.common.Authorization).toBe('Bearer secret-token-value')
    expect(getApiTokenConfigured()).toBe(true)
    setApiToken('')
    expect(api.defaults.headers.common.Authorization).toBeUndefined()
    expect(getApiTokenConfigured()).toBe(false)
    expect(listener).toHaveBeenCalledTimes(2)
    unsubscribe()

    expect(localSet).not.toHaveBeenCalled()
    expect(sessionSet).not.toHaveBeenCalled()

    setApiToken('tab-only-token')
    vi.resetModules()
    const freshClient = await import('../api/client')
    expect(freshClient.getApiTokenConfigured()).toBe(false)
    expect(freshClient.api.defaults.headers.common.Authorization).toBeUndefined()
    expect(listener).toHaveBeenCalledTimes(2)
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
