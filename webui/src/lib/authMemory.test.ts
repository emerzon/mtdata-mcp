import { describe, expect, it, vi, beforeEach } from 'vitest'

/**
 * Auth token must stay in process memory only — never browser storage.
 * This test drives the shipped setApiToken module and asserts storage is untouched.
 */
describe('API auth token memory-only contract', () => {
  beforeEach(() => {
    vi.resetModules()
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

  it('clears the query cache after setApiToken instead of a reload callback', async () => {
    const fs = await import('node:fs')
    const path = await import('node:path')
    const authSrc = fs.readFileSync(path.join(__dirname, '../components/ApiAuthControl.tsx'), 'utf8')
    const toolbarSrc = fs.readFileSync(path.join(__dirname, '../components/ChartToolbar.tsx'), 'utf8')
    const appSrc = fs.readFileSync(path.join(__dirname, '../App.tsx'), 'utf8')
    expect(authSrc).toMatch(/useQueryClient\(\)/)
    expect(authSrc).toMatch(/setApiToken\(token\)[\s\S]*?queryClient\.clear\(\)/)
    expect(authSrc).toMatch(/setApiToken\(''\)[\s\S]*?queryClient\.clear\(\)/)
    expect(authSrc).not.toMatch(/onChange:\s*\(\)\s*=>/)
    expect(toolbarSrc).not.toMatch(/onAuthChange/)
    expect(appSrc).not.toMatch(/onAuthChange/)
  })
})
