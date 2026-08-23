import { describe, expect, it, vi, beforeEach } from 'vitest'

const getMock = vi.fn()

vi.mock('axios', () => {
  const instance = {
    get: (...args: unknown[]) => getMock(...args),
    post: vi.fn(),
    interceptors: { request: { use: vi.fn() }, response: { use: vi.fn() } },
  }
  return {
    default: {
      create: () => instance,
      isAxiosError: (error: unknown) => Boolean(error && typeof error === 'object' && (error as { isAxiosError?: boolean }).isAxiosError),
    },
  }
})

describe('getModels / readyCheck client surface', () => {
  beforeEach(() => {
    getMock.mockReset()
    vi.resetModules()
  })

  it('getModels normalizes missing models array and calls models', async () => {
    getMock.mockResolvedValueOnce({ data: { success: true } })
    const { getModels } = await import('./client')
    const result = await getModels('theta')
    expect(getMock).toHaveBeenCalled()
    const [path, config] = getMock.mock.calls[0]
    expect(String(path)).toBe('models')
    expect(config?.params).toEqual({ method: 'theta' })
    expect(result.models).toEqual([])
    expect(result.count).toBe(0)
  })

  it('getModels preserves models list and count', async () => {
    getMock.mockResolvedValueOnce({
      data: {
        models: [{ model_id: 'm1', method: 'theta' }],
        count: 1,
      },
    })
    const { getModels } = await import('./client')
    const result = await getModels()
    expect(result.models).toHaveLength(1)
    expect(result.count).toBe(1)
  })

  it('readyCheck treats non-2xx as not ok without throwing', async () => {
    getMock.mockResolvedValueOnce({
      status: 503,
      data: { status: 'not_ready', message: 'MT5 down' },
    })
    const { readyCheck } = await import('./client')
    const result = await readyCheck()
    expect(result.ok).toBe(false)
    expect(result.payload.message).toBe('MT5 down')
  })

  it('readyCheck marks 200 as ok', async () => {
    getMock.mockResolvedValueOnce({
      status: 200,
      data: { status: 'ready' },
    })
    const { readyCheck } = await import('./client')
    const result = await readyCheck()
    expect(result.ok).toBe(true)
  })

  it('readyCheck lets transport failures reject', async () => {
    getMock.mockRejectedValueOnce(new Error('network down'))
    const { readyCheck } = await import('./client')
    await expect(readyCheck()).rejects.toThrow('network down')
  })
})
