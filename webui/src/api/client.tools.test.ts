import { describe, expect, it, vi, beforeEach } from 'vitest'

const getMock = vi.fn()
const postMock = vi.fn()

vi.mock('axios', () => {
  const instance = {
    get: (...args: unknown[]) => getMock(...args),
    post: (...args: unknown[]) => postMock(...args),
    interceptors: { request: { use: vi.fn() }, response: { use: vi.fn() } },
  }
  return {
    default: {
      create: () => instance,
      isAxiosError: () => false,
    },
  }
})

describe('tools client adapters', () => {
  beforeEach(() => {
    getMock.mockReset()
    postMock.mockReset()
    vi.resetModules()
  })

  it('listTools normalizes missing tools array and hits tools', async () => {
    getMock.mockResolvedValueOnce({ data: { success: true } })
    const { listTools, TOOL_CATALOG_INDEX_LIMIT } = await import('./client')
    const result = await listTools({ search: 'regime' })
    expect(String(getMock.mock.calls[0][0])).toBe('tools')
    expect(getMock.mock.calls[0][1]?.params?.search).toBe('regime')
    expect(getMock.mock.calls[0][1]?.params?.detail).toBe('compact')
    expect(getMock.mock.calls[0][1]?.params?.limit).toBe(TOOL_CATALOG_INDEX_LIMIT)
    expect(result.tools).toEqual([])
    expect(result.count).toBe(0)
  })

  it('invokeTool posts arguments and confirm flag', async () => {
    postMock.mockResolvedValueOnce({
      data: { success: true, tool: 'tools_list', result: { count: 1 } },
    })
    const { invokeTool } = await import('./client')
    const result = await invokeTool('tools_list', {
      arguments: { limit: 1 },
      confirm: false,
    })
    expect(String(postMock.mock.calls[0][0])).toBe('tools/tools_list/invoke')
    expect(postMock.mock.calls[0][1]).toEqual({
      arguments: { limit: 1 },
      confirm: false,
    })
    expect(result.success).toBe(true)
  })
})
