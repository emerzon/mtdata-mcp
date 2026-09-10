import { describe, expect, it } from 'vitest'
import { getErrorMessage } from './client'

describe('getErrorMessage', () => {
  it('returns Error.message for plain errors', () => {
    expect(getErrorMessage(new Error('network down'))).toBe('network down')
  })

  it('extracts the canonical top-level API error envelope', () => {
    expect(getErrorMessage({
      success: false,
      error: 'Symbol missing',
      error_code: 'symbol_not_found',
      request_id: 'request-1',
    })).toBe('Symbol missing')
  })

  it('extracts nested detail/error/message fields from plain objects', () => {
    expect(getErrorMessage({ detail: 'Symbol missing' })).toBe('Symbol missing')
    expect(getErrorMessage({ error: { message: 'bad request' } })).toBe('bad request')
    expect(getErrorMessage({ msg: 'halted' })).toBe('halted')
  })

  it('joins array payloads', () => {
    expect(getErrorMessage([{ detail: 'a' }, { message: 'b' }])).toBe('a; b')
  })

  it('falls back for empty or unknown values', () => {
    expect(getErrorMessage(null)).toBe('An unknown error occurred')
    expect(getErrorMessage(undefined)).toBe('An unknown error occurred')
    expect(getErrorMessage('')).toBe('An unknown error occurred')
  })
})
