import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { createElement, type ReactNode } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { denoiseMethodsQuery } from '../api/queries'
import type { DenoiseMethodInfo, DenoiseSpecUI } from '../types'
import { DenoiseModal } from './DenoiseModal'

vi.mock('react-dom', () => ({ createPortal: (children: ReactNode) => children }))

function renderModal(value: DenoiseSpecUI, methods?: DenoiseMethodInfo[]): string {
  vi.stubGlobal('document', { getElementById: () => null, body: {} })
  const client = new QueryClient()
  if (methods) client.setQueryData(denoiseMethodsQuery.queryKey, { methods })
  try {
    return renderToStaticMarkup(createElement(QueryClientProvider, { client },
      createElement(DenoiseModal, { open: true, value, onClose: vi.fn(), onApply: vi.fn() })
    ))
  } finally {
    client.clear()
  }
}

describe('DenoiseModal metadata readiness', () => {
  afterEach(() => vi.unstubAllGlobals())

  it('disables Apply for a legacy spec until method capabilities resolve, keeping Disable available', () => {
    const markup = renderModal({ method: 'wavelet' })
    expect(markup).toMatch(/<button[^>]*disabled=""[^>]*>Apply<\/button>/)
    expect(markup).toMatch(/<button(?:(?!disabled)[^>])*>Disable<\/button>/)
  })

  it('allows an explicit saved causality without waiting for metadata', () => {
    const markup = renderModal({ method: 'wavelet', causality: 'zero_phase' })
    expect(markup).toMatch(/<button(?:(?!disabled)[^>])*>Apply<\/button>/)
  })

  it('allows a legacy spec when server metadata provides its default', () => {
    const markup = renderModal({ method: 'new_server_filter' }, [{
      method: 'new_server_filter', available: true, description: '', params: [], supports_causal: false,
    }])
    expect(markup).toMatch(/<button(?:(?!disabled)[^>])*>Apply<\/button>/)
  })
})
