import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { afterEach, describe, expect, it } from 'vitest'
import { getApiTokenConfigured, setApiToken } from '../api/client'
import { ApiAuthControl } from './ApiAuthControl'

function renderControl(): string {
  return renderToStaticMarkup(
    createElement(
      QueryClientProvider,
      { client: new QueryClient() },
      createElement(ApiAuthControl)
    )
  )
}

describe('ApiAuthControl', () => {
  afterEach(() => setApiToken(''))

  it('derives configured state from the stable tab credential across remounts', () => {
    setApiToken('')
    expect(renderControl()).toContain('Auth</button>')

    setApiToken('secret-token-value')
    expect(getApiTokenConfigured()).toBe(true)
    const remounted = renderControl()
    expect(remounted).toContain('Auth ✓')
    expect(remounted).not.toContain('secret-token-value')
  })
})
