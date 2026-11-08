import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import type { ToolCatalogEntry } from '../lib/toolCatalog'
import { ToolOmissionNotice } from './ToolsRunnerPanel'

describe('ToolOmissionNotice', () => {
  it('renders the server-provided intentional-omission rationale', () => {
    const tool: ToolCatalogEntry = {
      name: 'wait_event',
      surface: 'intentional_omit',
      safety: {
        omit_rationale: 'Blocking waits have no HTTP cancellation contract. Use CLI or MCP.',
      },
    }

    const markup = renderToStaticMarkup(createElement(ToolOmissionNotice, { tool }))
    expect(markup).toContain('Not available in the synchronous Tools runner.')
    expect(markup).toContain('Blocking waits have no HTTP cancellation contract. Use CLI or MCP.')
  })
})
