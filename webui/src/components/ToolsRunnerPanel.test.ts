import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import { createToolRunGate } from '../lib/toolRunState'
import type { ToolCatalogEntry } from '../lib/toolCatalog'
import { ToolOmissionNotice } from './ToolsRunnerPanel'

describe('Tools runner request identity', () => {
  it('rejects a completion after the selected tool changes', () => {
    const gate = createToolRunGate()
    const firstRun = gate.begin('market_scan')
    gate.invalidate()

    expect(gate.isCurrent(firstRun, 'forecast_train')).toBe(false)

    const secondRun = gate.begin('forecast_train')
    expect(gate.isCurrent(firstRun, 'forecast_train')).toBe(false)
    expect(gate.isCurrent(secondRun, 'forecast_train')).toBe(true)
  })
})

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
