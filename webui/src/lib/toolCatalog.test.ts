import { describe, expect, it } from 'vitest'
import {
  coerceParamValue,
  defaultParamValues,
  filterToolCatalog,
  formatToolResult,
  humanizeIdentifier,
  invocationNeedsConfirmation,
  parseBoolLike,
  shapeInvokeArguments,
  toolChangesTradingState,
  toolIsRunnable,
  uniqueCategories,
  type ToolCatalogEntry,
  type ToolField,
} from './toolCatalog'

const SAMPLE: ToolCatalogEntry[] = [
  {
    name: 'forecast_generate',
    category: 'forecast',
    description: 'Generate forecasts',
    surface: 'dedicated_ui',
  },
  {
    name: 'trade_place',
    category: 'trading',
    description: 'Place live order',
    surface: 'generic_runner',
    safety: { requires_confirmation: true, is_live_trade_mutation: true },
  },
  {
    name: 'market_depth_fetch',
    category: 'market',
    description: 'DOM depth',
    surface: 'generic_runner',
    enabled: false,
  },
]

describe('humanizeIdentifier', () => {
  it('turns snake_case into title words', () => {
    expect(humanizeIdentifier('trade_place')).toBe('Trade Place')
    expect(humanizeIdentifier('ci_alpha')).toBe('Ci Alpha')
  })
})

describe('toolChangesTradingState', () => {
  it('identifies successful tool runs that must refresh trading queries', () => {
    expect(toolChangesTradingState(SAMPLE[1])).toBe(true)
    expect(toolChangesTradingState(SAMPLE[0])).toBe(false)
    expect(toolChangesTradingState(null)).toBe(false)
  })
})

describe('filterToolCatalog', () => {
  it('filters by search and category using shipped entries', () => {
    expect(filterToolCatalog(SAMPLE, { search: 'trade' }).map((t) => t.name)).toEqual([
      'trade_place',
    ])
    expect(filterToolCatalog(SAMPLE, { category: 'forecast' }).map((t) => t.name)).toEqual([
      'forecast_generate',
    ])
  })
})

describe('uniqueCategories', () => {
  it('returns sorted unique categories', () => {
    expect(uniqueCategories(SAMPLE)).toEqual(['forecast', 'market', 'trading'])
  })
})

describe('defaultParamValues + shapeInvokeArguments', () => {
  const fields: ToolField[] = [
    { name: 'symbol', required: true, type: 'str' },
    { name: 'horizon', required: false, default: 12, type: 'int' },
    { name: 'params', required: false, type: 'Dict[str, Any]' },
    { name: 'async_mode', required: false, default: false, type: 'bool' },
  ]

  it('seeds defaults as form strings', () => {
    expect(defaultParamValues(fields)).toEqual({
      symbol: '',
      horizon: '12',
      params: '',
      async_mode: 'false',
    })
  })

  it('shapes invoke payload: omit empty optionals, coerce types', () => {
    const shaped = shapeInvokeArguments(fields, {
      symbol: 'EURUSD',
      horizon: '24',
      params: '{"alpha": 0.1}',
      async_mode: 'true',
    })
    expect(shaped).toEqual({
      symbol: 'EURUSD',
      horizon: 24,
      params: { alpha: 0.1 },
      async_mode: true,
    })
  })

  it('keeps required empty string so validation can fail server-side', () => {
    const shaped = shapeInvokeArguments(fields, {
      symbol: '',
      horizon: '',
      params: '',
      async_mode: '',
    })
    expect(shaped).toEqual({ symbol: '' })
  })
})

describe('coerceParamValue', () => {
  it.each([
    ['true', true], ['1', true], [' YES ', true], ['On', true],
    ['false', false], ['0', false], [' NO ', false], ['Off', false],
  ])('uses the same boolean tokens for form fields and dry-run values: %s', (value, expected) => {
    expect(coerceParamValue(String(value), 'bool')).toBe(expected)
    expect(parseBoolLike(value)).toBe(expected)
  })

  it('leaves invalid boolean text available for server validation', () => {
    expect(coerceParamValue('maybe', 'bool')).toBe('maybe')
    expect(parseBoolLike('maybe')).toBeUndefined()
    expect(coerceParamValue('false', 'str')).toBe('false')
  })

  it('parses bools, ints, and json', () => {
    expect(coerceParamValue('yes', 'bool')).toBe(true)
    expect(coerceParamValue('off', 'boolean')).toBe(false)
    expect(coerceParamValue('42', 'int')).toBe(42)
    expect(coerceParamValue('[1,2]', 'list')).toEqual([1, 2])
  })

  it('keeps uint64 identifiers outside MAX_SAFE_INTEGER as decimal strings', () => {
    expect(coerceParamValue('9007199254740991', 'integer')).toBe(9007199254740991)
    expect(coerceParamValue('9007199254740993', 'integer')).toBe('9007199254740993')
    expect(coerceParamValue('18446744073709551615', 'int')).toBe('18446744073709551615')
    expect(coerceParamValue('9007199254740993')).toBe('9007199254740993')
    expect(
      shapeInvokeArguments(
        [{ name: 'ticket', required: true, type: 'integer' }],
        { ticket: '9007199254740993' }
      )
    ).toEqual({ ticket: '9007199254740993' })
    expect(
      shapeInvokeArguments(
        [{ name: 'magic', required: false, type: 'integer' }],
        { magic: '18446744073709551615' }
      )
    ).toEqual({ magic: '18446744073709551615' })
  })
})

describe('invocationNeedsConfirmation', () => {
  const tradePlace: ToolCatalogEntry = {
    name: 'trade_place',
    safety: { requires_confirmation: true, is_live_trade_mutation: true },
  }
  const fields: ToolField[] = [
    { name: 'symbol', required: true, type: 'str' },
    { name: 'dry_run', required: false, default: true, type: 'bool' },
  ]

  it('does not require confirm for dry-run previews', () => {
    expect(
      invocationNeedsConfirmation(tradePlace, fields, { symbol: 'EURUSD', dry_run: 'true' })
    ).toBe(false)
    expect(
      invocationNeedsConfirmation(tradePlace, fields, { symbol: 'EURUSD', dry_run: '' })
    ).toBe(false)
  })

  it('requires confirm for live mutations and tools without dry_run', () => {
    expect(
      invocationNeedsConfirmation(tradePlace, fields, { symbol: 'EURUSD', dry_run: 'false' })
    ).toBe(true)
    expect(
      invocationNeedsConfirmation(
        { name: 'forecast_task_cancel', safety: { requires_confirmation: true } },
        [{ name: 'task_id', required: true, type: 'str' }],
        { task_id: 'task-1' }
      )
    ).toBe(true)
  })

  it('does not require confirm for non-mutating tools', () => {
    expect(
      invocationNeedsConfirmation(
        { name: 'tools_list', safety: { requires_confirmation: false } },
        [],
        {}
      )
    ).toBe(false)
  })
})

describe('formatToolResult', () => {
  it('pretty-prints objects', () => {
    expect(formatToolResult({ ok: true })).toContain('"ok"')
    expect(formatToolResult('plain')).toBe('plain')
  })
})

describe('toolIsRunnable', () => {
  it('blocks omit and disabled tools', () => {
    expect(toolIsRunnable(SAMPLE[0])).toBe(true)
    expect(toolIsRunnable(SAMPLE[2])).toBe(false)
    expect(toolIsRunnable({ name: 'x', surface: 'intentional_omit' })).toBe(false)
  })
})
