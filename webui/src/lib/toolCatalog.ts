/**
 * Pure catalog / param / safety helpers for the schema-driven Tools runner.
 */

export type ToolSurface = 'dedicated_ui' | 'generic_runner' | 'intentional_omit'

export type ToolField = {
  name: string
  required: boolean
  default?: unknown
  type?: string
  description?: string | null
}

export type JsonSchema = {
  type?: string
  description?: string | null
  default?: unknown
  enum?: unknown[]
  properties?: Record<string, JsonSchema>
  required?: string[]
  items?: JsonSchema
  anyOf?: JsonSchema[]
  oneOf?: JsonSchema[]
  $ref?: string
  $defs?: Record<string, JsonSchema>
}

export type ToolSafety = {
  requires_confirmation?: boolean
  is_live_trade_mutation?: boolean
  surface?: ToolSurface
  dedicated_path?: string
  omit_rationale?: string
  warning?: string
}

export type ToolCatalogEntry = {
  name: string
  category?: string
  description?: string
  surface?: ToolSurface
  parameters?: Record<string, string>
  input_schema?: JsonSchema
  safety?: ToolSafety
  enabled?: boolean
  enable_env?: string
  status?: string
  why_disabled?: string
  related_tools?: string[]
}

export function toolChangesTradingState(tool: ToolCatalogEntry | null | undefined): boolean {
  return tool?.safety?.is_live_trade_mutation === true
}

function schemaType(schema: JsonSchema): string {
  if (schema.type === 'array') return `list[${schema.items ? schemaType(schema.items) : 'any'}]`
  if (schema.type === 'object' || schema.properties) return 'json'
  if (schema.enum?.length) return schema.enum.map(String).join(' | ')
  const options = schema.anyOf ?? schema.oneOf
  if (options?.length) {
    return options.map(schemaType).filter((value, index, all) => value !== 'null' && all.indexOf(value) === index).join(' | ')
  }
  return schema.type ?? 'any'
}

/** Convert the canonical JSON Schema into the flat controls used by the runner. */
export function schemaToToolFields(schema: JsonSchema | undefined | null): ToolField[] {
  const required = new Set(schema?.required ?? [])
  return Object.entries(schema?.properties ?? {}).map(([name, property]) => ({
    name,
    required: required.has(name),
    ...(Object.prototype.hasOwnProperty.call(property, 'default') ? { default: property.default } : {}),
    type: schemaType(property),
    description: property.description,
  }))
}

export type ToolParamValues = Record<string, string>

/** Human label from snake_case tool/param names. */
export function humanizeIdentifier(raw: string): string {
  const text = String(raw || '').trim()
  if (!text) return ''
  return text
    .replace(/[_.-]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
    .replace(/\b\w/g, (ch) => ch.toUpperCase())
}

export function filterToolCatalog(
  tools: ToolCatalogEntry[],
  opts: { search?: string; category?: string } = {}
): ToolCatalogEntry[] {
  const search = (opts.search ?? '').trim().toLowerCase()
  const category = (opts.category ?? '').trim().toLowerCase()
  return tools.filter((tool) => {
    if (category && String(tool.category || '').toLowerCase() !== category) return false
    if (!search) return true
    const hay = [tool.name, tool.category, tool.description]
      .map((part) => String(part || '').toLowerCase())
      .join(' ')
    return hay.includes(search)
  })
}

export function uniqueCategories(tools: ToolCatalogEntry[]): string[] {
  const set = new Set<string>()
  for (const tool of tools) {
    const c = String(tool.category || '').trim()
    if (c) set.add(c)
  }
  return Array.from(set).sort((a, b) => a.localeCompare(b))
}

/** Build initial form values from field metadata (defaults as strings for inputs). */
export function defaultParamValues(fields: ToolField[] | undefined | null): ToolParamValues {
  const out: ToolParamValues = {}
  for (const field of fields ?? []) {
    if (!field?.name) continue
    if (field.default === undefined || field.default === null) {
      out[field.name] = ''
      continue
    }
    if (typeof field.default === 'string') {
      out[field.name] = field.default
    } else {
      try {
        out[field.name] = JSON.stringify(field.default)
      } catch {
        out[field.name] = String(field.default)
      }
    }
  }
  return out
}

/**
 * Coerce form strings into JSON-friendly argument values for POST /tools/{name}/invoke.
 * Empty optional strings are omitted; required empties stay as "" so the server can reject.
 */
export function shapeInvokeArguments(
  fields: ToolField[] | undefined | null,
  values: ToolParamValues
): Record<string, unknown> {
  const args: Record<string, unknown> = {}
  const fieldMap = new Map((fields ?? []).map((f) => [f.name, f]))

  for (const [name, raw] of Object.entries(values)) {
    const text = raw ?? ''
    const field = fieldMap.get(name)
    const trimmed = text.trim()
    if (!trimmed) {
      if (field?.required) {
        args[name] = ''
      }
      continue
    }
    args[name] = coerceParamValue(trimmed, field?.type)
  }
  return args
}

export function coerceParamValue(text: string, typeHint?: string): unknown {
  const t = (typeHint || '').toLowerCase()
  const value = text.trim()
  if (!value) return value

  if (t.includes('bool')) {
    const parsed = parseBoolLike(value)
    if (parsed !== undefined) return parsed
  }

  if (
    (value.startsWith('{') && value.endsWith('}')) ||
    (value.startsWith('[') && value.endsWith(']'))
  ) {
    try {
      return JSON.parse(value) as unknown
    } catch {
      // fall through
    }
  }

  if (t.includes('int') && !t.includes('float') && /^-?\d+$/.test(value)) {
    return coerceIntegerText(value)
  }
  if ((t.includes('float') || t.includes('number')) && /^-?\d+(\.\d+)?$/.test(value)) {
    return Number(value)
  }
  if (!t && /^-?\d+$/.test(value)) return coerceIntegerText(value)
  if (!t && /^-?\d+\.\d+$/.test(value)) return Number(value)

  return value
}

/** Keep integers outside MAX_SAFE_INTEGER as decimal strings for uint64 tickets/magic. */
export function coerceIntegerText(value: string): number | string {
  const asNumber = Number(value)
  if (Number.isSafeInteger(asNumber)) return asNumber
  return value
}

export function parseBoolLike(value: unknown): boolean | undefined {
  if (typeof value === 'boolean') return value
  if (value == null) return undefined
  const text = String(value).trim().toLowerCase()
  if (['true', '1', 'yes', 'on'].includes(text)) return true
  if (['false', '0', 'no', 'off'].includes(text)) return false
  return undefined
}

export function effectiveDryRun(
  fields: ToolField[] | undefined | null,
  values: ToolParamValues
): boolean | undefined {
  const args = shapeInvokeArguments(fields, values)
  if ('dry_run' in args) {
    const parsed = parseBoolLike(args.dry_run)
    if (parsed !== undefined) return parsed
  }
  const field = (fields ?? []).find((item) => item.name === 'dry_run')
  if (field && field.default !== undefined && field.default !== null) {
    return parseBoolLike(field.default)
  }
  return undefined
}

/** Confirm only when the current arguments can mutate (live dry_run=false or no preview). */
export function invocationNeedsConfirmation(
  tool: ToolCatalogEntry | null | undefined,
  fields: ToolField[] | undefined | null,
  values: ToolParamValues
): boolean {
  if (!tool?.safety?.requires_confirmation) return false
  return effectiveDryRun(fields, values) !== true
}

export function formatToolResult(result: unknown): string {
  if (result === undefined) return ''
  if (typeof result === 'string') return result
  try {
    return JSON.stringify(result, null, 2)
  } catch {
    return String(result)
  }
}

export function toolIsRunnable(tool: ToolCatalogEntry | null | undefined): boolean {
  if (!tool?.name) return false
  if (tool.surface === 'intentional_omit') return false
  if (tool.enabled === false) return false
  return true
}
