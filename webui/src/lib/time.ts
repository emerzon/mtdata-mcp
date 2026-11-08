export function formatDateTime(input: Date | number): string {
  const date = typeof input === 'number' ? new Date(input * 1000) : input
  return date.toISOString().slice(0, 19).replace('T', ' ')
}

export function toUtcSec(input: number | string): number {
  if (typeof input === 'number' && Number.isFinite(input)) return Math.floor(input)
  const s = String(input).trim()
  let iso = s.replace(' ', 'T')
  if (/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}$/.test(iso)) iso += ':00'
  if (/^\d{4}-\d{2}-\d{2}$/.test(iso)) iso += 'T00:00:00'
  if (!iso.endsWith('Z')) iso += 'Z'
  const ms = Date.parse(iso)
  if (Number.isNaN(ms)) throw new Error(`Invalid date string=${input}`)
  return Math.floor(ms / 1000)
}

export function formatEpochTime(epochSeconds: number, timeZone: string): string {
  return new Intl.DateTimeFormat(undefined, {
    timeZone,
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hourCycle: 'h23',
  }).format(new Date(epochSeconds * 1000))
}
