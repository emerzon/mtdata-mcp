/**
 * Format a Date object or epoch timestamp to ISO-like string (YYYY-MM-DD HH:MM:SS).
 */
export function formatDateTime(input: Date | number): string {
  const date = typeof input === 'number' ? new Date(input * 1000) : input
  return date.toISOString().slice(0, 19).replace('T', ' ')
}
