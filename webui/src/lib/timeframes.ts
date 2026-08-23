const FALLBACK_TIMEFRAME_SECONDS: Record<string, number> = {
  M1: 60,
  M2: 120,
  M3: 180,
  M4: 240,
  M5: 300,
  M6: 360,
  M10: 600,
  M12: 720,
  M15: 900,
  M20: 1200,
  M30: 1800,
  H1: 3600,
  H2: 7200,
  H3: 10800,
  H4: 14400,
  H6: 21600,
  H8: 28800,
  H12: 43200,
  D1: 86400,
  W1: 604800,
  MN1: 2592000,
}

let liveTimeframeSeconds: Record<string, number> | null = null

function normalizeTimeframe(tf: string): string {
  return tf?.trim().toUpperCase() ?? ''
}

export function setTimeframeSeconds(seconds: Record<string, number> | null | undefined): void {
  if (!seconds || Object.keys(seconds).length === 0) {
    liveTimeframeSeconds = null
    return
  }
  liveTimeframeSeconds = Object.fromEntries(
    Object.entries(seconds).map(([name, value]) => [normalizeTimeframe(name), Number(value)]),
  )
}

function secondsFor(tf: string): number | undefined {
  const key = normalizeTimeframe(tf)
  return liveTimeframeSeconds?.[key] ?? FALLBACK_TIMEFRAME_SECONDS[key]
}

export function tfSeconds(tf: string): number {
  return secondsFor(tf) ?? 3600
}

export function chartWorkspaceLivePollMs(tf: string): number {
  const seconds = secondsFor(tf)
  if (seconds === undefined) return 2000
  if (seconds <= 15 * 60) return 2000
  if (seconds <= 60 * 60) return 5000
  if (seconds <= 4 * 60 * 60) return 10000
  if (seconds <= 24 * 60 * 60) return 15000
  return 30000
}
