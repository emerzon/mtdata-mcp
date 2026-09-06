export type ToolRunIdentity = Readonly<{
  generation: number
  requestKey: string
}>

export type ToolRunGate = {
  begin: (requestKey: string) => ToolRunIdentity
  invalidate: () => void
  isCurrent: (identity: ToolRunIdentity, requestKey: string | null) => boolean
}

/** Keep asynchronous tool output bound to the request that started it. */
export function createToolRunGate(): ToolRunGate {
  let generation = 0

  return {
    begin(requestKey) {
      generation += 1
      return { generation, requestKey }
    },
    invalidate() {
      generation += 1
    },
    isCurrent(identity, requestKey) {
      return identity.generation === generation && identity.requestKey === requestKey
    },
  }
}
