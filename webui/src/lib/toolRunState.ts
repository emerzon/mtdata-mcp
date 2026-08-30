export type ToolRunIdentity = Readonly<{
  generation: number
  toolName: string
}>

export type ToolRunGate = {
  begin: (toolName: string) => ToolRunIdentity
  invalidate: () => void
  isCurrent: (identity: ToolRunIdentity, selectedName: string | null) => boolean
}

/** Keep asynchronous tool output bound to the selection that started it. */
export function createToolRunGate(): ToolRunGate {
  let generation = 0

  return {
    begin(toolName) {
      generation += 1
      return { generation, toolName }
    },
    invalidate() {
      generation += 1
    },
    isCurrent(identity, selectedName) {
      return identity.generation === generation && identity.toolName === selectedName
    },
  }
}
