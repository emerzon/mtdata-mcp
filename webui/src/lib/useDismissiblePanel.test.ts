import { describe, expect, it, vi } from 'vitest'
import { isDismissiblePortalTarget } from './useDismissiblePanel'

describe('dismissible panel portal ownership', () => {
  it('treats content in an owned modal portal as inside the panel boundary', () => {
    const closest = vi.fn().mockReturnValue({})

    expect(isDismissiblePortalTarget({ closest } as unknown as EventTarget)).toBe(true)
    expect(closest).toHaveBeenCalledWith('[data-dismissible-panel-portal]')
  })

  it('allows ordinary outside targets to dismiss the panel', () => {
    expect(isDismissiblePortalTarget({ closest: () => null } as unknown as EventTarget)).toBe(false)
  })
})
