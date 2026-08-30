import type { QueryClient } from '@tanstack/react-query'
import { setApiToken } from '../api/client'

/** Replace the tab credential and re-run every mounted server workflow with it. */
export async function replaceApiToken(
  queryClient: QueryClient,
  token: string
): Promise<void> {
  await queryClient.cancelQueries()
  setApiToken(token)
  await queryClient.resetQueries()
}
