import { act, renderHook, waitFor } from '@testing-library/react'
import { afterEach, expect, test, vi } from 'vitest'
import type { ChatEvent, ChatMessage, SearchTrace } from '../lib/types'
import { type AssistantTurn, useChat } from './useChat'

const trace: SearchTrace = { query: 'emergency fund', total_ms: 12, stages: [] }

function ndjson(events: ChatEvent[]): Response {
  return new Response(events.map((e) => JSON.stringify(e) + '\n').join(''))
}

/** Each call to fetch answers with the next script; returns the request bodies it saw. */
function mockFetch(...scripts: ChatEvent[][]): ChatMessage[][] {
  const bodies: ChatMessage[][] = []
  vi.stubGlobal(
    'fetch',
    vi.fn(async (_url: string, init: RequestInit) => {
      bodies.push(JSON.parse(init.body as string).messages)
      return ndjson(scripts[bodies.length - 1])
    }),
  )
  return bodies
}

function lastAssistant(turns: ReturnType<typeof useChat>['turns']): AssistantTurn {
  const turn = turns.at(-1)
  if (turn?.role !== 'assistant') throw new Error('last turn is not an assistant turn')
  return turn
}

afterEach(() => vi.unstubAllGlobals())

test('streams deltas into text, attaches the trace to its search, and finishes', async () => {
  mockFetch([
    { type: 'search_started', search_id: 1, query: 'emergency fund' },
    { type: 'search_result', search_id: 1, trace, sources: [{ n: 1, doc_id: 'd1', text: 'post' }] },
    { type: 'text_delta', delta: 'Keep it ' },
    { type: 'text_delta', delta: 'liquid [1].' },
    { type: 'done' },
  ])
  const { result } = renderHook(() => useChat())

  act(() => result.current.send('Where should I keep my emergency fund?'))

  await waitFor(() => expect(lastAssistant(result.current.turns).status).toBe('done'))
  const turn = lastAssistant(result.current.turns)
  expect(turn.text).toBe('Keep it liquid [1].')
  expect(turn.searches).toEqual([{ id: 1, query: 'emergency fund', trace }])
  expect(turn.sources).toEqual([{ n: 1, doc_id: 'd1', text: 'post' }])
  expect(result.current.selectedTurnId).toBe(turn.id)
})

test('an error keeps the partial text, and retry resends history without the failed turn', async () => {
  const bodies = mockFetch(
    [
      { type: 'text_delta', delta: 'Partial' },
      { type: 'error', message: 'Something went wrong while answering. Please retry.' },
    ],
    [{ type: 'text_delta', delta: 'Full answer.' }, { type: 'done' }],
  )
  const { result } = renderHook(() => useChat())

  act(() => result.current.send('Question?'))
  await waitFor(() => expect(lastAssistant(result.current.turns).status).toBe('error'))
  const failed = lastAssistant(result.current.turns)
  expect(failed.text).toBe('Partial')
  expect(failed.error).toBe('Something went wrong while answering. Please retry.')

  act(() => result.current.retry(failed.id))
  await waitFor(() => expect(lastAssistant(result.current.turns).status).toBe('done'))

  expect(bodies[1]).toEqual([{ role: 'user', content: 'Question?' }])
  expect(result.current.turns).toHaveLength(2)
  expect(lastAssistant(result.current.turns).text).toBe('Full answer.')
})

test('a stream that ends without done is reported as a lost connection', async () => {
  mockFetch([{ type: 'text_delta', delta: 'Half' }])
  const { result } = renderHook(() => useChat())

  act(() => result.current.send('Question?'))

  await waitFor(() => expect(lastAssistant(result.current.turns).status).toBe('error'))
  expect(lastAssistant(result.current.turns).error).toBe('Connection lost.')
})

test('the second request carries the whole earlier conversation', async () => {
  const bodies = mockFetch(
    [{ type: 'text_delta', delta: 'Use a HYSA [1].' }, { type: 'done' }],
    [{ type: 'text_delta', delta: 'CDs lock it up.' }, { type: 'done' }],
  )
  const { result } = renderHook(() => useChat())

  act(() => result.current.send('Where should I keep my emergency fund?'))
  await waitFor(() => expect(lastAssistant(result.current.turns).status).toBe('done'))
  act(() => result.current.send('what about CDs?'))
  await waitFor(() => expect(bodies).toHaveLength(2))

  expect(bodies[1]).toEqual([
    { role: 'user', content: 'Where should I keep my emergency fund?' },
    { role: 'assistant', content: 'Use a HYSA [1].' },
    { role: 'user', content: 'what about CDs?' },
  ])
})
