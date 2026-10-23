import { expect, test } from 'vitest'
import { readNdjson } from './stream'
import type { ChatEvent } from './types'

function responseFromChunks(chunks: string[]): Response {
  const encoder = new TextEncoder()
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(encoder.encode(chunk))
      controller.close()
    },
  })
  return new Response(body)
}

async function collect(res: Response): Promise<ChatEvent[]> {
  const events: ChatEvent[] = []
  for await (const event of readNdjson(res)) events.push(event)
  return events
}

test('a line split across two chunks becomes one event', async () => {
  const res = responseFromChunks(['{"type":"text_d', 'elta","delta":"hi"}\n{"type":"done"}\n'])

  expect(await collect(res)).toEqual([{ type: 'text_delta', delta: 'hi' }, { type: 'done' }])
})

test('a final line without a trailing newline is still read', async () => {
  const res = responseFromChunks(['{"type":"done"}'])

  expect(await collect(res)).toEqual([{ type: 'done' }])
})
