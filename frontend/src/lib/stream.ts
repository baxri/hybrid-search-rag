import type { ChatEvent } from './types'

/** Yield one event per NDJSON line; a line may arrive split across network chunks. */
export async function* readNdjson(res: Response): AsyncGenerator<ChatEvent> {
  if (!res.body) return
  const reader = res.body.pipeThrough(new TextDecoderStream()).getReader()
  let buffer = ''
  while (true) {
    const { value, done } = await reader.read()
    if (done) break
    buffer += value
    const lines = buffer.split('\n')
    buffer = lines.pop() ?? ''
    for (const line of lines) {
      if (line.trim()) yield JSON.parse(line) as ChatEvent
    }
  }
  if (buffer.trim()) yield JSON.parse(buffer) as ChatEvent
}
