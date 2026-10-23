import { useCallback, useRef, useState } from 'react'
import { readNdjson } from '../lib/stream'
import type { ChatEvent, ChatMessage, SearchTrace, Source } from '../lib/types'

export interface UserTurn {
  role: 'user'
  id: string
  text: string
}

export interface Search {
  id: number
  query: string
  trace?: SearchTrace // undefined while the pipeline is still running
}

export interface AssistantTurn {
  role: 'assistant'
  id: string
  text: string
  searches: Search[]
  sources: Source[]
  status: 'streaming' | 'done' | 'error'
  error?: string
}

export type Turn = UserTurn | AssistantTurn

const CONNECTION_LOST = 'Connection lost.'

let lastId = 0
const newId = () => `t${++lastId}`

function newAssistantTurn(): AssistantTurn {
  return { role: 'assistant', id: newId(), text: '', searches: [], sources: [], status: 'streaming' }
}

/** The conversation as the backend wants it. Failed answers are left out. */
function toMessages(turns: Turn[]): ChatMessage[] {
  return turns
    .filter((t) => t.role === 'user' || t.status === 'done')
    .map((t) => ({ role: t.role, content: t.text }))
}

function applyEvent(turn: AssistantTurn, event: ChatEvent): AssistantTurn {
  switch (event.type) {
    case 'search_started':
      return { ...turn, searches: [...turn.searches, { id: event.search_id, query: event.query }] }
    case 'search_result':
      return {
        ...turn,
        searches: turn.searches.map((s) =>
          s.id === event.search_id ? { ...s, trace: event.trace } : s,
        ),
        // Each search sends only the sources it numbered for the first time.
        sources: [...turn.sources, ...event.sources],
      }
    case 'text_delta':
      return { ...turn, text: turn.text + event.delta }
    case 'error':
      return { ...turn, status: 'error', error: event.message }
    case 'done':
      return { ...turn, status: 'done' }
  }
}

export function useChat() {
  const [turns, setTurns] = useState<Turn[]>([])
  const [selectedTurnId, setSelectedTurnId] = useState<string | undefined>()
  // send/retry read the latest turns without waiting for a re-render.
  const turnsRef = useRef<Turn[]>([])

  const update = useCallback((next: (turns: Turn[]) => Turn[]) => {
    turnsRef.current = next(turnsRef.current)
    setTurns(turnsRef.current)
  }, [])

  const updateTurn = useCallback(
    (id: string, change: (turn: AssistantTurn) => AssistantTurn) =>
      update((all) => all.map((t) => (t.id === id && t.role === 'assistant' ? change(t) : t))),
    [update],
  )

  const stream = useCallback(
    async (messages: ChatMessage[], turnId: string) => {
      let finished = false
      try {
        const res = await fetch('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ messages }),
        })
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        for await (const event of readNdjson(res)) {
          updateTurn(turnId, (t) => applyEvent(t, event))
          if (event.type === 'done' || event.type === 'error') finished = true
        }
      } catch {
        // Network failure or a bad response; handled below like a dropped stream.
      }
      if (!finished) updateTurn(turnId, (t) => ({ ...t, status: 'error', error: CONNECTION_LOST }))
    },
    [updateTurn],
  )

  const send = useCallback(
    (text: string) => {
      const user: UserTurn = { role: 'user', id: newId(), text }
      const assistant = newAssistantTurn()
      const messages = toMessages([...turnsRef.current, user])
      update((all) => [...all, user, assistant])
      setSelectedTurnId(assistant.id)
      void stream(messages, assistant.id)
    },
    [stream, update],
  )

  const retry = useCallback(
    (turnId: string) => {
      const index = turnsRef.current.findIndex((t) => t.id === turnId)
      if (index < 0) return
      const before = turnsRef.current.slice(0, index)
      const assistant = newAssistantTurn()
      update(() => [...before, assistant])
      setSelectedTurnId(assistant.id)
      void stream(toMessages(before), assistant.id)
    },
    [stream, update],
  )

  return { turns, send, retry, selectedTurnId, select: setSelectedTurnId }
}
