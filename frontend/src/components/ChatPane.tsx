import { useEffect, useMemo, useRef } from 'react'
import type { Turn } from '../hooks/useChat'
import { Composer } from './Composer'
import { EmptyState } from './EmptyState'
import { ExamplePills } from './ExamplePills'
import { Message } from './Message'

interface Props {
  turns: Turn[]
  selectedTurnId?: string
  onSend: (text: string) => void
  onRetry: (id: string) => void
  onSelect: (id: string) => void
}

export function ChatPane({ turns, selectedTurnId, onSend, onRetry, onSelect }: Props) {
  const scroller = useRef<HTMLDivElement>(null)
  const followBottom = useRef(true)
  const streaming = turns.some((t) => t.role === 'assistant' && t.status === 'streaming')
  const asked = useMemo(
    () => new Set(turns.filter((t) => t.role === 'user').map((t) => t.text)),
    [turns],
  )
  const sendFromBottom = (text: string) => {
    followBottom.current = true
    onSend(text)
  }

  // Follow the answer as it streams, unless the reader scrolled up to look at something.
  useEffect(() => {
    const el = scroller.current
    if (el && followBottom.current) el.scrollTop = el.scrollHeight
  }, [turns])

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div
        ref={scroller}
        onScroll={(e) => {
          const el = e.currentTarget
          followBottom.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80
        }}
        className="min-h-0 flex-1 overflow-y-auto"
      >
        <div className="mx-auto max-w-3xl space-y-4 px-4 py-6">
          {turns.length === 0 ? (
            <EmptyState onPick={onSend} />
          ) : (
            turns.map((t) => (
              <Message
                key={t.id}
                turn={t}
                selected={t.id === selectedTurnId}
                onSelect={onSelect}
                onRetry={onRetry}
              />
            ))
          )}
        </div>
      </div>
      <div className="mx-auto w-full max-w-3xl px-4 pb-4">
        {turns.length > 0 && (
          <ExamplePills asked={asked} disabled={streaming} onPick={sendFromBottom} />
        )}
        <Composer disabled={streaming} onSend={sendFromBottom} />
      </div>
    </div>
  )
}
