import { useMemo, useState } from 'react'
import type { AssistantTurn, Turn } from '../hooks/useChat'
import { citedNumbers, sourceElementId } from '../lib/citations'
import { AnswerText } from './AnswerText'
import { SourcesList } from './SourcesList'

interface Props {
  turn: Turn
  selected: boolean
  onSelect: (id: string) => void
  onRetry: (id: string) => void
}

export function Message({ turn, selected, onSelect, onRetry }: Props) {
  if (turn.role === 'user') {
    return (
      <div className="flex justify-end">
        <div className="max-w-[85%] rounded-2xl rounded-br-md bg-slate-800 px-4 py-2.5 whitespace-pre-wrap text-white">
          {turn.text}
        </div>
      </div>
    )
  }
  return <Answer turn={turn} selected={selected} onSelect={onSelect} onRetry={onRetry} />
}

function Status({ turn }: { turn: AssistantTurn }) {
  const pending = turn.searches.find((s) => !s.trace)
  const label = pending ? `Searching posts for “${pending.query}”…` : 'Thinking…'
  return (
    <div className="flex items-center gap-2 text-sm text-slate-500">
      <span className="h-2 w-2 animate-pulse rounded-full bg-blue-500" />
      {label}
    </div>
  )
}

function Answer({ turn, selected, onSelect, onRetry }: Props & { turn: AssistantTurn }) {
  const [expanded, setExpanded] = useState<Set<number>>(new Set())
  const valid = useMemo(() => new Set(turn.sources.map((s) => s.n)), [turn.sources])
  const cited = useMemo(() => {
    const byN = new Map(turn.sources.map((s) => [s.n, s]))
    return citedNumbers(turn.text, valid).map((n) => byN.get(n)!)
  }, [turn.text, turn.sources, valid])

  const toggle = (n: number) =>
    setExpanded((prev) => {
      const next = new Set(prev)
      if (next.has(n)) next.delete(n)
      else next.add(n)
      return next
    })

  const showSource = (n: number) => {
    setExpanded((prev) => new Set(prev).add(n))
    document
      .getElementById(sourceElementId(turn.id, n))
      ?.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
  }

  const waiting = turn.status === 'streaming' && !turn.text
  return (
    <div
      onClick={() => onSelect(turn.id)}
      className={`cursor-pointer rounded-2xl border bg-white px-5 py-4 text-slate-800 shadow-sm transition ${
        selected ? 'border-blue-300 ring-2 ring-blue-100' : 'border-slate-200 hover:border-slate-300'
      }`}
    >
      {waiting ? <Status turn={turn} /> : <AnswerText text={turn.text} valid={valid} onCite={showSource} />}
      {turn.status === 'streaming' && turn.text && (
        <span className="ml-0.5 inline-block h-4 w-1.5 animate-pulse bg-slate-400 align-middle" />
      )}
      {turn.status === 'done' && (
        <SourcesList turnId={turn.id} sources={cited} expanded={expanded} onToggle={toggle} />
      )}
      {turn.status === 'error' && (
        <div className="mt-3 flex items-center justify-between gap-3 rounded-lg bg-red-50 px-3 py-2 text-sm text-red-700">
          <span>{turn.error}</span>
          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation()
              onRetry(turn.id)
            }}
            className="rounded-md bg-white px-3 py-1 font-medium text-red-700 shadow-sm ring-1 ring-red-200 hover:bg-red-100"
          >
            Retry
          </button>
        </div>
      )}
    </div>
  )
}
