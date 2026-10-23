import type { ReactNode } from 'react'
import { splitCitations } from '../lib/citations'

interface Props {
  text: string
  valid: Set<number>
  onCite: (n: number) => void
}

const BULLET = /^\s*(?:[-*•]|\d+[.)])\s+/

/** **bold** only; anything else stays as written. */
function inline(text: string, key: string): ReactNode[] {
  return text.split(/(\*\*[^*]+\*\*)/g).map((piece, i) =>
    piece.startsWith('**') && piece.endsWith('**') && piece.length > 4 ? (
      <strong key={`${key}-${i}`}>{piece.slice(2, -2)}</strong>
    ) : (
      piece
    ),
  )
}

function Line({ text, valid, onCite }: Props) {
  return splitCitations(text, valid).map((part, i) =>
    part.kind === 'text' ? (
      inline(part.text, String(i))
    ) : (
      <button
        key={i}
        type="button"
        onClick={(e) => {
          e.stopPropagation()
          onCite(part.n)
        }}
        className="mx-0.5 inline-flex h-4 min-w-4 -translate-y-1.5 items-center justify-center rounded bg-blue-100 px-1 align-baseline text-[10px] font-semibold text-blue-700 hover:bg-blue-200"
      >
        {part.n}
      </button>
    ),
  )
}

/** The model's answer: paragraphs, bullet/numbered lists, **bold**, and [n] citation badges. */
export function AnswerText({ text, valid, onCite }: Props) {
  const blocks: ReactNode[] = []
  let list: string[] = []
  const flushList = () => {
    if (!list.length) return
    blocks.push(
      <ul key={`ul-${blocks.length}`} className="ml-5 list-disc space-y-1">
        {list.map((item, i) => (
          <li key={i}>
            <Line text={item} valid={valid} onCite={onCite} />
          </li>
        ))}
      </ul>,
    )
    list = []
  }

  for (const line of text.split('\n')) {
    if (BULLET.test(line)) {
      list.push(line.replace(BULLET, ''))
      continue
    }
    flushList()
    if (line.trim()) {
      blocks.push(
        <p key={`p-${blocks.length}`}>
          <Line text={line.replace(/^#+\s*/, '')} valid={valid} onCite={onCite} />
        </p>,
      )
    }
  }
  flushList()
  return <div className="space-y-3 leading-relaxed">{blocks}</div>
}
