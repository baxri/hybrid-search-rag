import { sourceElementId } from '../lib/citations'
import type { Source } from '../lib/types'

interface Props {
  turnId: string
  sources: Source[] // only the cited ones, in citation order
  expanded: Set<number>
  onToggle: (n: number) => void
}

const PREVIEW_CHARS = 140

export function SourcesList({ turnId, sources, expanded, onToggle }: Props) {
  if (!sources.length) return null
  return (
    <div className="mt-4 border-t border-slate-100 pt-3">
      <div className="mb-2 text-xs font-medium tracking-wide text-slate-500 uppercase">Sources</div>
      <ol className="space-y-1.5">
        {sources.map((s) => {
          const open = expanded.has(s.n)
          const long = s.text.length > PREVIEW_CHARS
          return (
            <li key={s.n} id={sourceElementId(turnId, s.n)}>
              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation()
                  onToggle(s.n)
                }}
                className={`w-full rounded-lg px-2.5 py-1.5 text-left text-sm transition-colors ${
                  open ? 'bg-blue-50' : 'hover:bg-slate-50'
                }`}
              >
                <span className="mr-1.5 font-semibold text-blue-700">[{s.n}]</span>
                <span className="text-slate-700">
                  {open || !long ? s.text : `${s.text.slice(0, PREVIEW_CHARS).trimEnd()}…`}
                </span>
                <span className="ml-2 font-mono text-xs whitespace-nowrap text-slate-400">
                  doc {s.doc_id}
                </span>
              </button>
            </li>
          )
        })}
      </ol>
    </div>
  )
}
