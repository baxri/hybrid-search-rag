import { useState } from 'react'
import { rankJump } from '../lib/rankJumps'
import type { SearchTrace, Stage, StageName } from '../lib/types'

const PREVIEW_ROWS = 5

const STAGE_INFO: Record<StageName, { title: string; detail: string }> = {
  bm25: { title: 'BM25', detail: 'keyword match' },
  dense: { title: 'Dense', detail: 'text-embedding-3-small, cosine' },
  rrf: { title: 'RRF fusion', detail: 'k=60, merges both lists' },
  rerank: { title: 'Cohere rerank', detail: 'rerank-v4.0-fast, top 8' },
}

const JUMP_LABEL = { bm25: 'BM25', dense: 'Dense' }

interface Props {
  stage: Stage
  trace: SearchTrace
  citedAs: Map<string, number> // doc_id → [n] for docs the answer cites
}

export function StageCard({ stage, trace, citedAs }: Props) {
  const [showAll, setShowAll] = useState(false)
  const info = STAGE_INFO[stage.name]
  const rows = showAll ? stage.results : stage.results.slice(0, PREVIEW_ROWS)

  return (
    <div className="rounded-xl border border-slate-200 bg-white">
      <div className="flex items-baseline justify-between gap-2 border-b border-slate-100 px-3 py-2">
        <div className="min-w-0">
          <span className="font-semibold text-slate-800">{info.title}</span>
          <span className="ml-2 text-xs text-slate-400">{info.detail}</span>
        </div>
        <div className="shrink-0 font-mono text-xs text-slate-500">
          {stage.results.length} docs · {Math.round(stage.ms)} ms
        </div>
      </div>
      {stage.results.length === 0 ? (
        <div className="px-3 py-2 text-sm text-slate-400">No results.</div>
      ) : (
        <table className="w-full font-mono text-xs">
          <tbody>
            {rows.map((r) => {
              const n = citedAs.get(r.doc_id)
              const jump = stage.name === 'rerank' ? rankJump(trace, r.doc_id) : null
              return (
                <tr key={r.doc_id} className={n ? 'bg-blue-50' : ''}>
                  <td className="w-8 py-1 pl-3 text-slate-400">{r.rank}</td>
                  <td className="py-1 text-slate-700">
                    {r.doc_id}
                    {n !== undefined && (
                      <span className="ml-1.5 rounded bg-blue-100 px-1 font-semibold text-blue-700">
                        [{n}]
                      </span>
                    )}
                    {jump && (
                      <span className="ml-1.5 text-emerald-600">
                        ▲{jump.delta} vs {JUMP_LABEL[jump.from]}
                      </span>
                    )}
                  </td>
                  <td className="py-1 pr-3 text-right text-slate-500">{r.score.toPrecision(3)}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      )}
      {stage.results.length > PREVIEW_ROWS && (
        <button
          type="button"
          onClick={() => setShowAll((v) => !v)}
          className="w-full border-t border-slate-100 py-1.5 text-xs text-blue-600 hover:bg-slate-50"
        >
          {showAll ? 'Show top 5' : `Show all ${stage.results.length}`}
        </button>
      )}
    </div>
  )
}

export function StageSkeleton({ name }: { name: StageName }) {
  return (
    <div className="rounded-xl border border-slate-200 bg-white px-3 py-2">
      <div className="font-semibold text-slate-400">{STAGE_INFO[name].title}</div>
      <div className="mt-2 space-y-1.5">
        {[0, 1, 2].map((i) => (
          <div key={i} className="h-3 animate-pulse rounded bg-slate-100" />
        ))}
      </div>
    </div>
  )
}
