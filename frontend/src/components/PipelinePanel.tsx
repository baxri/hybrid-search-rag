import { type ReactNode, useMemo } from 'react'
import type { AssistantTurn, Search } from '../hooks/useChat'
import { citedNumbers } from '../lib/citations'
import type { StageName } from '../lib/types'
import { StageCard, StageSkeleton } from './StageCard'

const STAGES: StageName[] = ['bm25', 'dense', 'rrf', 'rerank']

function Note({ children }: { children: ReactNode }) {
  return <p className="text-sm leading-relaxed text-slate-500">{children}</p>
}

function SearchSteps({ search, citedAs }: { search: Search; citedAs: Map<string, number> }) {
  const { trace } = search
  return (
    <section className="space-y-2">
      <div className="rounded-xl bg-slate-900 px-3 py-2 font-mono text-xs text-slate-100">
        <div className="flex justify-between gap-2 text-slate-400">
          <span>tool call #{search.id}</span>
          <span>{trace ? `${Math.round(trace.total_ms)} ms total` : 'running…'}</span>
        </div>
        <div className="mt-1 break-words">
          <span className="text-sky-300">search_finance_posts</span>(
          <span className="text-amber-200">"{search.query}"</span>)
        </div>
      </div>
      {trace
        ? trace.stages.map((stage) => (
            <StageCard key={stage.name} stage={stage} trace={trace} citedAs={citedAs} />
          ))
        : STAGES.map((name) => <StageSkeleton key={name} name={name} />)}
    </section>
  )
}

/** What happened on the backend for one answer: tool calls and each retrieval stage. */
export function PipelinePanel({ turn }: { turn?: AssistantTurn }) {
  const citedAs = useMemo(() => {
    if (!turn) return new Map<string, number>()
    const valid = new Set(turn.sources.map((s) => s.n))
    const cited = new Set(citedNumbers(turn.text, valid))
    return new Map(turn.sources.filter((s) => cited.has(s.n)).map((s) => [s.doc_id, s.n]))
  }, [turn])

  return (
    <div className="space-y-4 p-4">
      <div>
        <h2 className="font-semibold text-slate-800">Behind the answer</h2>
        <p className="text-xs text-slate-500">
          Click any answer to see its search. Cited posts are highlighted.
        </p>
      </div>
      {!turn ? (
        <Note>
          Ask a question and this panel shows each step of the search: the agent's tool call, the
          BM25 keyword hits, the embedding hits, how RRF fused them, and how Cohere reranked the
          final 8.
        </Note>
      ) : turn.searches.length === 0 ? (
        <Note>
          {turn.status === 'streaming'
            ? 'Waiting for the model to decide whether to search…'
            : 'No search for this message.'}
        </Note>
      ) : (
        turn.searches.map((s) => <SearchSteps key={s.id} search={s} citedAs={citedAs} />)
      )}
    </div>
  )
}
