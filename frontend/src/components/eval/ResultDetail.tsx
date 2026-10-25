import { useState } from 'react'
import { citedNumbers, sourceElementId } from '../../lib/citations'
import type { QuestionResult, Verdict } from '../../lib/evalTypes'
import type { RankedDoc, SearchTrace, StageName } from '../../lib/types'
import { AnswerText } from '../AnswerText'
import { SourcesList } from '../SourcesList'

const VERDICT_STYLE: Record<Verdict, string> = {
  supported: 'bg-emerald-50 text-emerald-700',
  unsupported: 'bg-red-50 text-red-700',
  uncited: 'bg-amber-50 text-amber-700',
  not_a_claim: 'bg-slate-100 text-slate-600',
}

const STAGE_LABEL: Record<StageName, string> = {
  bm25: 'BM25',
  dense: 'Dense',
  rrf: 'RRF',
  rerank: 'Rerank',
}

/** Best rank of a post in each stage, across every search in the turn. */
function ranksOf(docId: string, searches: SearchTrace[]): string {
  const parts = (['bm25', 'dense', 'rrf', 'rerank'] as const).map((stage) => {
    const ranks = searches.flatMap((t) =>
      t.stages.filter((s) => s.name === stage).flatMap((s) => s.results),
    )
    const hit = ranks.filter((r) => r.doc_id === docId).map((r) => r.rank)
    return `${STAGE_LABEL[stage]} ${hit.length ? `#${Math.min(...hit)}` : '—'}`
  })
  return parts.join(' · ')
}

const SHOWN = 8 // per stage until expanded; BM25 and dense keep 50 candidates each

/** One stage's ranking as doc chips, labeled posts highlighted; long lists expand. */
function StageRow({
  label,
  results,
  labeled,
}: {
  label: string
  results: RankedDoc[]
  labeled: Set<string>
}) {
  const [all, setAll] = useState(false)
  const shown = all ? results : results.slice(0, SHOWN)
  return (
    <div className="flex items-start gap-2">
      <span className="w-14 shrink-0 pt-0.5 text-xs text-slate-500">{label}</span>
      <div className="flex flex-wrap gap-1.5">
        {shown.map((r) => (
          <span
            key={r.doc_id}
            className={`rounded px-1.5 py-0.5 font-mono text-xs ${
              labeled.has(r.doc_id)
                ? 'bg-emerald-100 font-semibold text-emerald-800'
                : 'bg-white text-slate-600 ring-1 ring-slate-200'
            }`}
          >
            #{r.rank} {r.doc_id}
          </span>
        ))}
        {results.length > SHOWN && (
          <button
            type="button"
            onClick={() => setAll(!all)}
            className="px-1 py-0.5 text-xs text-blue-600 hover:underline"
          >
            {all ? 'show less' : `+${results.length - SHOWN} more`}
          </button>
        )}
      </div>
    </div>
  )
}

/** A labeled post's ranks per stage, with its text behind a toggle. */
function LabeledPost({
  docId,
  text,
  searches,
}: {
  docId: string
  text: string | undefined
  searches: SearchTrace[]
}) {
  const [open, setOpen] = useState(false)
  return (
    <li>
      <div className="font-mono text-xs">
        doc {docId}: <span className="text-slate-600">{ranksOf(docId, searches)}</span>
        {text !== undefined && (
          <button
            type="button"
            onClick={() => setOpen(!open)}
            className="ml-2 font-sans text-blue-600 hover:underline"
          >
            {open ? 'hide text' : 'show text'}
          </button>
        )}
      </div>
      {open && (
        <p className="mt-1.5 rounded-lg border border-slate-200 bg-white px-3 py-2 whitespace-pre-line text-slate-700">
          {text?.trim() || '(empty post)'}
        </p>
      )}
    </li>
  )
}

function Heading({ children }: { children: string }) {
  return (
    <div className="mb-1.5 text-xs font-medium tracking-wide text-slate-500 uppercase">
      {children}
    </div>
  )
}

interface Props {
  result: QuestionResult
  onTry: (question: string) => void
}

export function ResultDetail({ result, onTry }: Props) {
  const { question, transcript, judgement } = result
  const turnId = `eval-${question.id}`
  const [expanded, setExpanded] = useState<Set<number>>(new Set())
  const valid = new Set(transcript.sources.map((s) => s.n))
  const byN = new Map(transcript.sources.map((s) => [s.n, s]))
  const cited = citedNumbers(transcript.answer, valid).map((n) => byN.get(n)!)
  const labeled = new Set(question.relevant_doc_ids)

  const toggle = (n: number) =>
    setExpanded((prev) => {
      const next = new Set(prev)
      if (next.has(n)) next.delete(n)
      else next.add(n)
      return next
    })
  const showSource = (n: number) => {
    setExpanded((prev) => new Set(prev).add(n))
    document.getElementById(sourceElementId(turnId, n))?.scrollIntoView({ block: 'nearest' })
  }

  return (
    <div className="space-y-5 bg-slate-50 px-4 py-4 text-sm text-slate-800">
      {result.error && (
        <div className="rounded-lg bg-red-50 px-3 py-2 text-red-700">Error: {result.error}</div>
      )}

      {question.kind === 'fiqa' ? (
        <section>
          <Heading>Labeled correct posts</Heading>
          <ul className="space-y-1.5">
            {question.relevant_doc_ids.map((id) => (
              <LabeledPost
                key={id}
                docId={id}
                text={result.labeled_posts?.[id]}
                searches={transcript.searches}
              />
            ))}
          </ul>
        </section>
      ) : (
        <p className="text-slate-600">
          Unanswerable on purpose: the forum posts can't answer this, so the right answer is to say
          so.
        </p>
      )}

      {transcript.searches.length > 0 && (
        <section>
          <Heading>Retrieved</Heading>
          <div className="space-y-3">
            {transcript.searches.map((t, i) => {
              const stage = (name: StageName) =>
                t.stages.find((s) => s.name === name)?.results ?? []
              return (
                <div key={i} className="space-y-1.5">
                  <div className="text-xs text-slate-500">
                    Search: <span className="italic">“{t.query}”</span>
                  </div>
                  <StageRow label="BM25" results={stage('bm25')} labeled={labeled} />
                  <StageRow label="Dense" results={stage('dense')} labeled={labeled} />
                  <StageRow label="Top 8" results={stage('rerank')} labeled={labeled} />
                </div>
              )
            })}
          </div>
        </section>
      )}

      {transcript.answer && (
        <section>
          <Heading>Answer</Heading>
          <div className="rounded-xl border border-slate-200 bg-white px-4 py-3">
            <AnswerText text={transcript.answer} valid={valid} onCite={showSource} />
            <SourcesList turnId={turnId} sources={cited} expanded={expanded} onToggle={toggle} />
          </div>
        </section>
      )}

      {judgement && (
        <section>
          <Heading>Judge</Heading>
          <p className="mb-2 text-slate-600">
            Relevance {judgement.relevance}/5
            {judgement.declined && ' · declined to answer'} · {judgement.reason}
          </p>
          {judgement.claims.length > 0 && (
            <ul className="space-y-1.5">
              {judgement.claims.map((c, i) => (
                <li key={i} className="flex items-start gap-2">
                  <span
                    className={`shrink-0 rounded px-1.5 py-0.5 text-xs font-medium ${VERDICT_STYLE[c.verdict]}`}
                  >
                    {c.verdict.replaceAll('_', ' ')}
                  </span>
                  <span>
                    {c.text}
                    {c.cites.length > 0 && (
                      <span className="ml-1 text-xs text-slate-400">
                        {c.cites.map((n) => `[${n}]`).join('')}
                      </span>
                    )}
                  </span>
                </li>
              ))}
            </ul>
          )}
        </section>
      )}

      <button
        type="button"
        onClick={() => onTry(question.text)}
        className="rounded-lg bg-blue-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-blue-700"
      >
        Try this in chat →
      </button>
    </div>
  )
}
