import { Fragment, useState } from 'react'
import type { QuestionResult } from '../../lib/evalTypes'
import { ResultDetail } from './ResultDetail'

function Found({ result }: { result: QuestionResult }) {
  if (!result.retrieval) return <span className="text-slate-400">—</span>
  return result.retrieval.hit ? (
    <span className="text-emerald-700">✅ #{result.retrieval.first_rank}</span>
  ) : (
    <span>❌</span>
  )
}

function Claims({ result }: { result: QuestionResult }) {
  const claims = result.judgement?.claims
  if (!claims) return <span className="text-slate-400">—</span>
  if (!claims.length)
    return <span className="text-slate-500">{result.judgement?.declined ? 'declined' : '0'}</span>
  const ok = claims.filter((c) => c.verdict === 'supported').length
  return (
    <span className={ok < claims.length ? 'font-medium text-red-600' : 'text-slate-700'}>
      {ok}/{claims.length}
    </span>
  )
}

interface Props {
  results: QuestionResult[]
  onTry: (question: string) => void
}

export function ResultsTable({ results, onTry }: Props) {
  const [failuresOnly, setFailuresOnly] = useState(false)
  const [open, setOpen] = useState<string | undefined>()
  const shown = failuresOnly ? results.filter((r) => r.failed) : results

  return (
    <div>
      <div className="mb-2 flex items-center justify-between">
        <h2 className="font-semibold text-slate-900">
          Questions <span className="font-normal text-slate-500">({shown.length})</span>
        </h2>
        <label className="flex items-center gap-2 text-sm text-slate-700">
          <input
            type="checkbox"
            checked={failuresOnly}
            onChange={(e) => setFailuresOnly(e.target.checked)}
          />
          Failures only
        </label>
      </div>
      <div className="overflow-hidden rounded-xl border border-slate-200 bg-white">
        <table className="w-full text-sm">
          <thead className="bg-slate-50 text-left text-xs text-slate-500">
            <tr>
              <th className="px-3 py-2 font-medium">Question</th>
              <th className="px-2 py-2 font-medium whitespace-nowrap">Found?</th>
              <th className="hidden px-2 py-2 font-medium sm:table-cell">Relevance</th>
              <th className="px-2 py-2 font-medium">Claims</th>
              <th className="w-6" />
            </tr>
          </thead>
          <tbody>
            {shown.map((r) => {
              const isOpen = open === r.question.id
              return (
                <Fragment key={r.question.id}>
                  <tr
                    onClick={() => setOpen(isOpen ? undefined : r.question.id)}
                    className={`cursor-pointer border-t border-slate-100 hover:bg-slate-50 ${
                      r.failed ? 'bg-red-50/40' : ''
                    }`}
                  >
                    <td className="px-3 py-2 text-slate-800">
                      {r.question.kind === 'unanswerable' && (
                        <span className="mr-1.5 rounded bg-slate-100 px-1 text-[10px] text-slate-500 uppercase">
                          unanswerable
                        </span>
                      )}
                      {r.question.text}
                      {r.error && <span className="ml-1.5 text-xs text-red-600">error</span>}
                    </td>
                    <td className="px-2 py-2 whitespace-nowrap">
                      <Found result={r} />
                    </td>
                    <td className="hidden px-2 py-2 sm:table-cell">
                      {r.judgement ? (
                        <span className={r.judgement.relevance <= 2 ? 'text-red-600' : ''}>
                          {r.judgement.relevance}/5
                        </span>
                      ) : (
                        <span className="text-slate-400">—</span>
                      )}
                    </td>
                    <td className="px-2 py-2">
                      <Claims result={r} />
                    </td>
                    <td className="pr-3 text-slate-400">{isOpen ? '▾' : '▸'}</td>
                  </tr>
                  {isOpen && (
                    <tr>
                      <td colSpan={5} className="border-t border-slate-100 p-0">
                        <ResultDetail result={r} onTry={onTry} />
                      </td>
                    </tr>
                  )}
                </Fragment>
              )
            })}
            {!shown.length && (
              <tr>
                <td colSpan={5} className="px-3 py-6 text-center text-slate-500">
                  No failures.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  )
}
