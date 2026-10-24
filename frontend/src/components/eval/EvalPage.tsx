import { useEvalReport } from '../../hooks/useEvalReport'
import { ResultsTable } from './ResultsTable'
import { ScoreTiles } from './ScoreTiles'

export function EvalPage({ onTry }: { onTry: (question: string) => void }) {
  const load = useEvalReport()

  if (load.status !== 'ok') {
    return (
      <div className="p-8 text-center text-slate-500">
        {load.status === 'loading' ? 'Loading…' : 'No evaluation published yet.'}
      </div>
    )
  }

  const { meta, summary, results } = load.report
  return (
    <div className="mx-auto max-w-5xl space-y-6 px-4 py-6">
      <div>
        <h2 className="text-lg font-semibold text-slate-900">Evaluation</h2>
        <p className="mt-0.5 text-xs text-slate-500">
          {meta.date.slice(0, 10)} · commit {meta.commit} · answers {meta.answer_model} · judge{' '}
          {meta.judge_model} · {summary.fiqa_total} FiQA + {summary.unanswerable_total} unanswerable
          · {summary.failures} failed
          {summary.errors > 0 && ` · ${summary.errors} error${summary.errors === 1 ? '' : 's'}`}
        </p>
      </div>
      <ScoreTiles summary={summary} />
      <ResultsTable results={results} onTry={onTry} />
    </div>
  )
}
