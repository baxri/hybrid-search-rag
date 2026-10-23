import { EXAMPLE_QUESTIONS } from '../lib/examples'

export function EmptyState({ onPick }: { onPick: (question: string) => void }) {
  return (
    <div className="mx-auto flex max-w-xl flex-col items-center py-16 text-center">
      <h2 className="text-xl font-semibold text-slate-800">Ask the personal-finance forum</h2>
      <p className="mt-2 text-slate-500">
        Answers come from 57,638 FiQA forum posts, found with hybrid search and cited inline.
      </p>
      <div className="mt-8 grid w-full gap-2 sm:grid-cols-2">
        {EXAMPLE_QUESTIONS.slice(0, 4).map((q) => (
          <button
            key={q}
            type="button"
            onClick={() => onPick(q)}
            className="rounded-xl border border-slate-200 bg-white px-4 py-3 text-left text-sm text-slate-700 shadow-sm transition hover:border-blue-300 hover:bg-blue-50"
          >
            {q}
          </button>
        ))}
      </div>
    </div>
  )
}
