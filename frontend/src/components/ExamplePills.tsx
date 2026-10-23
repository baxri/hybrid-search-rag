import { EXAMPLE_QUESTIONS } from '../lib/examples'

interface Props {
  asked: Set<string>
  disabled: boolean
  onPick: (question: string) => void
}

/** One-click example questions above the composer; ones already asked drop out. */
export function ExamplePills({ asked, disabled, onPick }: Props) {
  const remaining = EXAMPLE_QUESTIONS.filter((q) => !asked.has(q))
  if (!remaining.length) return null
  return (
    // Wraps on wider screens; one swipeable row on phones.
    <div className="mb-2 flex items-center gap-1.5 overflow-x-auto [scrollbar-width:none] sm:flex-wrap">
      <span className="shrink-0 text-xs text-slate-400">Try asking:</span>
      {remaining.map((q) => (
        <button
          key={q}
          type="button"
          disabled={disabled}
          onClick={() => onPick(q)}
          className="shrink-0 rounded-full border border-slate-200 bg-white px-3 py-1 text-xs text-slate-600 shadow-sm transition hover:border-blue-300 hover:bg-blue-50 hover:text-blue-700 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {q}
        </button>
      ))}
    </div>
  )
}
