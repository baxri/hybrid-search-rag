import { useEvalReport } from '../hooks/useEvalReport'
import { EXAMPLE_QUESTIONS } from '../lib/examples'
import { pct } from '../lib/evalTypes'
import { DOCS_URL } from '../lib/links'
import { GitHubLink } from './GitHubLink'

interface Props {
  onTry: () => void
  onAsk: (question: string) => void
}

const PIPELINE = [
  {
    step: '01',
    title: 'Search two ways',
    body: 'BM25 catches exact terms like "ESPP" or "401(k)". Embeddings catch meaning: "rainy-day fund" finds "short-term savings". Both run at once, in memory.',
  },
  {
    step: '02',
    title: 'Fuse the rankings',
    body: 'Reciprocal Rank Fusion merges both top-50 lists by rank, not score, so a post both searches agree on rises to the top.',
  },
  {
    step: '03',
    title: 'Rerank the best',
    body: 'A Cohere cross-encoder reads all 50 candidates against the question and keeps the 8 that actually answer it.',
  },
  {
    step: '04',
    title: 'Answer with citations',
    body: 'An agent writes the answer from those 8 posts only, citing each claim as [n], and says so when the posts don’t cover it.',
  },
]

const WHY = [
  {
    title: 'It fits in memory',
    body: 'Both indexes load into the app’s memory at startup: the BM25 keyword index and a 354 MB NumPy matrix of 57,638 × 1,536 embeddings. Dense search is one matrix multiply: exact, no approximate index, no extra service to run.',
  },
  {
    title: 'Keywords still matter',
    body: 'Pure vector search blurs tickers, acronyms and account types. Running BM25 alongside embeddings recovers what similarity alone misses.',
  },
  {
    title: 'Precision comes from reranking',
    body: 'Fast first-stage search only has to get the right post into the top 50. The reranker turns that into a precise top 8.',
  },
]

const STACK = [
  'FastAPI',
  'Pydantic AI',
  'OpenAI',
  'bm25s',
  'NumPy',
  'Cohere Rerank',
  'React',
  'Tailwind',
]

function TryButton({ onClick, className = '' }: { onClick: () => void; className?: string }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`rounded-xl bg-blue-600 px-6 py-3 font-semibold text-white shadow-lg shadow-blue-600/25 transition hover:bg-blue-500 ${className}`}
    >
      Try it →
    </button>
  )
}

function Badge({ n }: { n: number }) {
  return (
    <span className="mx-0.5 inline-flex h-4 min-w-4 -translate-y-1 items-center justify-center rounded bg-blue-100 px-1 text-[10px] font-semibold text-blue-700">
      {n}
    </span>
  )
}

/** A real answer from the evaluation run, shown as a preview of the chat. */
function ChatPreview() {
  return (
    <div className="rounded-2xl border border-white/10 bg-white p-5 text-left text-sm shadow-2xl shadow-blue-950/40">
      <div className="ml-auto w-fit rounded-2xl rounded-br-sm bg-blue-600 px-4 py-2 text-white">
        How do index funds actually work?
      </div>
      <div className="mt-4 rounded-2xl rounded-bl-sm border border-slate-200 px-4 py-3 leading-relaxed text-slate-800">
        Index funds don’t “buy the index” itself; they hold the stocks in the index, usually in the
        same proportions the index uses, so the fund is meant to replicate the index’s performance
        <Badge n={2} />
        <Badge n={3} />
        <Badge n={6} />. The manager still does some buying and selling to keep the holdings aligned
        when dividends are paid or when the index changes
        <Badge n={1} />
        <Badge n={8} />.
        <div className="mt-3 space-y-1 border-t border-slate-100 pt-2 text-xs text-slate-500">
          <div>
            <span className="font-semibold text-blue-700">[2]</span> In general, the goal of an
            S&amp;P 500 index fund is to replicate…
          </div>
          <div>
            <span className="font-semibold text-blue-700">[3]</span> An index fund is just copying
            the definition of an index…
          </div>
        </div>
      </div>
      <div className="mt-3 flex flex-wrap gap-1.5 font-mono text-[10px] text-slate-500">
        <span className="rounded bg-slate-100 px-1.5 py-0.5">BM25 50</span>
        <span className="rounded bg-slate-100 px-1.5 py-0.5">Dense 50</span>
        <span className="rounded bg-slate-100 px-1.5 py-0.5">RRF 50</span>
        <span className="rounded bg-blue-50 px-1.5 py-0.5 text-blue-700">Rerank 8</span>
      </div>
    </div>
  )
}

function Stats() {
  const load = useEvalReport()
  const s = load.status === 'ok' ? load.report.summary : undefined
  const stats = [
    { value: '57,638', label: 'forum posts searched' },
    ...(s
      ? [
          { value: pct(s.hit_at_8), label: 'correct post in the top 8' },
          { value: pct(s.grounded_rate), label: 'of claims backed by a cited post' },
          {
            value: s.relevance_avg === null ? '—' : `${s.relevance_avg.toFixed(1)}/5`,
            label: 'answer relevance',
          },
        ]
      : [{ value: '0', label: 'vector databases' }]),
  ]
  return (
    <section className="border-y border-slate-200 bg-white">
      <div className="mx-auto max-w-6xl px-6 py-10">
        <div className="grid grid-cols-2 gap-6 md:grid-cols-4">
          {stats.map((st) => (
            <div key={st.label}>
              <div className="text-3xl font-bold tracking-tight text-slate-900">{st.value}</div>
              <div className="mt-1 text-sm text-slate-500">{st.label}</div>
            </div>
          ))}
        </div>
        {s && (
          <p className="mt-6 text-sm text-slate-500">
            Measured on {s.fiqa_total} FiQA questions + {s.unanswerable_total} unanswerable ones,
            graded against human labels and an LLM judge.{' '}
            <a href="#eval" className="font-medium text-blue-600 hover:underline">
              See every question and verdict →
            </a>
          </p>
        )}
      </div>
    </section>
  )
}

export function Landing({ onTry, onAsk }: Props) {
  return (
    <div className="min-h-dvh bg-slate-50 text-slate-900">
      <div className="bg-gradient-to-b from-slate-950 via-slate-900 to-blue-950 text-white">
        <nav className="mx-auto flex max-w-6xl items-center justify-between gap-4 px-6 py-5">
          <a href="#" className="flex items-center gap-2 font-semibold">
            <span className="grid h-7 w-7 place-items-center rounded-lg bg-blue-600 text-sm">
              ◆
            </span>
            Hybrid Search
          </a>
          <div className="flex items-center gap-1 text-sm sm:gap-4">
            <a
              href="#architecture"
              className="hidden px-2 text-slate-300 hover:text-white sm:inline"
            >
              Architecture
            </a>
            <a href="#eval" className="hidden px-2 text-slate-300 hover:text-white sm:inline">
              Evaluation
            </a>
            <a
              href={DOCS_URL}
              target="_blank"
              rel="noreferrer"
              className="hidden px-2 text-slate-300 hover:text-white sm:inline"
            >
              Docs
            </a>
            <button
              type="button"
              onClick={onTry}
              className="rounded-lg bg-white/10 px-4 py-1.5 font-medium ring-1 ring-white/20 hover:bg-white/20"
            >
              Try it
            </button>
            <GitHubLink className="text-slate-300 hover:bg-white/10 hover:text-white" />
          </div>
        </nav>

        <header className="mx-auto grid max-w-6xl items-center gap-12 px-6 pt-10 pb-20 lg:grid-cols-2 lg:pt-16 lg:pb-28">
          <div>
            <div className="mb-5 inline-flex rounded-full bg-blue-500/10 px-3 py-1 text-xs font-medium text-blue-300 ring-1 ring-blue-400/30">
              In-memory hybrid search · reranking · cited answers
            </div>
            <h1 className="text-4xl font-bold tracking-tight sm:text-5xl lg:text-6xl">
              RAG without a <span className="text-blue-400">vector database</span>
            </h1>
            <p className="mt-6 max-w-xl text-lg leading-relaxed text-slate-300">
              Ask a personal-finance question. BM25 keyword search and dense embedding search run
              side by side, in memory, over 57,638 forum posts. A reranker keeps the best 8, and
              every claim in the answer cites its source. The embeddings are a plain NumPy matrix
              and BM25 is an in-process index: no vector DB, no search service.
            </p>
            <div className="mt-8 flex flex-wrap items-center gap-4">
              <TryButton onClick={onTry} />
              <a
                href="#architecture"
                className="px-2 py-3 font-medium text-slate-300 hover:text-white"
              >
                See how it works
              </a>
            </div>
          </div>
          <ChatPreview />
        </header>
      </div>

      <Stats />

      <section className="mx-auto max-w-6xl px-6 py-20">
        <h2 className="text-3xl font-bold tracking-tight">How it works</h2>
        <p className="mt-2 max-w-2xl text-slate-600">
          Four stages, every one visible in the chat’s pipeline panel as it runs.
        </p>
        <div className="mt-10 grid gap-6 sm:grid-cols-2 lg:grid-cols-4">
          {PIPELINE.map((p) => (
            <div key={p.step} className="rounded-2xl border border-slate-200 bg-white p-6">
              <div className="font-mono text-sm font-semibold text-blue-600">{p.step}</div>
              <h3 className="mt-2 text-lg font-semibold">{p.title}</h3>
              <p className="mt-2 text-sm leading-relaxed text-slate-600">{p.body}</p>
            </div>
          ))}
        </div>
        <a
          href="#architecture"
          className="mt-6 inline-block text-sm font-medium text-blue-600 hover:underline"
        >
          Explore the full architecture →
        </a>
      </section>

      <section className="bg-white">
        <div className="mx-auto max-w-6xl px-6 py-20">
          <h2 className="text-3xl font-bold tracking-tight">Why no vector database?</h2>
          <p className="mt-2 max-w-2xl text-slate-600">
            Vector databases earn their keep at millions of documents. At this size, they’re a
            service to run, pay for and keep in sync, for no gain in quality.
          </p>
          <div className="mt-10 grid gap-6 md:grid-cols-3">
            {WHY.map((w) => (
              <div key={w.title} className="rounded-2xl bg-slate-50 p-6 ring-1 ring-slate-200">
                <h3 className="font-semibold">{w.title}</h3>
                <p className="mt-2 text-sm leading-relaxed text-slate-600">{w.body}</p>
              </div>
            ))}
          </div>
          <div className="mt-10 flex flex-wrap gap-2">
            {STACK.map((t) => (
              <span
                key={t}
                className="rounded-full border border-slate-200 px-3 py-1 text-sm text-slate-600"
              >
                {t}
              </span>
            ))}
          </div>
        </div>
      </section>

      <section className="mx-auto max-w-6xl px-6 py-20 text-center">
        <h2 className="text-3xl font-bold tracking-tight">Ask it something</h2>
        <p className="mt-2 text-slate-600">Pick a question, or write your own.</p>
        <div className="mx-auto mt-8 flex max-w-3xl flex-wrap justify-center gap-2">
          {EXAMPLE_QUESTIONS.slice(0, 4).map((q) => (
            <button
              key={q}
              type="button"
              onClick={() => onAsk(q)}
              className="rounded-full border border-slate-300 bg-white px-4 py-2 text-sm text-slate-700 hover:border-blue-400 hover:text-blue-700"
            >
              {q}
            </button>
          ))}
        </div>
        <TryButton onClick={onTry} className="mt-8" />
      </section>

      <footer className="border-t border-slate-200 py-8 text-center text-xs text-slate-500">
        Answers come from the FiQA forum dataset (BEIR). Forum opinions, not financial advice.
      </footer>
    </div>
  )
}
