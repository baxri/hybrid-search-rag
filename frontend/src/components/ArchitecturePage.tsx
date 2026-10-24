import { Fragment, useState } from 'react'

// The live request path through the backend, top to bottom. Values mirror backend/app.

interface Node {
  id: string
  title: string
  tech: string
  external?: boolean // calls a paid outside API
  what: string
  why: string
  facts: string[]
}

type Row = Node | Node[] // an array is steps that run at the same time

const ROWS: Row[] = [
  {
    id: 'api',
    title: 'POST /api/chat',
    tech: 'FastAPI · Python 3.12',
    what: 'Receives the whole conversation as JSON and streams the reply back over the same request.',
    why: 'Sending the full history each turn keeps the server stateless: any instance can answer any message.',
    facts: [
      'Request: { messages: [{ role, content }, …] }',
      'Earlier answers have their [n] markers stripped before the model sees them, since numbers restart every turn',
    ],
  },
  {
    id: 'agent',
    title: 'Agent',
    tech: 'Pydantic AI · OpenAI gpt-5.4-mini',
    external: true,
    what: 'Decides whether the message needs a search, and rewrites follow-ups ("what about for kids?") into a standalone search query.',
    why: 'A greeting needs no search; a vague follow-up searched as-is would find nothing. The agent may search more than once if the first results are weak.',
    facts: [
      'One tool: search_finance_posts(query)',
      'Instructed to answer only from returned posts and cite them as [n]',
      'Says so plainly when the posts do not cover the question',
    ],
  },
  [
    {
      id: 'bm25',
      title: 'BM25',
      tech: 'bm25s · keyword match',
      what: 'Classic keyword search: scores posts by how often the query’s words appear, weighted by how rare each word is.',
      why: 'Exact terms matter in finance: "ESPP", "UTMA", "401(k)", ticker symbols. Embeddings can blur these.',
      facts: [
        'Searches all 57,638 FiQA posts',
        'Returns the top 50',
        'Runs in about 10 ms, in-process',
      ],
    },
    {
      id: 'dense',
      title: 'Dense',
      tech: 'OpenAI text-embedding-3-small · cosine',
      external: true,
      what: 'Turns the query into a vector and compares it with a precomputed vector for every post (one matrix multiply).',
      why: 'Finds posts that mean the same thing in different words: "park my rainy-day fund" ≈ "short-term savings".',
      facts: [
        'Post vectors are computed once, offline, and loaded at startup',
        'Returns the top 50',
        'Most of its ~1–2 s is the embedding API call',
      ],
    },
  ],
  {
    id: 'rrf',
    title: 'RRF fusion',
    tech: 'Reciprocal Rank Fusion · k = 60',
    what: 'Merges the two top-50 lists. Each post scores 1 / (60 + rank) in every list it appears in; the scores are added.',
    why: 'BM25 and cosine scores are on different scales, so they can’t be added directly. Ranks can. A post both searches agree on rises to the top.',
    facts: ['Keeps the top 50', 'Blank posts are dropped here', 'Takes under 1 ms'],
  },
  {
    id: 'rerank',
    title: 'Rerank',
    tech: 'Cohere rerank-v4.0-fast',
    external: true,
    what: 'A cross-encoder reads the query and each of the 50 posts together and scores how well the post answers it.',
    why: 'Much more accurate than the first two searches, but too slow to run on 57,638 posts. Running it on 50 candidates gets the accuracy at a fraction of the cost.',
    facts: ['50 in → best 8 out', 'Only these 8 posts reach the model'],
  },
  {
    id: 'cite',
    title: 'Numbered sources → model',
    tech: 'SourceRegistry · format_for_model',
    what: 'The 8 posts get numbers [1]–[8] and go back to the agent as plain text, one numbered block per post: "[1] post text…", "[2] post text…". The model writes its answer citing those numbers.',
    why: 'The model only ever sees numbers and text, never IDs or scores, so citations stay simple and checkable. A second search continues at [9]; a post seen twice keeps its first number.',
    facts: [
      'Each post is cut to 2,000 characters for the model',
      'Search details travel separately as tool metadata, which the model never sees',
    ],
  },
  {
    id: 'stream',
    title: 'NDJSON stream',
    tech: 'one JSON event per line',
    what: 'Pydantic AI’s internal events are mapped onto five small events the browser understands.',
    why: 'The browser can show "searching…", the pipeline trace and the answer as they happen, instead of waiting for the whole reply.',
    facts: [
      'search_started: the query, as soon as the agent searches',
      'search_result: every stage’s ranking + the newly numbered posts',
      'text_delta: the answer, token by token',
      'error / done: how the turn ended',
    ],
  },
]

function Box({ node, open, onToggle }: { node: Node; open: boolean; onToggle: () => void }) {
  return (
    <button
      type="button"
      onClick={onToggle}
      aria-expanded={open}
      className={`w-full rounded-xl border-2 px-4 py-3 text-left transition-colors ${
        node.external ? 'border-dashed' : ''
      } ${
        open
          ? 'border-blue-500 bg-blue-50'
          : `bg-white hover:border-blue-300 ${node.external ? 'border-violet-300' : 'border-slate-200'}`
      }`}
    >
      <div className="flex items-center justify-between gap-2">
        <span className="font-semibold text-slate-900">{node.title}</span>
        <span className="text-slate-400">{open ? '▾' : '▸'}</span>
      </div>
      <div className="mt-0.5 text-xs text-slate-500">{node.tech}</div>
    </button>
  )
}

function Detail({ node }: { node: Node }) {
  return (
    <div className="mt-2 space-y-2 rounded-xl border border-blue-200 bg-white px-4 py-3 text-sm text-slate-700">
      <p>{node.what}</p>
      <p>
        <span className="font-medium text-slate-900">Why: </span>
        {node.why}
      </p>
      <ul className="ml-5 list-disc space-y-0.5 text-slate-600">
        {node.facts.map((f) => (
          <li key={f}>{f}</li>
        ))}
      </ul>
    </div>
  )
}

function Arrow({ label }: { label?: string }) {
  return (
    <div className="flex flex-col items-center py-1 text-slate-400">
      <span className="h-4 w-px bg-slate-300" />
      {label && <span className="my-0.5 font-mono text-[11px] text-slate-500">{label}</span>}
      <span className="leading-none">▼</span>
    </div>
  )
}

const ARROW_LABELS: Record<string, string> = {
  agent: 'search_finance_posts(query)',
  rrf: 'top 50 + top 50',
  rerank: 'top 50',
  cite: 'top 8',
}

export function ArchitecturePage() {
  const [open, setOpen] = useState<string | undefined>()
  const toggle = (id: string) => setOpen((cur) => (cur === id ? undefined : id))

  return (
    <div className="mx-auto max-w-3xl px-4 py-6">
      <h2 className="text-lg font-semibold text-slate-900">How a question is answered</h2>
      <p className="mt-0.5 text-sm text-slate-600">
        The backend path for one chat message. Click any box for what it does and why it’s there.
      </p>
      <p className="mt-1 flex items-center gap-1.5 text-xs text-slate-500">
        <span className="inline-block h-3 w-5 rounded border-2 border-dashed border-violet-300" />
        calls an outside API (OpenAI or Cohere)
      </p>

      <div className="mt-5">
        {ROWS.map((row, i) => {
          const nodes = Array.isArray(row) ? row : [row]
          const opened = nodes.find((n) => n.id === open)
          return (
            <Fragment key={nodes[0].id}>
              {i > 0 && <Arrow label={ARROW_LABELS[nodes[0].id]} />}
              {nodes.length > 1 && (
                <div className="mb-1 text-center text-xs text-slate-500">run at the same time</div>
              )}
              <div className={nodes.length > 1 ? 'grid grid-cols-2 gap-3' : ''}>
                {nodes.map((n) => (
                  <Box key={n.id} node={n} open={open === n.id} onToggle={() => toggle(n.id)} />
                ))}
              </div>
              {opened && <Detail node={opened} />}
            </Fragment>
          )
        })}
      </div>
    </div>
  )
}
