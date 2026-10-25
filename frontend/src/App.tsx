import { useEffect, useState } from 'react'
import { ChatPane } from './components/ChatPane'
import { ArchitecturePage } from './components/ArchitecturePage'
import { EvalPage } from './components/eval/EvalPage'
import { GitHubLink } from './components/GitHubLink'
import { Landing } from './components/Landing'
import { PipelinePanel } from './components/PipelinePanel'
import { useChat } from './hooks/useChat'
import { DOCS_URL } from './lib/links'

export default function App() {
  const { turns, send, retry, selectedTurnId, select } = useChat()
  const [drawerOpen, setDrawerOpen] = useState(false)
  const tab = useHashTab()
  const selected = turns.find((t) => t.id === selectedTurnId)
  const turn = selected?.role === 'assistant' ? selected : undefined
  const ask = (question: string) => {
    window.location.hash = 'chat'
    send(question)
  }

  if (tab === 'home') {
    return (
      <Landing
        onTry={() => {
          window.location.hash = 'chat'
        }}
        onAsk={ask}
      />
    )
  }

  return (
    <div className="flex h-dvh flex-col bg-slate-50">
      <header className="flex flex-wrap items-center justify-between gap-x-3 gap-y-2 border-b border-slate-200 bg-white px-4 py-3">
        <a href="#" className="min-w-0">
          <h1 className="font-semibold text-slate-900">Hybrid Search Chat</h1>
          <p className="truncate text-xs text-slate-500">
            BM25 + embeddings → RRF → Cohere rerank → GPT answer with citations
          </p>
        </a>
        <div className="flex shrink-0 items-center gap-2">
          <nav className="flex rounded-lg bg-slate-100 p-0.5 text-sm">
            {TABS.map(({ id, label }) => (
              <a
                key={id}
                href={`#${id}`}
                className={`rounded-md px-3 py-1 ${
                  tab === id
                    ? 'bg-white font-medium text-slate-900 shadow-sm'
                    : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                {label}
              </a>
            ))}
          </nav>
          {tab === 'chat' && (
            <button
              type="button"
              onClick={() => setDrawerOpen(true)}
              className="rounded-lg border border-slate-200 px-3 py-1.5 text-sm text-slate-700 hover:bg-slate-50 lg:hidden"
            >
              Pipeline
            </button>
          )}
          <a
            href={DOCS_URL}
            target="_blank"
            rel="noreferrer"
            className="rounded-md px-2 py-1 text-sm text-slate-600 hover:text-slate-900"
          >
            Docs
          </a>
          <GitHubLink className="text-slate-500 hover:bg-slate-100 hover:text-slate-900" />
        </div>
      </header>

      {tab === 'architecture' ? (
        <div className="min-h-0 flex-1 overflow-y-auto">
          <ArchitecturePage />
        </div>
      ) : tab === 'eval' ? (
        <div className="min-h-0 flex-1 overflow-y-auto">
          <EvalPage onTry={ask} />
        </div>
      ) : (
        <div className="flex min-h-0 flex-1">
          {drawerOpen && (
            <div
              className="fixed inset-0 z-10 bg-slate-900/30 lg:hidden"
              onClick={() => setDrawerOpen(false)}
            />
          )}
          <aside
            className={`fixed inset-y-0 left-0 z-20 w-[min(90vw,28rem)] overflow-y-auto border-r border-slate-200 bg-slate-100 shadow-xl transition-transform lg:static lg:w-2/5 lg:translate-x-0 lg:shadow-none ${
              drawerOpen ? 'translate-x-0' : '-translate-x-full'
            }`}
          >
            <button
              type="button"
              onClick={() => setDrawerOpen(false)}
              className="absolute top-3 right-3 rounded-md px-2 text-slate-500 hover:bg-slate-200 lg:hidden"
              aria-label="Close pipeline panel"
            >
              ✕
            </button>
            <PipelinePanel turn={turn} />
          </aside>
          <main className="min-w-0 flex-1">
            <ChatPane
              turns={turns}
              selectedTurnId={selectedTurnId}
              onSend={send}
              onRetry={retry}
              onSelect={select}
            />
          </main>
        </div>
      )}
    </div>
  )
}

type Tab = 'home' | 'chat' | 'eval' | 'architecture'

const TABS: { id: Exclude<Tab, 'home'>; label: string }[] = [
  { id: 'chat', label: 'Chat' },
  { id: 'eval', label: 'Evaluation' },
  { id: 'architecture', label: 'Architecture' },
]

const tabFromHash = (): Tab => TABS.find((t) => window.location.hash === `#${t.id}`)?.id ?? 'home'

/** The landing page is the home page; #chat, #eval and #architecture can be linked directly. */
function useHashTab(): Tab {
  const [tab, setTab] = useState(tabFromHash)
  useEffect(() => {
    const onChange = () => setTab(tabFromHash())
    window.addEventListener('hashchange', onChange)
    return () => window.removeEventListener('hashchange', onChange)
  }, [])
  return tab
}
