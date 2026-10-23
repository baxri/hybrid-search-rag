import { useState } from 'react'
import { ChatPane } from './components/ChatPane'
import { PipelinePanel } from './components/PipelinePanel'
import { useChat } from './hooks/useChat'

export default function App() {
  const { turns, send, retry, selectedTurnId, select } = useChat()
  const [drawerOpen, setDrawerOpen] = useState(false)
  const selected = turns.find((t) => t.id === selectedTurnId)
  const turn = selected?.role === 'assistant' ? selected : undefined

  return (
    <div className="flex h-dvh flex-col bg-slate-50">
      <header className="flex items-center justify-between gap-3 border-b border-slate-200 bg-white px-4 py-3">
        <div className="min-w-0">
          <h1 className="font-semibold text-slate-900">Hybrid Search Chat</h1>
          <p className="truncate text-xs text-slate-500">
            BM25 + embeddings → RRF → Cohere rerank → GPT answer with citations
          </p>
        </div>
        <button
          type="button"
          onClick={() => setDrawerOpen(true)}
          className="shrink-0 rounded-lg border border-slate-200 px-3 py-1.5 text-sm text-slate-700 hover:bg-slate-50 lg:hidden"
        >
          Pipeline
        </button>
      </header>

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
    </div>
  )
}
