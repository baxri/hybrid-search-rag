// Mirrors backend/app/schemas.py. /api/chat answers with NDJSON: one ChatEvent per line.

export type StageName = 'bm25' | 'dense' | 'rrf' | 'rerank'

export interface RankedDoc {
  doc_id: string
  rank: number // 1-based
  score: number
}

export interface Stage {
  name: StageName
  ms: number
  results: RankedDoc[]
}

/** What each retrieval stage returned, in order: bm25, dense, rrf, rerank. */
export interface SearchTrace {
  query: string
  total_ms: number
  stages: Stage[]
}

/** A post the answer may cite as [n]. */
export interface Source {
  n: number
  doc_id: string
  text: string
}

export interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
}

export type ChatEvent =
  | { type: 'search_started'; search_id: number; query: string }
  | { type: 'search_result'; search_id: number; trace: SearchTrace; sources: Source[] }
  | { type: 'text_delta'; delta: string }
  | { type: 'error'; message: string }
  | { type: 'done' }
