import type { SearchTrace, StageName } from './types'

export interface RankJump {
  from: 'bm25' | 'dense'
  delta: number
}

/** Climbs smaller than this are noise, not a story worth a badge. */
const MIN_JUMP = 5

function rankIn(trace: SearchTrace, stage: StageName, docId: string): number | undefined {
  return trace.stages.find((s) => s.name === stage)?.results.find((r) => r.doc_id === docId)?.rank
}

/** How far the reranker lifted a doc compared with the retriever that found it ("▲13 vs BM25"). */
export function rankJump(trace: SearchTrace, docId: string): RankJump | null {
  const reranked = rankIn(trace, 'rerank', docId)
  if (reranked === undefined) return null
  let best: RankJump | null = null
  for (const from of ['bm25', 'dense'] as const) {
    const rank = rankIn(trace, from, docId)
    if (rank === undefined) continue
    const delta = rank - reranked
    if (!best || delta > best.delta) best = { from, delta }
  }
  return best && best.delta >= MIN_JUMP ? best : null
}
