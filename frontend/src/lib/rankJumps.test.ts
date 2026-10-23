import { expect, test } from 'vitest'
import { rankJump } from './rankJumps'
import type { SearchTrace, StageName } from './types'

/** Build a trace where `d` sits at the given rank in each named stage (absent if omitted). */
function traceWith(ranks: Partial<Record<StageName, number>>): SearchTrace {
  const names: StageName[] = ['bm25', 'dense', 'rrf', 'rerank']
  return {
    query: 'q',
    total_ms: 1,
    stages: names.map((name) => ({
      name,
      ms: 1,
      results: ranks[name] ? [{ doc_id: 'd', rank: ranks[name], score: 0.5 }] : [],
    })),
  }
}

test('reports the retriever the doc climbed most from', () => {
  expect(rankJump(traceWith({ bm25: 14, dense: 1, rerank: 1 }), 'd')).toEqual({
    from: 'bm25',
    delta: 13,
  })
})

test('small climbs are not worth a badge', () => {
  expect(rankJump(traceWith({ bm25: 2, dense: 3, rerank: 1 }), 'd')).toBeNull()
})

test('a doc only one retriever found is measured against that one', () => {
  expect(rankJump(traceWith({ dense: 9, rerank: 1 }), 'd')).toEqual({ from: 'dense', delta: 8 })
})

test('a doc missing from the rerank stage has no jump', () => {
  expect(rankJump(traceWith({ bm25: 20 }), 'd')).toBeNull()
})
