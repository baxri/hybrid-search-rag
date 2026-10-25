// Mirrors backend/evals/report.py. The Evaluation tab reads /eval/results.json.
import type { SearchTrace, Source, StageName } from './types'

export interface EvalQuestion {
  id: string
  text: string
  kind: 'fiqa' | 'unanswerable'
  relevant_doc_ids: string[]
}

export interface Transcript {
  searches: SearchTrace[]
  sources: Source[]
  answer: string
  error: string | null
}

export interface RetrievalScore {
  hit: boolean
  first_rank: number | null
  mrr: number
  stage_recall: Record<StageName, number>
}

export type Verdict = 'supported' | 'unsupported' | 'uncited' | 'not_a_claim'

export interface Claim {
  text: string
  cites: number[]
  verdict: Verdict
}

export interface Judgement {
  relevance: number
  declined: boolean
  claims: Claim[]
  reason: string
}

export interface QuestionResult {
  question: EvalQuestion
  transcript: Transcript
  retrieval: RetrievalScore | null
  judgement: Judgement | null
  error: string | null
  hallucinated: boolean
  failed: boolean
}

export interface Summary {
  fiqa_total: number
  unanswerable_total: number
  errors: number
  failures: number
  hit_at_8: number
  mrr: number
  stage_recall: Record<StageName, number>
  relevance_avg: number | null
  grounded_rate: number | null
  hallucination_rate: number | null
  declined_correctly: number
}

export interface RunMeta {
  date: string
  commit: string
  answer_model: string
  judge_model: string
  seed: number
  question_count: number
}

export interface EvalReport {
  meta: RunMeta
  summary: Summary
  results: QuestionResult[]
}

export const pct = (x: number | null) => (x === null ? '—' : `${Math.round(x * 100)}%`)
