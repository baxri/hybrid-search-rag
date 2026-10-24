import { pct, type Summary } from '../../lib/evalTypes'

function Tile({ label, value, detail }: { label: string; value: string; detail: string }) {
  return (
    <div className="rounded-xl border border-slate-200 bg-white p-4">
      <div className="text-xs font-medium tracking-wide text-slate-500 uppercase">{label}</div>
      <div className="mt-1 text-2xl font-semibold text-slate-900">{value}</div>
      <div className="mt-0.5 text-xs text-slate-500">{detail}</div>
    </div>
  )
}

export function ScoreTiles({ summary: s }: { summary: Summary }) {
  return (
    <div>
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <Tile
          label="Retrieval"
          value={`${pct(s.hit_at_8)} Hit@8`}
          detail={`MRR ${s.mrr.toFixed(2)} · correct post in the final 8`}
        />
        <Tile
          label="Relevance"
          value={s.relevance_avg === null ? '—' : `${s.relevance_avg.toFixed(1)} / 5`}
          detail="judge: does the answer address the question?"
        />
        <Tile
          label="Grounded"
          value={pct(s.grounded_rate)}
          detail="of claims backed by the post they cite"
        />
        <Tile
          label="Hallucination"
          value={pct(s.hallucination_rate)}
          detail={`of answers · declined correctly ${s.declined_correctly}/${s.unanswerable_total}`}
        />
      </div>
      <p className="mt-2 text-xs text-slate-500">
        Recall@50 by stage: BM25 {pct(s.stage_recall.bm25)} · Dense {pct(s.stage_recall.dense)} ·
        RRF {pct(s.stage_recall.rrf)} · Rerank top 8 {pct(s.stage_recall.rerank)}
      </p>
      <p className="mt-1 text-xs text-slate-500">
        FiQA labels only one or two posts per question, so a "miss" can still be a good answer from
        another relevant post. Relevance and grounding are judged by an LLM.
      </p>
    </div>
  )
}
