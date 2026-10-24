"""Did the system retrieve a correct post? Scored exactly against FiQA's labels, no judge."""

from pydantic import BaseModel

from app.schemas import SearchTrace, StageName

STAGES: tuple[StageName, ...] = ("bm25", "dense", "rrf", "rerank")


class RetrievalScore(BaseModel):
    hit: bool  # a labeled post made the final reranked top 8
    first_rank: int | None  # best rank of any labeled post there
    mrr: float  # 1 / first_rank, 0 on a miss
    stage_recall: dict[StageName, float]  # share of labeled posts each stage found


def _ids(searches: list[SearchTrace], stage: StageName) -> set[str]:
    return {r.doc_id for t in searches for s in t.stages if s.name == stage for r in s.results}


def score_retrieval(searches: list[SearchTrace], relevant: list[str]) -> RetrievalScore:
    """Union of every search in the turn; an answer that never searched is a miss."""
    wanted = set(relevant)
    ranks = [
        r.rank
        for t in searches
        for s in t.stages
        if s.name == "rerank"
        for r in s.results
        if r.doc_id in wanted
    ]
    first_rank = min(ranks, default=None)

    return RetrievalScore(
        hit=first_rank is not None,
        first_rank=first_rank,
        mrr=1 / first_rank if first_rank else 0.0,
        stage_recall={
            stage: len(wanted & _ids(searches, stage)) / len(wanted) if wanted else 0.0
            for stage in STAGES
        },
    )
