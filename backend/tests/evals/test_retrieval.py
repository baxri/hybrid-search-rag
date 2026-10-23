from app.schemas import RankedDoc, SearchTrace, Stage
from evals.retrieval import score_retrieval


def trace(**stages: list[str]) -> SearchTrace:
    """A trace with the given doc ids, in rank order, per stage; unnamed stages are empty."""
    return SearchTrace(
        query="q",
        total_ms=1.0,
        stages=[
            Stage(
                name=name,
                ms=1.0,
                results=[
                    RankedDoc(doc_id=d, rank=i, score=1.0)
                    for i, d in enumerate(stages.get(name, []), start=1)
                ],
            )
            for name in ("bm25", "dense", "rrf", "rerank")
        ],
    )


def test_hit_at_rank_three():
    score = score_retrieval([trace(rerank=["x", "y", "good", "z"])], relevant=["good"])

    assert score.hit is True
    assert score.first_rank == 3
    assert score.mrr == 1 / 3


def test_miss_when_no_labeled_post_is_reranked():
    score = score_retrieval([trace(bm25=["good"], rerank=["x", "y"])], relevant=["good"])

    assert score.hit is False
    assert score.first_rank is None
    assert score.mrr == 0.0


def test_best_rank_across_several_searches_wins():
    searches = [trace(rerank=["x", "y", "good"]), trace(rerank=["other", "good"])]

    assert score_retrieval(searches, relevant=["good", "other"]).first_rank == 1


def test_no_search_is_a_miss():
    score = score_retrieval([], relevant=["good"])

    assert score.hit is False
    assert score.mrr == 0.0
    assert score.stage_recall == {"bm25": 0.0, "dense": 0.0, "rrf": 0.0, "rerank": 0.0}


def test_stage_recall_is_share_of_labeled_posts_found():
    score = score_retrieval(
        [trace(bm25=["a", "x"], dense=["a", "b"], rrf=["b"])], relevant=["a", "b"]
    )

    assert score.stage_recall == {"bm25": 0.5, "dense": 1.0, "rrf": 0.5, "rerank": 0.0}
