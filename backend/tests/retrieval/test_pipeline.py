import asyncio
import time

from app.retrieval.corpus import Corpus
from app.retrieval.pipeline import HybridPipeline
from tests.fixtures import FIXTURE_DOCS


class FakeRetriever:
    """Returns a fixed ranking; optionally sleeps to simulate latency."""

    def __init__(self, ranking: list[str], delay: float = 0.0) -> None:
        self.ranking = ranking
        self.delay = delay

    def _results(self, k: int) -> list[tuple[str, float]]:
        return [(doc_id, 1.0 / rank) for rank, doc_id in enumerate(self.ranking[:k], start=1)]


class FakeBM25(FakeRetriever):
    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        time.sleep(self.delay)
        return self._results(k)


class FakeDense(FakeRetriever):
    async def search(self, query: str, k: int) -> list[tuple[str, float]]:
        await asyncio.sleep(self.delay)
        return self._results(k)


class ReversingReranker:
    def __init__(self) -> None:
        self.calls = 0

    def rerank(
        self, query: str, docs: list[tuple[str, str]], top_n: int
    ) -> list[tuple[str, float]]:
        self.calls += 1
        reversed_docs = list(reversed(docs))[:top_n]
        return [(doc_id, 1.0 - i / 10) for i, (doc_id, _) in enumerate(reversed_docs)]


def make_pipeline(bm25, dense, reranker=None, top_k: int = 3) -> HybridPipeline:
    return HybridPipeline(
        corpus=Corpus(list(FIXTURE_DOCS), list(FIXTURE_DOCS.values())),
        bm25=bm25,
        dense=dense,
        reranker=reranker or ReversingReranker(),
        candidate_k=5,
        top_k=top_k,
        rrf_k=60,
    )


def stage(trace, name):
    return next(s for s in trace.stages if s.name == name)


async def test_trace_has_four_stages_in_order():
    pipeline = make_pipeline(FakeBM25(["d1", "d0", "d2"]), FakeDense(["d0", "d3", "d5"]))

    trace = await pipeline.search("emergency fund")

    assert trace.query == "emergency fund"
    assert [s.name for s in trace.stages] == ["bm25", "dense", "rrf", "rerank"]
    assert all(s.ms >= 0 for s in trace.stages)
    assert len(stage(trace, "rerank").results) <= 3
    assert [d.rank for d in stage(trace, "bm25").results] == [1, 2, 3]
    assert stage(trace, "rrf").results[0].doc_id == "d0"  # top-2 in both lists


async def test_blank_docs_never_reach_rerank():
    pipeline = make_pipeline(FakeBM25(["d4", "d1"]), FakeDense(["d4", "d0"]))

    trace = await pipeline.search("q")

    assert stage(trace, "bm25").results[0].doc_id == "d4"
    assert "d4" not in [d.doc_id for d in stage(trace, "rrf").results]
    assert "d4" not in [d.doc_id for d in stage(trace, "rerank").results]


async def test_bm25_and_dense_run_concurrently():
    pipeline = make_pipeline(FakeBM25(["d1"], delay=0.2), FakeDense(["d0"], delay=0.2))

    trace = await pipeline.search("q")

    assert trace.total_ms < 350


async def test_no_candidates_skips_rerank():
    reranker = ReversingReranker()
    pipeline = make_pipeline(FakeBM25(["d4"]), FakeDense(["d4"]), reranker=reranker)

    trace = await pipeline.search("q")

    assert stage(trace, "rerank").results == []
    assert reranker.calls == 0
