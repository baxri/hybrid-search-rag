from app.retrieval.corpus import Corpus
from app.schemas import RankedDoc, SearchTrace, Stage
from tests.fixtures import FIXTURE_DOCS


def canned_trace(query: str, ranking: list[str]) -> SearchTrace:
    docs = [RankedDoc(doc_id=d, rank=i, score=1.0 / i) for i, d in enumerate(ranking, start=1)]
    return SearchTrace(
        query=query,
        total_ms=5.0,
        stages=[
            Stage(name=name, ms=1.0, results=docs) for name in ("bm25", "dense", "rrf", "rerank")
        ],
    )


class FakePipeline:
    """Stand-in for HybridPipeline so API tests never touch real indexes or APIs.

    Every search returns `ranking` (default d0, d3) at every stage and records the query.
    """

    def __init__(self, ranking: list[str] | None = None, error: Exception | None = None) -> None:
        self.corpus = Corpus(list(FIXTURE_DOCS), list(FIXTURE_DOCS.values()))
        self.ranking = ranking or ["d0", "d3"]
        self.error = error
        self.calls: list[str] = []

    async def search(self, query: str) -> SearchTrace:
        self.calls.append(query)
        if self.error:
            raise self.error
        return canned_trace(query, self.ranking)
