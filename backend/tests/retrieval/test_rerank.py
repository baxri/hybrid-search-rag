from types import SimpleNamespace

from cohere.errors import TooManyRequestsError

from app.retrieval.rerank import CohereReranker


class FakeCohereClient:
    def __init__(self, results: list[SimpleNamespace]) -> None:
        self.results = results
        self.calls: list[dict] = []

    def rerank(self, **kwargs) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(results=self.results)


def test_rerank_maps_indices_to_ids():
    client = FakeCohereClient(
        [
            SimpleNamespace(index=1, relevance_score=0.9),
            SimpleNamespace(index=0, relevance_score=0.2),
        ]
    )
    reranker = CohereReranker(client)

    results = reranker.rerank("q", [("id0", "text zero"), ("id1", "text one")], top_n=2)

    assert results == [("id1", 0.9), ("id0", 0.2)]
    assert client.calls[0]["documents"] == ["text zero", "text one"]
    assert client.calls[0]["model"] == "rerank-v4.0-fast"


class RateLimitedOnceClient(FakeCohereClient):
    def rerank(self, **kwargs) -> SimpleNamespace:
        if not self.calls:
            self.calls.append(kwargs)
            raise TooManyRequestsError(body=None)
        return super().rerank(**kwargs)


def test_rerank_retries_after_rate_limit():
    client = RateLimitedOnceClient([SimpleNamespace(index=0, relevance_score=0.5)])
    reranker = CohereReranker(client, backoff_seconds=0)

    assert reranker.rerank("q", [("id0", "t")], top_n=1) == [("id0", 0.5)]
    assert len(client.calls) == 2
