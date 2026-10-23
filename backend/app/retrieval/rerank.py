"""Cross-encoder rerank with Cohere: reads query and post together, so it scores relevance best."""

import time

import cohere
from cohere.errors import TooManyRequestsError

RERANK_MODEL = "rerank-v4.0-fast"


class CohereReranker:
    def __init__(
        self,
        client: cohere.ClientV2,
        model: str = RERANK_MODEL,
        max_retries: int = 3,
        backoff_seconds: float = 2.0,
    ) -> None:
        self._client = client
        self._model = model
        self._max_retries = max_retries
        self._backoff_seconds = backoff_seconds

    def rerank(
        self, query: str, docs: list[tuple[str, str]], top_n: int
    ) -> list[tuple[str, float]]:
        """Rerank (doc_id, text) pairs; returns (doc_id, relevance) best first."""
        documents = [text for _, text in docs]
        for attempt in range(self._max_retries + 1):
            try:
                response = self._client.rerank(
                    model=self._model, query=query, documents=documents, top_n=top_n
                )
                break
            except TooManyRequestsError:
                if attempt == self._max_retries:
                    raise
                time.sleep(self._backoff_seconds)
        return [(docs[r.index][0], r.relevance_score) for r in response.results]
