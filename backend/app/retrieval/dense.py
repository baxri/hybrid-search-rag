"""Dense retrieval: cosine similarity over precomputed text-embedding-3-small vectors.

Wins on paraphrase ("park my rainy-day fund" ~ "short-term savings").
"""

from collections.abc import Awaitable, Callable
from pathlib import Path

import numpy as np

EMBEDDING_MODEL = "text-embedding-3-small"

Embedder = Callable[[str], Awaitable[np.ndarray]]


class DenseRetriever:
    def __init__(self, embeddings_path: Path, doc_ids: list[str], embed: Embedder) -> None:
        raw = np.load(embeddings_path)
        # Normalize once so cosine similarity is a single dot product per query.
        self._embeddings = raw / np.linalg.norm(raw, axis=1, keepdims=True)
        self._doc_ids = doc_ids
        self._embed = embed

    async def search(self, query: str, k: int) -> list[tuple[str, float]]:
        vector = await self._embed(query)
        scores = self._embeddings @ (vector / np.linalg.norm(vector))
        top_k = np.argsort(-scores)[:k]
        return [(self._doc_ids[i], float(scores[i])) for i in top_k]
