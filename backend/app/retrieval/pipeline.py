"""The full hybrid search: BM25 + dense in parallel -> RRF -> Cohere rerank, with a per-stage trace."""

import asyncio
import time
from typing import Protocol

import cohere
import numpy as np
from openai import AsyncOpenAI

from app.config import Settings
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.corpus import Corpus
from app.retrieval.dense import EMBEDDING_MODEL, DenseRetriever
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.rerank import CohereReranker
from app.schemas import RankedDoc, SearchTrace, Stage


class SparseSearch(Protocol):
    def search(self, query: str, k: int) -> list[tuple[str, float]]: ...


class DenseSearch(Protocol):
    async def search(self, query: str, k: int) -> list[tuple[str, float]]: ...


class Reranker(Protocol):
    def rerank(
        self, query: str, docs: list[tuple[str, str]], top_n: int
    ) -> list[tuple[str, float]]: ...


def _ranked(results: list[tuple[str, float]]) -> list[RankedDoc]:
    return [
        RankedDoc(doc_id=doc_id, rank=rank, score=score)
        for rank, (doc_id, score) in enumerate(results, start=1)
    ]


def _ms_since(start: float) -> float:
    return (time.perf_counter() - start) * 1000


class HybridPipeline:
    def __init__(
        self,
        corpus: Corpus,
        bm25: SparseSearch,
        dense: DenseSearch,
        reranker: Reranker,
        candidate_k: int,
        top_k: int,
        rrf_k: int,
    ) -> None:
        self.corpus = corpus
        self._bm25 = bm25
        self._dense = dense
        self._reranker = reranker
        self._candidate_k = candidate_k
        self._top_k = top_k
        self._rrf_k = rrf_k

    @classmethod
    def from_settings(cls, settings: Settings) -> "HybridPipeline":
        corpus = Corpus.load(settings.data_dir)
        openai_client = AsyncOpenAI(api_key=settings.openai_api_key)

        async def embed(query: str) -> np.ndarray:
            response = await openai_client.embeddings.create(model=EMBEDDING_MODEL, input=[query])
            return np.array(response.data[0].embedding, dtype=np.float32)

        return cls(
            corpus=corpus,
            bm25=BM25Retriever(settings.data_dir / "bm25"),
            dense=DenseRetriever(settings.data_dir / "embeddings.npy", corpus.ids, embed),
            reranker=CohereReranker(cohere.ClientV2(api_key=settings.cohere_api_key)),
            candidate_k=settings.candidate_k,
            top_k=settings.top_k,
            rrf_k=settings.rrf_k,
        )

    async def search(self, query: str) -> SearchTrace:
        start = time.perf_counter()

        async def timed_bm25() -> Stage:
            t = time.perf_counter()
            results = await asyncio.to_thread(self._bm25.search, query, self._candidate_k)
            return Stage(name="bm25", ms=_ms_since(t), results=_ranked(results))

        async def timed_dense() -> Stage:
            t = time.perf_counter()
            results = await self._dense.search(query, self._candidate_k)
            return Stage(name="dense", ms=_ms_since(t), results=_ranked(results))

        bm25_stage, dense_stage = await asyncio.gather(timed_bm25(), timed_dense())

        t = time.perf_counter()
        fused = reciprocal_rank_fusion(
            [[d.doc_id for d in bm25_stage.results], [d.doc_id for d in dense_stage.results]],
            k=self._rrf_k,
        )
        # Blank posts can't be cited or reranked, so they stop here.
        candidates = [(doc_id, s) for doc_id, s in fused if not self.corpus.is_blank(doc_id)]
        candidates = candidates[: self._candidate_k]
        rrf_stage = Stage(name="rrf", ms=_ms_since(t), results=_ranked(candidates))

        t = time.perf_counter()
        reranked: list[tuple[str, float]] = []
        if candidates:
            docs = [(doc_id, self.corpus.text(doc_id)) for doc_id, _ in candidates]
            reranked = await asyncio.to_thread(self._reranker.rerank, query, docs, self._top_k)
        rerank_stage = Stage(name="rerank", ms=_ms_since(t), results=_ranked(reranked))

        return SearchTrace(
            query=query,
            total_ms=_ms_since(start),
            stages=[bm25_stage, dense_stage, rrf_stage, rerank_stage],
        )
