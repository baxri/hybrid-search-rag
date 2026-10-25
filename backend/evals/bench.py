"""Retrieval-only bench: compare search configurations on the 50 FiQA questions, no chat model.

Deterministic and cheap. First-stage results and Cohere scores are cached in evals/.bench_cache,
so only the first pass calls the APIs. A cross-encoder scores each (query, post) pair on its own,
so one rerank call over every candidate any config might need serves all configs.

    uv run python -m evals.bench [--agent-run evals/runs/<run>.json]

--agent-run supplies the search query the agent actually wrote for each question (the live
system searches with that, not the user's wording).
"""

import argparse
import asyncio
import hashlib
import json
import time
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import cohere
from cohere.errors import TooManyRequestsError

from app.config import BACKEND_DIR, get_settings
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.pipeline import HybridPipeline
from evals.dataset import load_questions

CACHE_DIR = BACKEND_DIR / "evals" / ".bench_cache"
WIDE_K = 100  # first-stage depth cached; configs use 50 or 100 of it
TOP = 8
RERANK_MODELS = ("rerank-v4.0-fast", "rerank-v4.0-pro")
RERANK_GAP_S = 6.5  # trial key: 10 calls a minute


def _key(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()[:24]


class Cache:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.data: dict = json.loads(path.read_text()) if path.exists() else {}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data))


@dataclass
class Question:
    id: str
    text: str
    agent_query: str | None
    relevant: set[str]


async def first_stage(pipeline: HybridPipeline, cache: Cache, query: str) -> dict:
    k = _key("first", query)
    if k not in cache.data:
        bm25 = pipeline._bm25.search(query, WIDE_K)
        dense = await pipeline._dense.search(query, WIDE_K)
        cache.data[k] = {"bm25": [d for d, _ in bm25], "dense": [d for d, _ in dense]}
    return cache.data[k]


def rerank_scores(
    client: cohere.ClientV2,
    cache: Cache,
    pipeline: HybridPipeline,
    model: str,
    query: str,
    doc_ids: list[str],
) -> dict[str, float]:
    """Relevance of every doc to the query; only uncached docs are sent."""
    k = _key("rerank", model, query)
    scores: dict[str, float] = cache.data.setdefault(k, {})
    todo = [d for d in doc_ids if d not in scores and not pipeline.corpus.is_blank(d)]
    if todo:
        for attempt in range(6):
            try:
                time.sleep(RERANK_GAP_S)
                response = client.rerank(
                    model=model,
                    query=query,
                    top_n=len(todo),
                    documents=[pipeline.corpus.text(d) for d in todo],
                )
                break
            except TooManyRequestsError:
                time.sleep(20 * (attempt + 1))
        else:
            raise RuntimeError("Cohere kept rate-limiting")
        for r in response.results:
            scores[todo[r.index]] = r.relevance_score
        cache.save()
    return scores


def fused(lists: list[list[str]], k: int, depth: int, corpus) -> list[str]:
    ranking = [d for d, _ in reciprocal_rank_fusion([lst[:depth] for lst in lists], k=60)]
    return [d for d in ranking if not corpus.is_blank(d)][:k]


def final_top(candidates: list[str], scores: dict[str, float], blend: bool) -> list[str]:
    by_rerank = sorted(candidates, key=lambda d: -scores.get(d, -1.0))
    if not blend:
        return by_rerank[:TOP]
    # Blend: RRF over the reranker's order and the first-stage fused order.
    return [d for d, _ in reciprocal_rank_fusion([by_rerank, candidates], k=60)][:TOP]


async def main(agent_run: Path | None) -> None:
    settings = get_settings()
    pipeline = HybridPipeline.from_settings(settings)
    client = cohere.ClientV2(api_key=settings.cohere_api_key)
    cache = Cache(CACHE_DIR / "cache.json")

    agent_queries: dict[str, str] = {}
    if agent_run:
        for r in json.loads(agent_run.read_text())["results"]:
            if r["transcript"]["searches"]:
                agent_queries[r["question"]["id"]] = r["transcript"]["searches"][0]["query"]

    questions = [
        Question(q.id, q.text, agent_queries.get(q.id), set(q.relevant_doc_ids))
        for q in load_questions()
        if q.kind == "fiqa"
    ]

    # 1. First stage for every query wording, and the union of candidates any config can use.
    first: dict[str, dict[str, dict]] = {}
    for q in questions:
        first[q.id] = {"orig": await first_stage(pipeline, cache, q.text)}
        if q.agent_query:
            first[q.id]["agent"] = await first_stage(pipeline, cache, q.agent_query)
    cache.save()

    def candidates(q: Question, source: str, depth: int) -> list[str]:
        runs = first[q.id]
        if source == "both" and "agent" in runs:
            lists = [runs[w][s] for w in ("orig", "agent") for s in ("bm25", "dense")]
        else:
            w = "agent" if source == "agent" and "agent" in runs else "orig"
            lists = [runs[w]["bm25"], runs[w]["dense"]]
        return fused(lists, depth, depth, pipeline.corpus)

    # 2. Rerank scores: per model and per rerank-query wording, over all candidates.
    scores: dict[tuple[str, str, str], dict[str, float]] = {}
    for n, q in enumerate(questions, 1):
        pool = sorted({d for s in ("orig", "agent", "both") for d in candidates(q, s, WIDE_K)})
        for model, wording in product(RERANK_MODELS, ("orig", "agent")):
            query = q.agent_query if wording == "agent" and q.agent_query else q.text
            scores[(q.id, model, wording)] = rerank_scores(
                client, cache, pipeline, model, query, pool
            )
        print(f"[{n}/{len(questions)}] scored", flush=True)

    # 3. Every config, offline.
    rows = []
    for source, depth, model, wording, blend in product(
        ("agent", "orig", "both"), (50, 100), RERANK_MODELS, ("agent", "orig"), (False, True)
    ):
        if source == "orig" and wording == "agent":
            continue
        hits, rr, ceiling = 0, 0.0, 0
        for q in questions:
            cands = candidates(q, source, depth)
            ceiling += bool(q.relevant & set(cands))
            top = final_top(cands, scores[(q.id, model, wording)], blend)
            ranks = [i for i, d in enumerate(top, 1) if d in q.relevant]
            hits += bool(ranks)
            rr += 1 / ranks[0] if ranks else 0.0
        rows.append((hits, rr, source, depth, model, wording, blend, ceiling))

    n = len(questions)
    print(
        f"\n{'search with':12}{'cands':>6}  {'rerank model':18}{'rerank q':9}{'blend':6}"
        f"{'Hit@8':>7}{'MRR':>7}{'in cands':>10}"
    )
    for hits, rr, source, depth, model, wording, blend, ceiling in sorted(rows, reverse=True):
        print(
            f"{source:12}{depth:>6}  {model:18}{wording:9}{'yes' if blend else 'no':6}"
            f"{hits / n:>7.0%}{rr / n:>7.2f}{ceiling / n:>10.0%}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-run", type=Path, default=None)
    args = parser.parse_args()
    asyncio.run(main(args.agent_run))
