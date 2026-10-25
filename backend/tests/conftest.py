from collections.abc import AsyncIterator

from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel

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
        self.questions: list[str | None] = []

    async def search(self, query: str, question: str | None = None) -> SearchTrace:
        self.calls.append(query)
        self.questions.append(question)
        if self.error:
            raise self.error
        return canned_trace(query, self.ranking)


def scripted_model(
    queries: list[str],
    answer: str = "Use a savings account [1].",
    tool_name: str = "search_finance_posts",
) -> FunctionModel:
    """Calls the search tool once per query (one per model step), then streams `answer`."""

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator:
        searches_done = sum(1 for m in messages for p in m.parts if p.part_kind == "tool-return")
        if searches_done < len(queries):
            yield {
                0: DeltaToolCall(
                    name=tool_name,
                    json_args=f'{{"query": "{queries[searches_done]}"}}',
                    tool_call_id=f"call-{searches_done}",
                )
            }
        else:
            yield answer

    return FunctionModel(stream_function=stream)
