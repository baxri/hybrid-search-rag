from pydantic_ai.messages import FunctionToolResultEvent

from app.agent import ChatDeps, build_agent
from app.citations import SourceRegistry
from tests.conftest import FakePipeline, scripted_model


async def run(
    queries: list[str], pipeline: FakePipeline
) -> tuple[list[FunctionToolResultEvent], list[tuple[int, str]]]:
    started: list[tuple[int, str]] = []

    async def on_search(search_id: int, query: str) -> None:
        started.append((search_id, query))

    deps = ChatDeps(pipeline=pipeline, registry=SourceRegistry(), on_search=on_search)
    agent = build_agent(scripted_model(queries))
    results: list[FunctionToolResultEvent] = []
    async with agent.run_stream_events("where to keep emergency fund?", deps=deps) as stream:
        async for event in stream:
            if isinstance(event, FunctionToolResultEvent):
                results.append(event)
    return results, started


async def test_finance_question_calls_search_tool():
    pipeline = FakePipeline()

    results, started = await run(["emergency fund savings"], pipeline)

    assert pipeline.calls == ["emergency fund savings"]
    assert started == [(1, "emergency fund savings")]
    assert results[0].part.content.startswith("[1] ")


async def test_tool_metadata_carries_trace_and_sources():
    results, _ = await run(["emergency fund savings"], FakePipeline(ranking=["d0", "d3"]))

    metadata = results[0].part.metadata
    assert metadata.search_id == 1
    assert len(metadata.trace.stages) == 4
    assert [(s.n, s.doc_id) for s in metadata.sources] == [(1, "d0"), (2, "d3")]


async def test_two_searches_get_ids_1_and_2():
    results, started = await run(["emergency fund", "CDs emergency fund"], FakePipeline())

    assert [r.part.metadata.search_id for r in results] == [1, 2]
    assert [search_id for search_id, _ in started] == [1, 2]
    # Same canned posts both times: the second search adds no new sources.
    assert results[1].part.metadata.sources == []
    assert results[1].part.content.startswith("[1] ")
