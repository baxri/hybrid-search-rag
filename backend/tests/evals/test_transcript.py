from app.agent import build_agent
from evals.transcript import collect
from tests.conftest import FakePipeline, scripted_model


async def test_collects_answer_trace_and_sources():
    pipeline = FakePipeline(ranking=["d0", "d3"])
    agent = build_agent(scripted_model(["emergency fund"], "Use a savings account [1]."))

    transcript = await collect("Where should I keep my emergency fund?", agent, pipeline)

    assert transcript.answer == "Use a savings account [1]."
    assert transcript.error is None
    assert [t.query for t in transcript.searches] == ["emergency fund"]
    assert [(s.n, s.doc_id) for s in transcript.sources] == [(1, "d0"), (2, "d3")]
    assert pipeline.calls == ["emergency fund"]


async def test_pipeline_failure_is_recorded_not_raised():
    pipeline = FakePipeline(error=RuntimeError("cohere down"))
    agent = build_agent(scripted_model(["emergency fund"]))

    transcript = await collect("Where should I keep my emergency fund?", agent, pipeline)

    assert transcript.error is not None
    assert transcript.searches == []
