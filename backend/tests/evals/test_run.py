from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

from app.agent import build_agent
from evals.dataset import EvalQuestion
from evals.judge import build_judge
from evals.run import evaluate
from tests.conftest import FakePipeline, scripted_model

QUESTIONS = [
    EvalQuestion(id=str(i), text=f"question {i}", kind="fiqa", relevant_doc_ids=["d0"])
    for i in range(5)
]
VERDICT = {
    "relevance": 5,
    "declined": False,
    "claims": [{"text": "Use savings.", "cites": [1], "verdict": "supported"}],
    "reason": "ok",
}


async def test_every_question_gets_a_result_in_order():
    agent = build_agent(scripted_model(["emergency fund"]))
    judge = build_judge(TestModel(custom_output_args=VERDICT))

    results = await evaluate(QUESTIONS, agent, FakePipeline(), judge, concurrency=2)

    assert [r.question.id for r in results] == ["0", "1", "2", "3", "4"]
    assert all(r.judgement is not None and not r.failed for r in results)


async def test_a_judge_crash_is_recorded_on_that_question():
    def broken(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise RuntimeError("judge offline")

    agent = build_agent(scripted_model(["emergency fund"]))

    results = await evaluate(
        QUESTIONS[:1], agent, FakePipeline(), build_judge(FunctionModel(broken))
    )

    assert results[0].judgement is None
    assert results[0].error is not None and "judge" in results[0].error.lower()
    assert results[0].failed is True


def test_each_run_is_archived_and_published_as_latest(tmp_path):
    from evals.report import EvalReport, RunMeta, summarize
    from evals.run import write_report

    meta = RunMeta(
        date="2026-10-23T09:05:07+00:00",
        commit="abc1234-dirty",
        answer_model="m",
        judge_model="m",
        seed=42,
        question_count=0,
    )
    report = EvalReport(meta=meta, summary=summarize([]), results=[])
    latest, runs = tmp_path / "public" / "results.json", tmp_path / "runs"

    first = write_report(report, latest, runs)
    second = write_report(
        report.model_copy(
            update={"meta": meta.model_copy(update={"date": "2026-10-23T10:00:00+00:00"})}
        ),
        latest,
        runs,
    )

    assert first.name == "2026-10-23T09-05-07_abc1234-dirty_n0.json"
    assert sorted(p.name for p in runs.iterdir()) == [first.name, second.name]
    assert latest.read_text() == second.read_text()


class FlakyPipeline(FakePipeline):
    """Fails the first `failures` searches, like a rate-limited reranker, then recovers."""

    def __init__(self, failures: int) -> None:
        super().__init__()
        self.failures = failures

    async def search(self, query: str, question: str | None = None):
        if self.failures:
            self.failures -= 1
            raise RuntimeError("429 Too Many Requests")
        return await super().search(query, question)


async def test_a_failed_question_is_retried_after_a_wait():
    agent = build_agent(scripted_model(["emergency fund"]))
    judge = build_judge(TestModel(custom_output_args=VERDICT))

    results = await evaluate(QUESTIONS[:1], agent, FlakyPipeline(failures=1), judge, retry_wait=0)

    assert results[0].error is None and not results[0].failed


async def test_retries_give_up_and_keep_the_error():
    agent = build_agent(scripted_model(["emergency fund"]))
    judge = build_judge(TestModel(custom_output_args=VERDICT))

    results = await evaluate(QUESTIONS[:1], agent, FlakyPipeline(failures=99), judge, retry_wait=0)

    assert results[0].error is not None and results[0].failed
