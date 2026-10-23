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
