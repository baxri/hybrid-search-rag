import pytest
from pydantic import ValidationError
from pydantic_ai.models.test import TestModel

from app.citations import MAX_SOURCE_CHARS_FOR_MODEL
from app.schemas import Source
from evals.judge import Judgement, build_judge, judge_answer, judge_prompt
from evals.transcript import Transcript

SOURCES = [
    Source(n=1, doc_id="d0", text="Keep an emergency fund in a high-yield savings account."),
    Source(n=2, doc_id="d3", text="CDs charge a penalty for early withdrawal."),
]


def test_prompt_carries_question_answer_and_every_numbered_source():
    prompt = judge_prompt("Where to keep it?", "A savings account [1].", SOURCES)

    assert "Where to keep it?" in prompt
    assert "A savings account [1]." in prompt
    assert "[1] Keep an emergency fund in a high-yield savings account." in prompt
    assert "[2] CDs charge a penalty for early withdrawal." in prompt


def test_judge_sees_posts_cut_where_the_answering_model_saw_them():
    long_post = Source(n=1, doc_id="d0", text="a" * MAX_SOURCE_CHARS_FOR_MODEL + "HIDDEN TAIL")

    prompt = judge_prompt("q", "answer [1]", [long_post])

    assert "a" * MAX_SOURCE_CHARS_FOR_MODEL in prompt
    assert "HIDDEN TAIL" not in prompt


def test_prompt_says_when_nothing_was_retrieved():
    prompt = judge_prompt("Tesla price today?", "The posts don't cover that.", [])

    assert "No sources were retrieved." in prompt


async def test_judge_returns_a_typed_judgement():
    verdict = {
        "relevance": 4,
        "declined": False,
        "claims": [{"text": "Use a savings account.", "cites": [1], "verdict": "supported"}],
        "reason": "Answers the question from source 1.",
    }
    judge = build_judge(TestModel(custom_output_args=verdict))
    transcript = Transcript(answer="Use a savings account [1].", sources=SOURCES)

    judgement = await judge_answer(judge, "Where to keep it?", transcript)

    assert judgement == Judgement.model_validate(verdict)


def test_relevance_must_be_between_one_and_five():
    with pytest.raises(ValidationError):
        Judgement(relevance=6, declined=False, claims=[], reason="")
