"""Was the answer relevant, and is every claim backed by the post it cites? An LLM judge decides.

The judge sees what the answering model saw (the question and the numbered posts) plus the
answer. It never sees FiQA's labels: grounding is about the posts given, not the "right" post.
"""

from typing import Literal

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models import Model

from app.citations import format_for_model
from app.schemas import Source
from evals.transcript import Transcript

JUDGE_INSTRUCTIONS = """\
You grade answers from a personal-finance assistant that must answer only from the numbered \
forum posts it was given, citing them as [n].

1. relevance (1-5): does the answer address the question asked?
   5 = directly and fully answers it; 3 = partly, or drifts; 1 = ignores it.
   An honest "the posts don't cover this" to an unanswerable question is a 5.
2. declined: true if the answer says the posts don't cover the question instead of answering it.
3. claims: split the answer into its statements. For each give the [n] numbers it cites and a \
verdict:
   - not_a_claim: not a factual claim about finance or the world, wherever it appears and even \
when it cites posts. That is: disclaimers ("not financial advice", "these are forum opinions"), \
greetings, offers and questions to the user ("send me the ticker"), generic hedges ("it \
depends", "it depends on your situation"), and anything about the posts or the search ("the \
posts don't cover 2026", "I couldn't find the ASX rule", "the posts don't say X").
   - supported: a cited post states it or clearly implies it as stated, including its dates, \
names and specifics. A post about a different year, law or proposal does not support it.
   - unsupported: a factual claim that cites posts, but none of them backs it.
   - uncited: a factual claim with no citation.
   Judge only against the posts shown. Your own knowledge does not make a claim supported.
4. reason: one sentence a reader can check, naming the weakest point if there is one."""


class Claim(BaseModel):
    text: str
    cites: list[int]
    verdict: Literal["supported", "unsupported", "uncited", "not_a_claim"]


class Judgement(BaseModel):
    relevance: int = Field(ge=1, le=5)
    declined: bool
    claims: list[Claim]
    reason: str


def build_judge(model: Model | str) -> Agent[None, Judgement]:
    return Agent(model, output_type=Judgement, instructions=JUDGE_INSTRUCTIONS)


def judge_prompt(question: str, answer: str, sources: list[Source]) -> str:
    # Same text, cut at the same length, as the answering model was shown.
    posts = format_for_model(sources) if sources else "No sources were retrieved."
    return f"Question:\n{question}\n\nPosts given to the assistant:\n{posts}\n\nAnswer:\n{answer}"


async def judge_answer(
    judge: Agent[None, Judgement], question: str, transcript: Transcript
) -> Judgement:
    result = await judge.run(judge_prompt(question, transcript.answer, transcript.sources))
    return result.output
