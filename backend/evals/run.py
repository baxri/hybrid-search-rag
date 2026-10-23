"""Evaluate the real pipeline and agent on the fixed question set and publish the report.

Calls OpenAI and Cohere (cents to about a dollar per full run). From backend/:
    uv run python -m evals.run            # all 58 questions
    uv run python -m evals.run --limit 3  # quick smoke run
Writes frontend/public/eval/results.json, which the site's Evaluation tab shows.
"""

import argparse
import asyncio
import logging
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.providers.openai import OpenAIProvider

from app.agent import ChatDeps, build_agent
from app.config import BACKEND_DIR, get_settings
from app.retrieval.pipeline import HybridPipeline
from evals.dataset import SEED, EvalQuestion, load_questions
from evals.judge import Judgement, build_judge, judge_answer
from evals.report import EvalReport, QuestionResult, RunMeta, make_result, summarize
from evals.transcript import collect

logger = logging.getLogger(__name__)

RESULTS_PATH = BACKEND_DIR.parent / "frontend" / "public" / "eval" / "results.json"
CONCURRENCY = 3  # keeps Cohere and OpenAI rate limits comfortable


async def evaluate_question(
    question: EvalQuestion,
    agent: Agent[ChatDeps, str],
    pipeline: HybridPipeline,
    judge: Agent[None, Judgement],
) -> QuestionResult:
    transcript = await collect(question.text, agent, pipeline)
    if transcript.error:
        return make_result(question, transcript, None)
    try:
        judgement = await judge_answer(judge, question.text, transcript)
    except Exception as exc:
        logger.exception("judge failed on question %s", question.id)
        return make_result(question, transcript, None, error=f"Judge failed: {exc}")
    return make_result(question, transcript, judgement)


async def evaluate(
    questions: list[EvalQuestion],
    agent: Agent[ChatDeps, str],
    pipeline: HybridPipeline,
    judge: Agent[None, Judgement],
    concurrency: int = CONCURRENCY,
) -> list[QuestionResult]:
    """One result per question, in question order; failures are results, never exceptions."""
    limit = asyncio.Semaphore(concurrency)
    done = 0

    async def one(question: EvalQuestion) -> QuestionResult:
        nonlocal done
        async with limit:
            result = await evaluate_question(question, agent, pipeline, judge)
        done += 1
        status = "error" if result.error else "fail" if result.failed else "pass"
        logger.info("[%d/%d] %s  %s", done, len(questions), status, question.text[:70])
        return result

    return list(await asyncio.gather(*(one(q) for q in questions)))


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "describe", "--always", "--dirty"],
            capture_output=True,
            text=True,
            check=True,
            cwd=BACKEND_DIR,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


async def main(limit: int | None, out: Path) -> None:
    settings = get_settings()
    provider = OpenAIProvider(api_key=settings.openai_api_key)
    agent = build_agent(OpenAIResponsesModel(settings.chat_model, provider=provider))
    judge = build_judge(OpenAIResponsesModel(settings.judge_model, provider=provider))
    pipeline = HybridPipeline.from_settings(settings)

    questions = load_questions()[:limit]
    results = await evaluate(questions, agent, pipeline, judge)
    report = EvalReport(
        meta=RunMeta(
            date=datetime.now(UTC).isoformat(timespec="seconds"),
            commit=_git_commit(),
            answer_model=settings.chat_model,
            judge_model=settings.judge_model,
            seed=SEED,
            question_count=len(questions),
        ),
        summary=summarize(results),
        results=results,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report.model_dump_json(indent=2) + "\n")

    s = report.summary
    print(f"\nWrote {out}")
    print(f"  Hit@8 {s.hit_at_8:.0%}   MRR {s.mrr:.2f}   failures {s.failures}   errors {s.errors}")
    print(f"  relevance {s.relevance_avg}   grounded {s.grounded_rate}")
    print(f"  hallucination {s.hallucination_rate}   declined {s.declined_correctly}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="only the first N questions")
    parser.add_argument("--out", type=Path, default=RESULTS_PATH)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    asyncio.run(main(args.limit, args.out))
