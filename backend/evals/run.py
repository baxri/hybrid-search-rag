"""Evaluate the real pipeline and agent on the fixed question set and publish the report.

Calls OpenAI and Cohere (cents to about a dollar per full run). From backend/:
    uv run python -m evals.run            # all 58 questions
    uv run python -m evals.run --limit 3  # quick smoke run
Every run is kept in evals/runs/, and the latest is copied to frontend/public/eval/results.json,
which the site's Evaluation tab shows.
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
RUNS_DIR = BACKEND_DIR / "evals" / "runs"
CONCURRENCY = 8  # a production Cohere key; a trial key (10 reranks a minute) needs 1
RETRIES = 2  # a question that errors (usually a rate limit) is tried again...
RETRY_WAIT_S = 60.0  # ...after the rate-limit window has passed


async def evaluate_question(
    question: EvalQuestion,
    agent: Agent[ChatDeps, str],
    pipeline: HybridPipeline,
    judge: Agent[None, Judgement],
    retry_wait: float = RETRY_WAIT_S,
) -> QuestionResult:
    transcript = await collect(question.text, agent, pipeline)
    for _ in range(RETRIES):
        if not transcript.error:
            break
        logger.warning("question %s errored, retrying in %.0fs", question.id, retry_wait)
        await asyncio.sleep(retry_wait)
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
    retry_wait: float = RETRY_WAIT_S,
) -> list[QuestionResult]:
    """One result per question, in question order; failures are results, never exceptions."""
    limit = asyncio.Semaphore(concurrency)
    done = 0

    async def one(question: EvalQuestion) -> QuestionResult:
        nonlocal done
        async with limit:
            result = await evaluate_question(question, agent, pipeline, judge, retry_wait)
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


def write_report(report: EvalReport, latest: Path, runs_dir: Path) -> Path:
    """Keep this run under its own name and publish it as the latest; returns the run's file."""
    stamp = report.meta.date[:19].replace(":", "-")
    run_file = runs_dir / f"{stamp}_{report.meta.commit}_n{report.meta.question_count}.json"
    text = report.model_dump_json(indent=2) + "\n"
    for path in (run_file, latest):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return run_file


async def main(limit: int | None, out: Path, concurrency: int) -> None:
    settings = get_settings()
    provider = OpenAIProvider(api_key=settings.openai_api_key)
    agent = build_agent(OpenAIResponsesModel(settings.chat_model, provider=provider))
    judge = build_judge(OpenAIResponsesModel(settings.judge_model, provider=provider))
    pipeline = HybridPipeline.from_settings(settings)

    questions = load_questions()[:limit]
    results = await evaluate(questions, agent, pipeline, judge, concurrency)
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
    run_file = write_report(report, out, RUNS_DIR)

    s = report.summary
    print(f"\nWrote {run_file}\n  and {out}")
    print(f"  Hit@8 {s.hit_at_8:.0%}   MRR {s.mrr:.2f}   failures {s.failures}   errors {s.errors}")
    print(f"  relevance {s.relevance_avg}   grounded {s.grounded_rate}")
    print(f"  hallucination {s.hallucination_rate}   declined {s.declined_correctly}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="only the first N questions")
    parser.add_argument("--out", type=Path, default=RESULTS_PATH)
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    asyncio.run(main(args.limit, args.out, args.concurrency))
