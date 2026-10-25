"""Per-question verdicts and the scorecard numbers. Pure functions: no API calls here."""

from statistics import mean

from pydantic import BaseModel

from app.schemas import StageName
from evals.dataset import EvalQuestion
from evals.judge import Judgement
from evals.retrieval import STAGES, RetrievalScore, score_retrieval
from evals.transcript import Transcript

LOW_RELEVANCE = 2  # a relevance score at or below this is a failure
UNGROUNDED = ("unsupported", "uncited")  # a "not_a_claim" statement is never one


class QuestionResult(BaseModel):
    question: EvalQuestion
    transcript: Transcript
    retrieval: RetrievalScore | None  # FiQA questions only
    judgement: Judgement | None  # None when the run or the judge errored
    error: str | None
    hallucinated: bool
    failed: bool
    labeled_posts: dict[str, str] = {}  # doc id -> text of each labeled post, for the report


def make_result(
    question: EvalQuestion,
    transcript: Transcript,
    judgement: Judgement | None,
    error: str | None = None,
    labeled_posts: dict[str, str] | None = None,
) -> QuestionResult:
    error = error or transcript.error
    fiqa = question.kind == "fiqa"
    retrieval = score_retrieval(transcript.searches, question.relevant_doc_ids) if fiqa else None

    hallucinated = judgement is not None and (
        any(c.verdict in UNGROUNDED for c in judgement.claims)
        or (not fiqa and not judgement.declined)
    )
    failed = (
        error is not None
        or hallucinated
        or (retrieval is not None and not retrieval.hit)
        or (fiqa and judgement is not None and judgement.relevance <= LOW_RELEVANCE)
    )
    return QuestionResult(
        question=question,
        transcript=transcript,
        retrieval=retrieval,
        judgement=judgement,
        error=error,
        hallucinated=hallucinated,
        failed=failed,
        labeled_posts=labeled_posts or {},
    )


class Summary(BaseModel):
    fiqa_total: int
    unanswerable_total: int
    errors: int
    failures: int
    # Retrieval, over every FiQA question (an errored one counts as a miss)
    hit_at_8: float
    mrr: float
    stage_recall: dict[StageName, float]
    # Judge-based; None when nothing was judged
    relevance_avg: float | None  # judged FiQA answers
    grounded_rate: float | None  # supported / factual claims (not not_a_claim), every judged answer
    hallucination_rate: float | None  # judged answers with any unsupported or uncited claim
    declined_correctly: int  # unanswerable questions that declined without inventing facts


class RunMeta(BaseModel):
    date: str
    commit: str
    answer_model: str
    judge_model: str
    seed: int
    question_count: int


class EvalReport(BaseModel):
    meta: RunMeta
    summary: Summary
    results: list[QuestionResult]


def _mean(values: list[float]) -> float | None:
    return mean(values) if values else None


def summarize(results: list[QuestionResult]) -> Summary:
    retrieval = [r.retrieval for r in results if r.retrieval is not None]
    judged = [(r.question, r.judgement, r) for r in results if r.judgement is not None]
    claims = [c for _, j, _ in judged for c in j.claims if c.verdict != "not_a_claim"]
    return Summary(
        fiqa_total=len(retrieval),
        unanswerable_total=sum(r.question.kind == "unanswerable" for r in results),
        errors=sum(r.error is not None for r in results),
        failures=sum(r.failed for r in results),
        hit_at_8=_mean([float(s.hit) for s in retrieval]) or 0.0,
        mrr=_mean([s.mrr for s in retrieval]) or 0.0,
        stage_recall={
            stage: _mean([s.stage_recall[stage] for s in retrieval]) or 0.0 for stage in STAGES
        },
        relevance_avg=_mean([j.relevance for q, j, _ in judged if q.kind == "fiqa"]),
        grounded_rate=(
            sum(c.verdict == "supported" for c in claims) / len(claims) if claims else None
        ),
        hallucination_rate=_mean([float(r.hallucinated) for _, _, r in judged]),
        declined_correctly=sum(q.kind == "unanswerable" and not r.failed for q, _, r in judged),
    )
