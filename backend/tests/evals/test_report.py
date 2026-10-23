from evals.dataset import EvalQuestion
from evals.judge import Claim, Judgement
from evals.report import make_result, summarize
from evals.transcript import Transcript
from tests.conftest import canned_trace

FIQA = EvalQuestion(id="1", text="q", kind="fiqa", relevant_doc_ids=["good"])
UNANSWERABLE = EvalQuestion(id="u1", text="Tesla price?", kind="unanswerable")

FOUND = Transcript(searches=[canned_trace("q", ["good", "x"])], answer="a [1]")
MISSED = Transcript(searches=[canned_trace("q", ["x", "y"])], answer="a [1]")


def judgement(*verdicts: str, relevance: int = 5, declined: bool = False) -> Judgement:
    return Judgement(
        relevance=relevance,
        declined=declined,
        claims=[Claim(text=f"claim {i}", cites=[1], verdict=v) for i, v in enumerate(verdicts)],
        reason="",
    )


def test_grounded_answer_with_a_hit_passes():
    result = make_result(FIQA, FOUND, judgement("supported", "supported"))

    assert result.retrieval is not None and result.retrieval.first_rank == 1
    assert (result.hallucinated, result.failed) == (False, False)


def test_unsupported_or_uncited_claim_is_a_hallucination():
    for verdict in ("unsupported", "uncited"):
        result = make_result(FIQA, FOUND, judgement("supported", verdict))
        assert (result.hallucinated, result.failed) == (True, True)


def test_retrieval_miss_or_low_relevance_fails_without_hallucinating():
    missed = make_result(FIQA, MISSED, judgement("supported"))
    off_topic = make_result(FIQA, FOUND, judgement("supported", relevance=2))

    assert (missed.hallucinated, missed.failed) == (False, True)
    assert (off_topic.hallucinated, off_topic.failed) == (False, True)


def test_unanswerable_passes_only_when_declined():
    declined = make_result(
        UNANSWERABLE, Transcript(answer="Not covered."), judgement(declined=True)
    )
    confident = make_result(UNANSWERABLE, Transcript(answer="$250."), judgement("uncited"))

    assert declined.retrieval is None
    assert (declined.hallucinated, declined.failed) == (False, False)
    assert (confident.hallucinated, confident.failed) == (True, True)


def test_unanswerable_answered_without_claims_still_fails_if_not_declined():
    result = make_result(UNANSWERABLE, Transcript(answer="Hmm."), judgement(declined=False))

    assert (result.hallucinated, result.failed) == (True, True)


def test_an_error_is_a_failure_not_a_pass():
    result = make_result(FIQA, Transcript(error="Something went wrong."), None)

    assert result.error == "Something went wrong."
    assert (result.hallucinated, result.failed) == (False, True)


def test_summary_math():
    results = [
        make_result(FIQA, FOUND, judgement("supported", "supported", relevance=5)),
        make_result(FIQA, MISSED, judgement("supported", "unsupported", relevance=3)),
        make_result(FIQA, Transcript(error="boom"), None),
        make_result(UNANSWERABLE, Transcript(answer="Not covered."), judgement(declined=True)),
        make_result(UNANSWERABLE, Transcript(answer="$250."), judgement("uncited")),
    ]

    s = summarize(results)

    assert (s.fiqa_total, s.unanswerable_total, s.errors, s.failures) == (3, 2, 1, 3)
    assert s.hit_at_8 == 1 / 3  # the errored question counts as a miss, not dropped
    assert s.mrr == 1 / 3
    assert s.stage_recall["bm25"] == 1 / 3
    assert s.relevance_avg == 4.0  # FiQA answers that were judged
    assert s.grounded_rate == 3 / 5  # supported claims over all claims
    assert s.hallucination_rate == 2 / 4  # over judged answers
    assert s.declined_correctly == 1
