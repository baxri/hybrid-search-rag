import pandas as pd

from evals.dataset import (
    UNANSWERABLE,
    EvalQuestion,
    build_question_set,
    load_questions,
    sample_fiqa,
    save_questions,
)

QUERIES = pd.DataFrame(
    {
        "_id": [str(i) for i in range(10)],
        "text": [f"question {i}" for i in range(10)],
    }
)
# Queries 0-5 have labels; 6-9 are training-only queries with none in the test split.
QRELS = pd.DataFrame(
    {
        "query-id": ["0", "0", "1", "2", "3", "4", "5", "5"],
        "corpus-id": ["a", "b", "c", "d", "e", "f", "g", "h"],
        "score": [1, 1, 1, 1, 1, 1, 1, 0],
    }
)


def test_samples_only_labeled_questions():
    questions = sample_fiqa(QUERIES, QRELS, n=6, seed=1)

    assert sorted(q.id for q in questions) == ["0", "1", "2", "3", "4", "5"]
    assert all(q.kind == "fiqa" for q in questions)


def test_relevant_ids_come_from_positive_labels():
    by_id = {q.id: q for q in sample_fiqa(QUERIES, QRELS, n=6, seed=1)}

    assert by_id["0"].relevant_doc_ids == ["a", "b"]
    assert by_id["0"].text == "question 0"
    assert by_id["5"].relevant_doc_ids == ["g"]  # score 0 is not a correct post


def test_same_seed_gives_same_sample_of_size_n():
    first = sample_fiqa(QUERIES, QRELS, n=3, seed=42)
    second = sample_fiqa(QUERIES, QRELS, n=3, seed=42)

    assert len(first) == 3
    assert [q.id for q in first] == [q.id for q in second]


def test_excluded_questions_are_replaced_by_the_next_in_order():
    original = [q.id for q in sample_fiqa(QUERIES, QRELS, n=3, seed=7, excluded=frozenset())]
    full_order = [q.id for q in sample_fiqa(QUERIES, QRELS, n=6, seed=7, excluded=frozenset())]

    replaced = sample_fiqa(QUERIES, QRELS, n=3, seed=7, excluded=frozenset({original[1]}))

    assert full_order[:3] == original
    assert [q.id for q in replaced] == [original[0], original[2], full_order[3]]


def test_unanswerable_questions_have_no_labels():
    assert len(UNANSWERABLE) == 8
    assert len({q.id for q in UNANSWERABLE}) == 8
    assert all(q.kind == "unanswerable" and q.relevant_doc_ids == [] for q in UNANSWERABLE)


def test_question_set_is_fiqa_sample_then_unanswerable():
    questions = build_question_set(QUERIES, QRELS, n=4)

    assert [q.kind for q in questions] == ["fiqa"] * 4 + ["unanswerable"] * 8


def test_save_and_load_round_trip(tmp_path):
    path = tmp_path / "questions.json"
    questions = [
        EvalQuestion(id="0", text="q", kind="fiqa", relevant_doc_ids=["a"]),
        UNANSWERABLE[0],
    ]

    save_questions(questions, path)

    assert load_questions(path) == questions
