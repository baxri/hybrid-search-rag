"""The fixed question set: 50 FiQA test questions with known-correct posts, plus 8 the forum
can't answer.

Built once and committed, so every run scores the same questions:
    uv run python -m evals.dataset
"""

import argparse
from pathlib import Path
from typing import Literal

import pandas as pd
from huggingface_hub import hf_hub_download
from pydantic import BaseModel, TypeAdapter

QUESTIONS_PATH = Path(__file__).parent / "questions.json"
SAMPLE_SIZE = 50
SEED = 42  # same sample as exp-python/hybrid_retrieval/evaluate.py, so NDCG numbers compare


class EvalQuestion(BaseModel):
    id: str
    text: str
    kind: Literal["fiqa", "unanswerable"]
    relevant_doc_ids: list[str] = []  # FiQA's labeled correct posts; empty when unanswerable


def _unanswerable(n: int, text: str) -> EvalQuestion:
    return EvalQuestion(id=f"u{n}", text=text, kind="unanswerable")


# The right answer to each is "the posts don't cover this", never a made-up fact.
UNANSWERABLE = [
    # Live data
    _unanswerable(1, "What is Tesla's stock price today?"),
    _unanswerable(2, "What is the current 30-year fixed mortgage rate?"),
    # The user's own data
    _unanswerable(3, "How much money is in my 401(k) right now?"),
    _unanswerable(4, "What was my credit score last month?"),
    # Off-topic
    _unanswerable(5, "Who won the 2022 FIFA World Cup?"),
    _unanswerable(6, "What is a good recipe for banana bread?"),
    # Newer than the 2018 corpus
    _unanswerable(7, "What are the 2026 IRS contribution limits for a Roth IRA?"),
    _unanswerable(8, "How did the 2025 tax law change the standard deduction?"),
]


def sample_fiqa(
    queries: pd.DataFrame, qrels: pd.DataFrame, n: int = SAMPLE_SIZE, seed: int = SEED
) -> list[EvalQuestion]:
    """Sample `n` labeled test questions; queries has `_id, text`, qrels `query-id, corpus-id, score`."""
    positive = qrels[qrels["score"] > 0]
    relevant: dict[str, list[str]] = {}
    for query_id, doc_id in zip(positive["query-id"], positive["corpus-id"], strict=True):
        relevant.setdefault(str(query_id), []).append(str(doc_id))

    labeled = queries[queries["_id"].astype(str).isin(relevant.keys())]
    sample = labeled.sample(n=n, random_state=seed)
    return [
        EvalQuestion(id=str(qid), text=text, kind="fiqa", relevant_doc_ids=relevant[str(qid)])
        for qid, text in zip(sample["_id"], sample["text"], strict=True)
    ]


def build_question_set(
    queries: pd.DataFrame, qrels: pd.DataFrame, n: int = SAMPLE_SIZE
) -> list[EvalQuestion]:
    return sample_fiqa(queries, qrels, n=n) + UNANSWERABLE


_QUESTIONS = TypeAdapter(list[EvalQuestion])


def save_questions(questions: list[EvalQuestion], path: Path = QUESTIONS_PATH) -> None:
    path.write_bytes(_QUESTIONS.dump_json(questions, indent=2) + b"\n")


def load_questions(path: Path = QUESTIONS_PATH) -> list[EvalQuestion]:
    return _QUESTIONS.validate_json(path.read_bytes())


def download_fiqa() -> tuple[pd.DataFrame, pd.DataFrame]:
    """FiQA queries and test-split labels from the public BEIR repos on Hugging Face."""
    queries_file = hf_hub_download(
        "BeIR/fiqa", "queries/queries-00000-of-00001.parquet", repo_type="dataset"
    )
    qrels_file = hf_hub_download("BeIR/fiqa-qrels", "test.tsv", repo_type="dataset")
    queries = pd.read_parquet(queries_file)
    qrels = pd.read_csv(qrels_file, sep="\t", dtype={"query-id": str, "corpus-id": str})
    return queries, qrels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=QUESTIONS_PATH)
    args = parser.parse_args()
    queries, qrels = download_fiqa()
    questions = build_question_set(queries, qrels)
    save_questions(questions, args.out)
    print(f"Wrote {len(questions)} questions to {args.out}")


if __name__ == "__main__":
    main()
