# RAG Evaluation Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** An offline evaluation of the real chat pipeline over 58 questions, published as an Evaluation tab on the site.

**Architecture:** A `backend/evals/` package drives the same `chat_events` stream the API serves, scores retrieval exactly against FiQA labels, scores answers with a Pydantic AI judge, and writes one results JSON into `frontend/public/eval/`. The frontend reads that static file; no new API route.

**Tech Stack:** Python 3.12, Pydantic AI (typed judge output), pandas, huggingface_hub; React + Tailwind for the page.

**Spec:** `docs/specs/2026-10-23-rag-evaluation-design.md`

## Global Constraints

- Imports nothing from `exp-python`.
- Tests never call OpenAI, Cohere or Hugging Face (fakes: `FakePipeline`, `scripted_model`, Pydantic AI `TestModel`).
- Owner reviews each backend task before its commit; frontend in one go. Never commit or push without explicit approval.
- Owner said "do not run yet": don't run the evaluation or the dataset download until asked. Unit tests are fine.
- Judge model default = answer model `gpt-5.4-mini`, configurable (`Settings.judge_model`).
- 50 FiQA questions (seed 42, `DataFrame.sample`) + 8 unanswerable.

## Review Focus

1. A question that errors mid-run is recorded as an error and a failure, never dropped and never a pass.
2. An unanswerable question answered confidently counts as a hallucination.
3. An answer with no search (agent skipped the tool) on a FiQA question is a retrieval miss, not a crash.
4. Results JSON stays loadable by the page when a field is empty (no searches, no claims, judge error).
5. Multiple searches in one turn: retrieval uses the union, best rank wins.

---

### Task 1: Question set

**Files:** Create `backend/evals/__init__.py`, `backend/evals/dataset.py`, `backend/tests/evals/__init__.py`, `backend/tests/evals/test_dataset.py`. (`backend/evals/questions.json` is generated later by the owner-approved download.)

**Produces:**
- `EvalQuestion(BaseModel)`: `id: str`, `text: str`, `kind: Literal["fiqa", "unanswerable"]`, `relevant_doc_ids: list[str] = []`
- `UNANSWERABLE: list[EvalQuestion]`, 8 questions, ids `u1`..`u8`, two each of live data / private data / off-topic / too recent.
- `sample_fiqa(queries: pd.DataFrame, qrels: pd.DataFrame, n=50, seed=42) -> list[EvalQuestion]`: queries columns `_id, text`; qrels columns `query-id, corpus-id, score`; keep only queries with a qrel, `.sample(n=n, random_state=seed)`; relevant ids = qrels rows with `score > 0`.
- `build_question_set(queries, qrels) -> list[EvalQuestion]` = sample + `UNANSWERABLE`
- `save_questions(questions, path)`, `load_questions(path=QUESTIONS_PATH) -> list[EvalQuestion]`
- CLI `python -m evals.dataset`: `hf_hub_download` `BeIR/fiqa` `queries/queries-00000-of-00001.parquet` and `BeIR/fiqa-qrels` `test.tsv`, writes `QUESTIONS_PATH`.

- [ ] Failing tests: sample keeps only queries that have qrels; relevant ids match qrels; same seed → same ids; `n` respected; 8 unanswerable with no relevant ids and unique ids; save/load round-trip.
- [ ] Implement → PASS, ruff clean. Owner review, then commit `Add evaluation question set builder`.

### Task 2: Transcript collection and retrieval scores

**Files:** Create `backend/evals/transcript.py`, `backend/evals/retrieval.py`, tests `tests/evals/test_transcript.py`, `tests/evals/test_retrieval.py`.

**Produces:**
- `Transcript(BaseModel)`: `searches: list[SearchTrace]`, `sources: list[Source]`, `answer: str`, `error: str | None`
- `async collect(question: str, agent, pipeline) -> Transcript`: builds a one-message `ChatRequest`, consumes `chat_events` and parses each line with `TypeAdapter(ChatEvent)`.
- `RetrievalScore(BaseModel)`: `hit: bool`, `first_rank: int | None`, `mrr: float`, `stage_recall: dict[StageName, float]`
- `score_retrieval(searches, relevant: list[str]) -> RetrievalScore`: rerank results of all searches unioned, `first_rank` = best rank of any relevant doc, `mrr = 1/first_rank`; `stage_recall[s]` = share of relevant ids present in stage `s` of any search; no searches → miss, all zeros.

- [ ] Failing tests: collect with `FakePipeline` + `scripted_model` returns answer, one trace, sources; pipeline error → `error` set; hit at rank 3 → mrr 1/3; miss; two searches → best rank; no searches → miss; stage recall 1/2 when one of two relevant docs is in BM25.
- [ ] Implement → PASS. Owner review, then commit `Collect chat transcripts and score retrieval for evaluation`.

### Task 3: Judge

**Files:** Create `backend/evals/judge.py`, `tests/evals/test_judge.py`; Modify `backend/app/config.py` (`judge_model: str = "gpt-5.4-mini"`).

**Produces:**
- `Claim(BaseModel)`: `text: str`, `cites: list[int]`, `verdict: Literal["supported", "unsupported", "uncited"]`
- `Judgement(BaseModel)`: `relevance: int` (1–5), `declined: bool`, `claims: list[Claim]`, `reason: str`
- `build_judge(model) -> Agent[None, Judgement]` with `JUDGE_INSTRUCTIONS`
- `judge_prompt(question, answer, sources) -> str`: question, answer, `[n] text` per source, or "No sources were retrieved."
- `async judge_answer(judge, question, transcript) -> Judgement`

- [ ] Failing tests: prompt carries question, answer and every `[n]` source; empty sources wording; `judge_answer` with `TestModel(custom_output_args=...)` returns a typed `Judgement`; relevance outside 1–5 rejected.
- [ ] Implement → PASS. Owner review, then commit `Add LLM judge for relevance and grounding`.

### Task 4: Runner and report

**Files:** Create `backend/evals/report.py`, `backend/evals/run.py`, `tests/evals/test_report.py`, `tests/evals/test_run.py`.

**Produces:**
- `QuestionResult`: `question`, `transcript`, `retrieval: RetrievalScore | None` (FiQA only), `judgement: Judgement | None`, `error: str | None`, `hallucinated: bool`, `failed: bool`
- `make_result(question, transcript, judgement, error) -> QuestionResult`. Hallucinated: any claim not `supported`; unanswerable also if not `declined`. Failed per spec: miss, relevance ≤ 2, hallucinated, unanswerable not declined, or error.
- `Summary`: counts, `errors`, `hit_at_8`, `mrr`, `stage_recall`, `relevance_avg`, `grounded_rate` (supported claims / all claims), `hallucination_rate` (over judged answers), `declined_correctly`, `unanswerable_total`
- `RunMeta`: `date`, `commit`, `answer_model`, `judge_model`, `seed`, `question_count`
- `EvalReport(meta, summary, results)`, `summarize(results) -> Summary`
- `async evaluate(questions, agent, pipeline, judge, concurrency=3) -> list[QuestionResult]`: keeps question order; a judge exception becomes that question's `error`.
- CLI `python -m evals.run [--limit N]` → `frontend/public/eval/results.json`.

- [ ] Failing tests: summarize math on hand-built results; declined unanswerable passes; confident unanswerable is hallucinated; uncited claim is hallucinated; error is a failure and counted, not dropped; `evaluate` end-to-end with fakes keeps order and length.
- [ ] Implement → PASS. Owner review, then commit `Add evaluation runner and report`.

### Task 5: Evaluation page

**Files:** Create `frontend/src/lib/evalTypes.ts`, `frontend/src/components/eval/{EvalPage,ScoreTiles,ResultsTable}.tsx`; Modify `frontend/src/App.tsx` (Chat | Evaluation tabs, `#eval` hash).

- Fetch `/eval/results.json`; missing → "No evaluation published yet."
- Layout A per spec: run line, four tiles + stage recall line + labels caveat, table with "Failures only" toggle, expandable rows with answer (reuses `AnswerText`), judge verdicts, "Try this in chat →" (switch to Chat, `send(question)`).
- [ ] Build + lint + existing tests green; manual check with a small hand-made results file.
- [ ] Owner approval, then commit `Add evaluation report page`.
