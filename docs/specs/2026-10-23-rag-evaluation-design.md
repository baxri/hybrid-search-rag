# RAG Evaluation Report — Design

Date: 2026-10-23
Status: approved 2026-10-23

Extends `docs/specs/2026-10-22-hybrid-search-chat-design.md`. That spec listed
retrieval evaluation as a non-goal; this one supersedes that line.

## Intent

The chat demo shows the RAG pipeline working on one question at a time. A
recruiter can't tell from that whether it works *in general*. This adds a
measured, reproducible evaluation over a fixed question set and publishes
the results on the live site as an **Evaluation** page, failures included,
with a "Try this in chat" button on every question so the visitor can watch
any case run live.

It also gives the owner a regression check: change the pipeline, re-run one
command, compare the numbers.

The four questions the evaluation answers (owner's framing):

1. Did the system retrieve the correct document?
2. Was the answer relevant?
3. Was the answer actually supported by the retrieved context?
4. How often did it hallucinate?

## Goals

- One command, run by the owner locally, evaluates the real pipeline and
  agent end to end and writes a results file.
- The live site shows those results on an Evaluation page (layout A below),
  with no API cost per visitor.
- Every number on the page is reproducible: the run records its date, git
  commit, answer model, judge model, sample seed, and question count.
- Failures are visible and explained, not hidden.

## Non-goals

- Running evaluations from the website. Visitors only view published results.
- CI gating or thresholds that fail a build.
- Multi-turn evaluation. Every eval question is a fresh single-turn chat.
- Comparing pipeline variants side by side (BM25-only vs hybrid etc.). The
  per-stage retrieval numbers below give a taste; a variant matrix is later.
- Scoring answers against a written reference answer. FiQA's "known answers"
  are labeled posts, not reference text.

## Anti-goals

- Numbers that flatter the system by hiding or dropping failed questions.
- An eval that tests different code than the site serves. The runner must
  drive the same `HybridPipeline`, `build_agent`, and `chat_events` stream the
  API uses.
- A run that silently counts an errored question as a pass, or silently drops it.

## The question set (58 questions)

- **50 FiQA test questions** with known-correct posts. FiQA ships 648 test
  questions; `qrels` labels the correct post(s) for each (avg 2.6, max 15).
  Sample 50 with seed 42 the same way `exp-python/hybrid_retrieval/evaluate.py`
  does (filter to queries that have test qrels, `DataFrame.sample(n=50,
  random_state=42)`), so the retrieval numbers are comparable with the NDCG
  numbers the owner already has.
- **8 hand-written unanswerable questions**: things the forum can't answer:
  live data ("What's Tesla's stock price today?"), the user's private data
  ("How much is in my 401(k)?"), off-topic ("Who won the 2022 World Cup?"),
  too recent ("What are the 2026 IRS contribution limits?"). The correct
  behavior is to say the posts don't cover it without inventing facts.
- The set is built once by a script and **committed** as a small JSON file
  (question id, text, kind `fiqa` | `unanswerable`, relevant doc ids), so
  runs never depend on `exp-python` or on re-sampling. The build script
  fetches FiQA queries and test qrels from the public Hugging Face repos
  `BeIR/fiqa` (queries) and `BeIR/fiqa-qrels` (test split), the same sources
  `exp-python` used.

## What gets measured

All four are computed per question and aggregated on the scorecard.

1. **Retrieval** (FiQA questions only, exact, no judge):
   - **Hit@8**: a labeled post is in the final reranked top 8.
   - **MRR@8**: 1 / rank of the first labeled post in the top 8 (0 if none).
   - **Recall@50 per stage**: share of labeled posts present in BM25, dense,
     and RRF candidate lists. It shows what each stage contributes.
   - If the agent searched more than once, use the union of all its
     searches. If it didn't search at all, it's a miss.
   - Shown with a caveat: FiQA labels are incomplete, so a good unlabeled
     post counts as a miss. The judge scores below catch that case.
2. **Relevance** (judge): 1–5, does the answer address the question asked.
   Not scored for unanswerable questions.
3. **Grounding** (judge): the judge splits the answer into factual claims;
   for each it records the cited source number(s) and whether those sources
   support it (`supported` | `unsupported` | `uncited`). Scorecard number =
   supported claims / all claims across answers. An `uncited` factual claim
   counts as not supported.
4. **Hallucination rate**: share of answers with at least one claim that
   isn't `supported`. For unanswerable questions, pass = the answer declines
   and makes no unsupported claim. Shown as its own line ("Declined
   correctly 7/8"), and a failed unanswerable counts as a hallucination.

**The judge** is a Pydantic AI agent with a typed output (relevance score,
claims list, whether the answer declines to answer, one-sentence reason).
The hallucination verdict is derived in code from the claims, not asked of
the judge separately. It sees the question,
the answer, and the numbered sources the answering model was given in that
turn. It never sees the FiQA labels. The judge model is a setting, and **for
now it is the same model as the answer model (`gpt-5.4-mini`)** (owner's
decision). The report shows both model names.

## The runner

- Single command run locally by the owner from `backend/`; it needs the
  same `.env` keys and local index data as the app.
- For each question: run it through the real chat stream and collect the
  events (search traces, numbered sources, answer text, error). Then judge it.
- Errors are recorded per question and shown on the report as errors. They
  never count as passes and are never dropped.
- Limited concurrency so the Cohere key's rate limit and OpenAI limits are
  respected. The existing rerank retry/backoff covers transient 429s.
- Cost per run: roughly 58 answer runs + 58 judge calls on `gpt-5.4-mini`
  plus ~60 rerank calls, so cents to about a dollar. On a Cohere trial key
  (10 calls/min) a run takes several minutes.
- Output: one results JSON (run metadata, aggregate scores, per-question
  details) written into the frontend's static files and committed, so it
  ships in the Docker image and the site serves it with no backend work.

## The Evaluation page (layout A, owner's choice)

- Header gets two tabs: **Chat** | **Evaluation**. The tab is reflected in the
  URL hash (`#eval`) so the page can be linked directly.
- Top: a run line (date, commit, answer model, judge model, "50 FiQA + 8
  unanswerable"), then four score tiles: Retrieval (Hit@8, MRR), Relevance
  (avg /5), Grounded (% of claims), Hallucination (% of answers, plus
  "Declined correctly n/8"). A one-line note explains the incomplete-labels
  caveat. A small per-stage Recall@50 line sits under the retrieval tile.
- Below: a table of all 58 questions: question, Found? (✅ rank / ❌ / — for
  unanswerable), relevance, grounded claims (e.g. 3/4, red if any
  unsupported), expand arrow. A "Failures only" toggle filters the table.
  A question is a failure if any holds: retrieval miss (FiQA), relevance
  ≤ 2, any claim not `supported`, unanswerable not declined, or a run error.
- Expanding a row shows: labeled post id(s) and whether/where each was
  retrieved, the retrieved top 8, the answer with the same citation badges
  as chat, the judge's per-claim verdicts and reason, and a **Try this in
  chat →** button.
- **Try this in chat** switches to the Chat tab and sends the question as a
  new message in the current conversation.
- If the results file is missing, the tab shows "No evaluation published yet."

## Constraints

- Evaluation code lives in this repo and imports nothing from `exp-python`.
- Same review cadence as the main build: backend tasks are reviewed by the
  owner task by task before each commit; frontend may be built in one go.
  Never commit or push without the owner's explicit approval.
- Tests use fakes (scripted model, fake pipeline) and never call paid APIs.

## Left to the builder

- Module layout under `backend/` and exact JSON field names.
- How FiQA parquet files are fetched from Hugging Face.
- Concurrency level, judge prompt wording, the exact 8 unanswerable questions
  (within the four categories above).
- Visual details of the page within layout A.

## Calls made by Claude (not discussed with the owner)

- 50 FiQA questions use the same seed/sampling as `exp-python` for
  comparability, plus 8 unanswerable, 58 total.
- "Try this in chat" sends into the current conversation rather than
  clearing it.
- `uncited` factual claims count against grounding.
- Results JSON is committed into the frontend static files (no new API
  route).
- Per-stage Recall@50 shown as a secondary retrieval line.
- Sequencing: finish the main plan's Tasks 11–12 (Docker, Railway) first,
  then build this, so the eval measures the deployed pipeline.
