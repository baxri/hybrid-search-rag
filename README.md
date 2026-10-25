# Hybrid Search RAG

A chat assistant for personal-finance questions, answered from **57,638 real forum posts**
([FiQA-2018](https://huggingface.co/datasets/BeIR/fiqa)). Every answer cites its sources, and
you can watch each retrieval stage work in real time.

```
           ┌──────────── BM25 (keywords) ────────────┐
question ──┤                                         ├──► RRF fusion ──► Cohere rerank ──► LLM answer [1][2]
           └──────── embeddings (meaning) ───────────┘
              top 50 each, run in parallel             merge by rank      top 8 posts      cited, grounded
```

## Why hybrid?

Neither search method works well alone:

| Query                              | BM25 (keywords)          | Embeddings (meaning)      |
| ---------------------------------- | ------------------------ | ------------------------- |
| "ESPP tax treatment", "401(k)"     | ✅ exact terms match      | ⚠️ codes can get blurred   |
| "where to park my rainy-day fund"  | ⚠️ no shared words        | ✅ finds "short-term savings" |

The app runs both and merges them with **Reciprocal Rank Fusion**, which uses only rank
positions, so the two very different score scales never need calibrating. A **Cohere
cross-encoder** then reads the query and each candidate together and keeps the best 8.

## Features

- **Agentic retrieval.** A [Pydantic AI](https://ai.pydantic.dev) agent decides when to search,
  rewrites follow-ups into standalone queries, and can search again if results are weak.
- **Citations.** Every claim is tagged `[n]` and linked to the forum post it came from.
- **Pipeline panel.** Shows each stage (BM25, dense, RRF, rerank), what it returned, and how
  long it took, streamed live over NDJSON.
- **Evaluation tab.** The site shows the latest eval run: scores, per-question results, and
  failures.
- **One container.** FastAPI serves both the API and the built React app from the same origin.

## Evaluation

58 fixed questions: 50 FiQA test questions with labeled correct posts, plus 8 the forum
can't answer. Retrieval is scored exactly against FiQA's labels. Answers are scored by an
LLM judge.

FiQA questions are forum titles; the askers' original text isn't in the dataset. Thirteen sampled
questions whose labeled posts reply to that missing text (e.g. "Alternative means of salary for
my employees", labeled with a post about paying staff in a self-made currency), or are a few of
many equally good generic answers, were replaced by the next questions in the same seeded
order. They're listed with reasons in [`backend/evals/dataset.py`](backend/evals/dataset.py).

| Metric                                        | Score    |
| --------------------------------------------- | -------- |
| Hit@8 (a labeled post in the final top 8)     | **92%**  |
| MRR                                           | **0.72** |
| Answer relevance (judge, 1–5)                 | **5.0**  |
| Grounded claims                               | **99.2%** |
| Answers with any unsupported claim            | **3.4%** |
| Unanswerable questions correctly declined     | **7 / 8** |

How many labeled posts each stage found: BM25 55%, dense 76%, RRF 76% (all from the top 50
candidates). Rerank found 69%, but it only keeps the top 8.

*Run from 2026-10-25, `gpt-5.4-mini` for answers and judging. Full results are in
[`frontend/public/eval/results.json`](frontend/public/eval/results.json). Past runs are in
[`backend/evals/runs/`](backend/evals/runs/).*

## Tech stack

| Layer     | Tools                                                                           |
| --------- | ------------------------------------------------------------------------------- |
| Backend   | Python 3.12, FastAPI, Pydantic AI, `bm25s`, NumPy, `uv`                          |
| Models    | OpenAI `gpt-5.4-mini` (chat) and `text-embedding-3-small`, Cohere `rerank-v4.0-pro` |
| Frontend  | React 19, TypeScript, Vite, Tailwind CSS 4                                       |
| Data      | Prebuilt indexes on [Hugging Face](https://huggingface.co/datasets/baxri/fiqa-hybrid-indexes) (~400 MB) |
| Deploy    | Docker, Railway                                                                  |

## Run locally

**Prerequisites:** Python 3.12+, [`uv`](https://docs.astral.sh/uv/), Node 22+, and OpenAI and
Cohere API keys.

```bash
# 1. Backend: install, add keys, fetch the indexes (~400 MB, public, no token needed)
cd backend
uv sync
cat > .env <<'EOF'
OPENAI_API_KEY=sk-...
COHERE_API_KEY=...
EOF
uv run python scripts/download_indexes.py --repo baxri/fiqa-hybrid-indexes --dest data
uv run uvicorn app.main:app --reload          # http://localhost:8000

# 2. Frontend (new terminal): Vite proxies /api to the backend
cd frontend
npm ci
npm run dev                                   # http://localhost:5173
```

### Or with Docker

```bash
docker build -t hybrid-search-rag .
docker run -p 8000:8000 -e OPENAI_API_KEY=sk-... -e COHERE_API_KEY=... hybrid-search-rag
```

The image builds the frontend and bakes in the indexes, so the container starts ready to go.

## Configuration

Settings are read from environment variables, or from `backend/.env`:

| Variable          | Default        | Purpose                                 |
| ----------------- | -------------- | --------------------------------------- |
| `OPENAI_API_KEY`  | *required*     | Chat model and query embeddings         |
| `COHERE_API_KEY`  | *required*     | Reranking                               |
| `CHAT_MODEL`      | `gpt-5.4-mini` | Model that writes the answers           |
| `JUDGE_MODEL`     | `gpt-5.4-mini` | Model that scores answers (evals only)  |
| `CANDIDATE_K`     | `50`           | Candidates from each of BM25 and dense  |
| `TOP_K`           | `8`            | Posts kept after reranking              |
| `RRF_K`           | `60`           | RRF smoothing constant                  |
| `RERANK_MODEL`    | `rerank-v4.0-pro` | Cohere reranker                      |
| `DATA_DIR`        | `backend/data` | Where the indexes are loaded from       |
| `INSTRUCTIONS_FILE` | built-in FiQA text | Your own agent instructions       |
| `SEARCH_TOOL_NAME` | `search_finance_posts` | The search tool's name for the model |
| `SEARCH_TOOL_DESCRIPTION` | FiQA description | What the model is told it searches |

To run it on your own documents, see [Using your own data](docs/USING_YOUR_DATA.md).

## Tests and evals

```bash
cd backend
uv run pytest                     # unit tests (no network, uses fake models)
uv run ruff check .

uv run python -m evals.run --limit 3   # quick smoke eval
uv run python -m evals.run             # all 58 questions (calls OpenAI and Cohere)

cd ../frontend
npm test
npm run lint
```

A full eval run costs between a few cents and about a dollar and takes about a minute
(8 questions at a time). With a Cohere trial key, reranking is limited to 10 calls a minute:
run with `--concurrency 1`, and the runner retries when it hits the limit.

## Deploy to Railway

[`railway.json`](railway.json) tells Railway to build the `Dockerfile` and to health-check
`/api/health`. To deploy:

1. Create a Railway project from this GitHub repo.
2. Under **Variables**, add `OPENAI_API_KEY` and `COHERE_API_KEY`.
3. Push to the deploy branch. Railway builds the image, injects `PORT`, and runs it.

## Project layout

```
backend/
  app/
    main.py          FastAPI app: /api/chat (NDJSON stream), /api/health, serves the SPA
    agent.py         Pydantic AI agent with a single search tool
    retrieval/       bm25 · dense · fusion (RRF) · rerank (Cohere) · pipeline
    citations.py     numbers sources and formats them for the model
    stream.py        streams the agent's events to the UI
  evals/             question set, retrieval scoring, LLM judge, runner, report
  scripts/           build indexes from your own documents; download/upload them on Hugging Face
  tests/
frontend/src/
  components/        chat, pipeline panel, sources, landing page, architecture page, eval/
docs/                guide to using your own data, design specs and implementation plans
Dockerfile           multi-stage build: React build + Python runtime with indexes
```

## Data and license

The corpus is FiQA-2018 via [BEIR](https://github.com/beir-cellar/beir), licensed CC BY-SA 4.0.
Answers summarize forum opinions. They are not financial advice.
