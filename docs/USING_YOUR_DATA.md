# Using your own data

This repo ships with FiQA, 57,638 personal-finance forum posts, but the search and the agent
work with any collection of text: a handbook, support tickets, product docs, notes. This guide
takes you from a file of documents to a deployed chat that answers from them with citations.

The whole thing runs without a vector database: the BM25 index and the embeddings are loaded
into the app's memory at startup.

**You need:** Python 3.12+, [`uv`](https://docs.astral.sh/uv/), Node 22+, an OpenAI API key
(embeddings and answers) and a Cohere API key (reranking).

## 1. Prepare your documents

One file, `.csv`, `.jsonl` or `.parquet`, with one row per document:

| Column  | Required | Notes                                                        |
| ------- | -------- | ------------------------------------------------------------ |
| `id`    | yes      | Unique, no spaces. Shown next to each cited source.           |
| `text`  | yes      | Plain text. Convert PDFs and HTML to text first.              |
| `title` | no       | Kept with the document; not searched.                        |

Your columns can have other names; pass them with `--id-column`, `--text-column` and
`--title-column`.

```jsonl
{"id": "leave-policy", "title": "Leave policy", "text": "Employees accrue 1.5 days of paid leave per month..."}
{"id": "expenses", "title": "Expense claims", "text": "Submit receipts within 30 days of purchase..."}
```

## 2. Build the indexes

```bash
cd backend
uv sync
echo "OPENAI_API_KEY=sk-..." >> .env
echo "COHERE_API_KEY=..." >> .env

uv run python -m scripts.build_indexes --input path/to/docs.jsonl --out data
```

This writes the three things the app loads:

- `data/corpus.parquet`: the searchable text
- `data/bm25/`: the keyword index
- `data/embeddings.npy`: one OpenAI `text-embedding-3-small` vector per chunk

**Chunking.** A document longer than 300 words is split into 300-word chunks that overlap by
50 words, and each chunk is searched and cited on its own as `<id>#1`, `<id>#2`, and so on. A
shorter document stays whole and keeps its id. Change the sizes with `--chunk-words` and
`--overlap-words`. Smaller chunks give more precise citations; larger ones give the model more
context per source.

**Cost and time.** Embedding costs about $0.03 per million words and runs at your OpenAI
account's rate limit, which the script waits out on its own. FiQA's 57k posts take about
10 minutes.

**Size.** Each chunk's embedding takes 6 KB of memory. A 1 GB server fits roughly 100,000
chunks alongside the app; beyond that, use a bigger server. At millions of chunks, a vector
database starts to earn its keep.

## 3. Tell the assistant what it's answering

The built-in instructions say the assistant answers personal-finance questions from forum
posts. Write your own in a text file inside `backend/app/`, so the Docker image includes it:

```markdown
You answer questions about Acme's employee handbook, using its sections.
- For any handbook question, call {tool_name} first. You may search again with a different query if the first results are weak. Rewrite follow-up questions into standalone search queries.
- Answer only from the returned sections. Cite every claim with the section's number in square brackets, e.g. [1] or [2][3]. Never cite a number you were not given in this turn.
- Never answer from your own knowledge. If the sections don't cover the question, say so plainly.
- Keep answers concise.
- For greetings or small talk, reply briefly without searching.
```

`{tool_name}` is replaced with the search tool's name. Keep the citation rule: the app turns
`[n]` into links to the sources. The built-in FiQA instructions in
[`backend/app/agent.py`](../backend/app/agent.py) are a good template; each of their rules
fixed a failure found by the eval.

Then point the settings at it, in `backend/.env` or as environment variables:

```bash
INSTRUCTIONS_FILE=app/instructions.md
SEARCH_TOOL_NAME=search_handbook
SEARCH_TOOL_DESCRIPTION="Search Acme's employee handbook (keywords + embeddings, reranked)."
```

`INSTRUCTIONS_FILE` is relative to `backend/`, where the app runs both locally and in Docker.

## 4. Run it

```bash
# backend/
uv run uvicorn app.main:app --reload          # http://localhost:8000

# frontend/, in a second terminal
npm ci
npm run dev                                   # http://localhost:5173
```

Ask a few questions and open the pipeline panel: it shows what BM25, dense search, fusion and
the reranker each returned, so you can see why a source was or wasn't picked.

## 5. Make the site yours (optional)

The frontend's wording is about FiQA. To rebrand it, edit:

- `frontend/src/components/Landing.tsx`: headline, pitch, numbers and stack
- `frontend/src/lib/examples.ts`: the example questions
- `frontend/src/App.tsx`: the header subtitle
- `frontend/src/components/PipelinePanel.tsx` and `ArchitecturePage.tsx`: the tool name
  `search_finance_posts`
- `frontend/index.html`: the page title
- `frontend/src/lib/links.ts`: the repository and docs links

The Evaluation tab shows `frontend/public/eval/results.json`. Replace it with your own run
(below) or remove the tab from `App.tsx`.

## 6. Measure it (optional)

The eval in `backend/evals/` scores retrieval against labeled answers and has an LLM judge
check every claim against its cited source. It's set up for FiQA, so to use it:

1. Write `backend/evals/questions.json` by hand: a list of
   `{"id", "text", "kind", "relevant_doc_ids"}`. Use `"kind": "fiqa"` (the name is historical)
   with the chunk ids that answer the question, or `"kind": "unanswerable"` with an empty list
   for questions your documents can't answer. Twenty or thirty questions already tell you a
   lot.
2. Edit the judge's description of the assistant in `backend/evals/judge.py` (it says
   "personal-finance assistant").
3. Run `uv run python -m evals.run`. It writes the report to the Evaluation tab's file.

Don't run `python -m evals.dataset`: it rebuilds the FiQA question set.

## 7. Deploy to Railway

The Docker image downloads the indexes from a Hugging Face dataset while it builds, so
publish yours there first:

```bash
cd backend
uv run hf auth login        # a token with write access
uv run python -m scripts.upload_indexes --repo <you>/<name>-indexes --src data
```

The upload is public by default, and its dataset card describes FiQA: edit `DATASET_CARD` in
[`backend/scripts/upload_indexes.py`](../backend/scripts/upload_indexes.py) first. To keep
private documents off Hugging Face, build the image with your `data/` folder copied in instead
of downloaded.

Then, in Railway:

1. Create a project from your fork of this repo. [`railway.json`](../railway.json) builds the
   `Dockerfile` and health-checks `/api/health`.
2. Under **Variables**, add `OPENAI_API_KEY`, `COHERE_API_KEY`, `HF_REPO=<you>/<name>-indexes`
   and your `INSTRUCTIONS_FILE`, `SEARCH_TOOL_NAME` and `SEARCH_TOOL_DESCRIPTION`.
3. Deploy: connect the GitHub repo so every push deploys, or run `railway up`.

Give the service at least as much memory as step 2 suggests for your corpus.

## Things to know

- **One document can fill the top 8.** If a long document matches well, several of its chunks
  can take every slot. That's usually fine for answering. If you want answers drawn from more
  documents, use larger chunks, so each document has fewer of them.
- **Search settings** (`CANDIDATE_K`, `TOP_K`, `RRF_K`, `RERANK_MODEL`, `CHAT_MODEL`) are in
  the README's Configuration table. The defaults were tuned on FiQA; measure before changing
  them.
- **Changing the embedding model** means rebuilding the indexes: the app embeds queries with
  the same model, `text-embedding-3-small`.
