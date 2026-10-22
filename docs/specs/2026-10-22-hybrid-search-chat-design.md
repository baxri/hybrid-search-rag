# Hybrid Search Chat — Design

Date: 2026-10-22
Status: approved 2026-10-22

## Intent

A public portfolio demo that recruiters and clients open from a link and play
with. It is a chat app over the FiQA personal-finance corpus (~57k forum
answers about savings, taxes, investing). Visitors ask questions, get answers
streamed token by token, and every claim carries a numbered citation pointing
at the exact forum post it came from.

The point is to show end-to-end RAG engineering: a real hybrid retrieval
pipeline (BM25 + dense embeddings, fused with RRF, reranked with Cohere)
behind a FastAPI backend, a Pydantic AI agent that decides when and what to
search, streamed responses, and a React UI that renders citations cleanly.

## Goals

- A live public URL on Railway that works for anyone, no login.
- Streamed answers grounded in FiQA posts, with numbered inline citations.
- Multi-turn conversation: follow-ups like "what about CDs?" use earlier
  context.
- A repo that reads well to a recruiter: self-contained, tested, with this
  spec and an implementation plan alongside the code.

## Non-goals

- User accounts, saved conversations, or a database.
- Rate limiting, captchas, or access codes. The owner watches the API bill
  instead (owner's decision).
- Ingesting new documents or any corpus other than FiQA.
- Retrieval evaluation (NDCG etc.) — that lives in `exp-python` already.
- Observability/tracing tooling for v1.

## Anti-goals (would make this a failure even if it works)

- A citation marker that points at the wrong post, or at a post the search
  never returned.
- A long silent pause with nothing on screen after sending a message.
- Answers that invent facts when the retrieved posts don't cover the question.

## Constraints

- **Source of retrieval code:** copy from
  `/Users/georgebibilashvili/Projects/test/exp-python/hybrid_retrieval/`
  (`utils/retrievers.py`, `utils/fusion.py`, `utils/reranker.py`) into this
  repo and adapt. This repo must not import from `exp-python`.
  - Drop the Cohere trial-key throttle (`MIN_SECONDS_BETWEEN_CALLS = 6.5`):
    the owner is upgrading to a paid Cohere key.
  - Make the clients async-friendly or run sync calls in a threadpool so one
    slow search doesn't block other visitors.
  - Load indexes once at app startup, not per request.
- **Index files are too big for git** (`embeddings.npy` 338 MB, BM25 index
  31 MB, `corpus.parquet` 26 MB). They are hosted in a **public Hugging Face
  dataset** under the owner's account and downloaded with `huggingface_hub`
  at Docker build time (so restarts don't re-download). A one-time upload
  script lives in the repo; the owner authenticates with `hf auth login`
  themselves. Runtime needs no HF token.
- The embedding model must stay `text-embedding-3-small` (the index was
  built with it). Note: ~38 docs have blank text and were embedded as
  `"[empty document]"`; row order of embeddings matches `corpus.parquet`.
- **Memory:** the dense matrix is ~338 MB float32 in RAM, plus corpus and
  BM25. Size the Railway service accordingly.
- Python ≥ 3.12, managed with `uv`; built-in generics (`list[int]`), ruff.
- Secrets via environment variables: `OPENAI_API_KEY`, `COHERE_API_KEY`.
- Never commit or push without the owner's explicit approval.

## Decided by the owner

- FastAPI backend + React frontend.
- Pydantic AI for the agent, OpenAI as the LLM provider.
- Streamed responses.
- Citation style **A**: numbered superscript markers inline in the answer
  (`[1]`, `[2]`), with a "Sources" list under each answer showing a snippet
  and doc id per cited post; each source expands to the full post text.
- **"Under the hood" pipeline panel**, layout **A (stage-by-stage
  timeline)**, on the left of the screen, outside the chat. For the selected
  answer it shows: the tool call with the query the agent wrote and total
  time, then one card per stage — BM25, Dense, RRF fusion, Cohere rerank —
  each with its top results (rank, doc id, score) and timing. Posts cited in
  the answer are highlighted across every stage, and notable rank jumps are
  called out (e.g. "▲13 vs BM25"). Each card expands to the full top-N.
  This is the showcase feature: it makes the hybrid pipeline visible.
- Conversation memory: the frontend keeps the whole conversation and sends
  the entire history with every request.
- Tailwind for styling, kept simple and comfortable. No component library.
- Deploy publicly on Railway, wide open.
- Paid Cohere key, reranking stays in the live pipeline.
- Index files hosted on Hugging Face (public dataset).

## Calls made by Claude (owner can override)

- **Search as an agent tool**, not on every message. The agent has one tool,
  `search_finance_posts(query: str)`, which runs BM25 + dense → RRF → Cohere
  rerank and returns the top ~5–8 posts with ids and text. The agent writes
  its own query, so follow-ups get rewritten with conversation context.
  Small talk skips search.
- **Stateless backend.** No server-side sessions; refresh starts a new chat.
- **Single Railway service.** FastAPI serves the built React app as static
  files plus the API; one URL, no CORS. Multi-stage Dockerfile (node build →
  python runtime).
- **Frontend stack:** Vite + React + TypeScript + Tailwind.
- **Citation integrity is enforced in code, not trusted to the model.**
  Within one assistant turn, every post returned by the tool gets a stable
  number. The backend sends the list of retrieved sources (number, doc id,
  text) to the frontend as structured data alongside the text stream. The
  frontend renders only markers whose number exists in that list, and the
  Sources list shows only posts actually cited in the answer. Markers that
  don't match a source are dropped rather than shown broken. This is the
  most tested part of the codebase.
- **History and citation numbers.** The request carries text-only history
  (user and assistant messages; no tool calls or earlier sources). Before
  history reaches the model, `[n]` markers are stripped from earlier
  assistant messages. Numbers restart each turn, so a stale `[2]` can never
  be matched to a new turn's post #2. "Tell me more about [2]" across turns
  is not supported; the model re-searches instead.
- **Grounding is prompt-enforced only.** The system prompt requires a search
  for any finance question; no extra classifier or validation step. A test
  with a scripted model checks the tool is called for a finance question.
- **Pipeline panel behavior.** The search tool records every stage's ranked
  results and timings, and the backend streams that trace to the frontend as
  structured data *before* the answer text, so the panel fills in while the
  model is still thinking (this also covers the "silent pause" anti-goal).
  The panel shows the latest answer by default; clicking any earlier answer
  switches it to that answer's trace. If the agent searches more than once
  in a turn, each search gets its own section. Small-talk answers with no
  search show "no search for this message". On narrow screens the panel
  becomes a pull-out drawer. Traces live only in frontend state and are not
  sent back in the history.
- **System prompt rules:** answer only from retrieved posts; cite with `[n]`;
  if posts don't cover the question, say so plainly; it's forum content, not
  financial advice (one short line, not a lecture).
- **Empty state:** a short intro line and 3–4 clickable example questions so
  a recruiter isn't facing a blank box.
- **Errors:** if OpenAI/Cohere fails mid-stream, the UI shows a short inline
  error on that message with a retry button; the rest of the chat survives.
- **Tests:** pytest for retrieval wiring, citation numbering/filtering, and
  the streaming endpoint (with Pydantic AI's test/function models, no real
  API calls in CI). Light frontend tests for citation parsing.
- **README** with the architecture in a short diagram, how to run locally,
  and the live link.

## Left to the builder

- Wire format for streaming (SSE or Vercel AI SDK data-stream protocol, or a
  small custom NDJSON) — pick whatever carries text deltas plus a structured
  "sources" event cleanly.
- Exact OpenAI model (a current, cost-sensible model with good tool use).
- Number of candidates per retriever and final top-k after rerank (exp-python
  used 50 candidates → top 10).
- Whether to store embeddings as float16 to halve memory, if it doesn't hurt
  results noticeably.
- Repo layout (e.g. `backend/` + `frontend/`), Railway config details,
  health-check endpoint.
- Exact HF dataset name (under the owner's HF account).

## Delivery scope

The plan runs all the way to a working live URL.

- **Git:** repo initialized on `main` with remote
  `git@github.com:baxri/hybrid-search-rag.git`.
- **Claude does everything else:** HF dataset upload (owner runs
  `! hf auth login` when prompted; the HF username comes from
  `hf auth whoami`), Railway project setup and env vars, deploy, and a smoke
  test of the live URL. Every commit and push still waits for the owner's
  explicit approval.

## Repo notes

- Lives in `/Users/georgebibilashvili/Projects/test/python/hybrid-search-rag`.
  Specs in `docs/specs/`, plans in `docs/plans/`, matching the owner's
  convention in `exp-python`.
- `.superpowers/` (brainstorm mockups) goes in `.gitignore`.
