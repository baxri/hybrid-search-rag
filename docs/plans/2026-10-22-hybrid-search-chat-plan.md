# Hybrid Search Chat Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A public chat app over the FiQA corpus with streamed, cited answers and a live "under the hood" panel showing every retrieval stage, deployed on Railway.

**Architecture:** FastAPI backend hosts a Pydantic AI agent with one tool, `search_finance_posts`, which runs BM25 + dense → RRF → Cohere rerank and records a per-stage trace. `POST /api/chat` streams NDJSON events (search trace, sources, text deltas) to a Vite/React/Tailwind frontend, which renders citations (style A) and the stage-by-stage pipeline panel. One Docker image: React build served by FastAPI; index files pulled from a public Hugging Face dataset at build time.

**Tech Stack:** Python 3.12, uv, FastAPI, pydantic-ai-slim[openai] 2.x, pydantic-settings, bm25s, numpy, pandas/pyarrow, openai, cohere, huggingface_hub, pytest + pytest-asyncio, ruff · Node 22, Vite, React, TypeScript, Tailwind v4, Vitest · Docker, Railway.

**Spec:** `docs/specs/2026-10-22-hybrid-search-chat-design.md`

## Global Constraints

- Python >= 3.12; built-in generics (`list[str]`, `dict[str, int]`), never `typing.List`/`Dict`; ruff clean.
- Retrieval code is copied/adapted from `/Users/georgebibilashvili/Projects/test/exp-python/hybrid_retrieval/utils/`; never import from `exp-python`.
- No Cohere throttle (`MIN_SECONDS_BETWEEN_CALLS` is removed); paid key.
- Embedding model `text-embedding-3-small`; embeddings row order == `corpus.parquet` row order. Rerank model `rerank-v4.0-fast`.
- RRF `k=60`; `candidate_k=50` per retriever; `top_k=8` after rerank.
- LLM: `openai:gpt-5.4-mini` by default, overridable with env `CHAT_MODEL`.
- Secrets only from env: `OPENAI_API_KEY`, `COHERE_API_KEY`. Runtime needs no HF token.
- Indexes load once at startup; blocking calls (BM25, numpy, Cohere) run via `asyncio.to_thread`.
- Backend stores nothing between requests; frontend sends text-only history each request.
- Citation numbers restart each assistant turn; `[n]` markers are stripped from earlier assistant messages before they reach the model.
- No rate limiting, auth, or database.
- **Review cadence:** backend tasks (1–7, 11, 12) stop after each task for the owner's step-by-step review. Frontend tasks (8–10) are implemented in one go, then reviewed together.
- **Every `git commit` / `git push` step: ask the owner for explicit approval first.** Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. Model emits a marker with no matching source (`[9]` when 8 sources, or `[3]` with no search) → marker silently dropped, no broken badge. (Task 8 test)
2. Grouped/adjacent markers `[1][2]`, `[1, 2]`, `[1,2]` → each number rendered as its own badge. (Task 8 test)
3. A blank FiQA doc (38 exist, embedded as `"[empty document]"`) ranks high → never reaches rerank, sources, or the panel. (Task 3 test)
4. OpenAI or Cohere raises mid-stream → stream emits one `error` event and closes; earlier deltas kept; UI offers retry. (Task 6 test, Task 9 test)
5. Same doc returned by two searches in one turn → keeps its first number; not listed twice in Sources. (Task 4 test)

---

## File Structure

```
backend/
  pyproject.toml                 uv project, package "app"
  app/
    config.py                    Settings
    schemas.py                   all Pydantic API + trace models
    retrieval/
      corpus.py                  Corpus (texts by id, blank-doc set)
      bm25.py                    BM25Retriever
      dense.py                   DenseRetriever
      fusion.py                  reciprocal_rank_fusion
      rerank.py                  CohereReranker
      pipeline.py                HybridPipeline -> SearchTrace
    citations.py                 SourceRegistry, strip_markers
    history.py                   to_model_history
    agent.py                     build_agent, ChatDeps, search tool, INSTRUCTIONS
    stream.py                    chat_events -> NDJSON lines
    main.py                      FastAPI app, lifespan, routes, static files
  scripts/upload_indexes.py
  scripts/download_indexes.py
  tests/...
frontend/
  src/lib/stream.ts              readNdjson
  src/lib/citations.ts           splitCitations, citedNumbers
  src/lib/types.ts               TS mirrors of backend events
  src/hooks/useChat.ts           conversation state + send/retry
  src/components/{ChatPane,Message,SourcesList,EmptyState,Composer}.tsx
  src/components/{PipelinePanel,StageCard}.tsx
  src/App.tsx
Dockerfile
railway.json
README.md
```

## Wire format (shared by Tasks 6, 8–10)

`POST /api/chat` body: `{"messages": [{"role": "user"|"assistant", "content": str}, ...]}`. Response `application/x-ndjson`, one JSON object per line:

| `type` | fields | when |
|---|---|---|
| `search_started` | `search_id: int`, `query: str` | tool call begins |
| `search_result` | `search_id: int`, `trace: SearchTrace`, `sources: Source[]` | tool finished; `sources` = newly numbered sources from this search |
| `text_delta` | `delta: str` | every text chunk |
| `error` | `message: str` | any exception; then stream ends |
| `done` | — | normal end |

---

### Task 1: Backend scaffold, settings, health

**Files:**
- Create: `backend/pyproject.toml`, `backend/app/__init__.py`, `backend/app/config.py`, `backend/app/main.py`, `backend/tests/conftest.py`, `backend/tests/test_health.py`, `.gitignore` (append `backend/data/`, `node_modules/`, `frontend/dist/`, `.venv/`, `__pycache__/`)

**Interfaces:**
- Produces: `Settings(BaseSettings)` with `openai_api_key: str`, `cohere_api_key: str`, `chat_model: str = "openai:gpt-5.4-mini"`, `data_dir: Path = Path(__file__).parents[1] / "data"` (layout: `corpus.parquet`, `embeddings.npy`, `bm25/`), `candidate_k: int = 50`, `top_k: int = 8`, `rrf_k: int = 60`, `hf_repo: str = ""`; `get_settings() -> Settings` (lru_cache). `create_app(pipeline: HybridPipeline | None = None) -> FastAPI` (pipeline injection used in tests; real one built in lifespan when None). `GET /api/health -> {"status": "ok"}`.

- [ ] **Step 1:** `uv init --package` in `backend/`, add deps listed in Tech Stack plus dev deps `pytest`, `pytest-asyncio`, `httpx`, `ruff`. Set `[tool.pytest.ini_options] asyncio_mode = "auto"`.
- [ ] **Step 2: Failing test** `test_health_ok`: `TestClient(create_app(pipeline=FakePipeline())).get("/api/health")` → 200, `{"status": "ok"}`. Put a minimal `FakePipeline` placeholder in `conftest.py` (filled in Task 3).
- [ ] **Step 3:** Run `cd backend && uv run pytest tests/test_health.py -v` → FAIL (import error).
- [ ] **Step 4:** Implement `config.py` and `main.py` (lifespan stores pipeline on `app.state.pipeline`).
- [ ] **Step 5:** Run → PASS. `uv run ruff check .` → clean.
- [ ] **Step 6:** Ask owner approval, then commit `Scaffold FastAPI backend with settings and health check`.

### Task 2: Port retrievers, fusion, reranker

**Files:**
- Create: `backend/app/retrieval/{__init__,corpus,bm25,dense,fusion,rerank}.py`, `backend/tests/retrieval/test_{corpus,bm25,dense,fusion,rerank}.py`, `backend/tests/fixtures.py` (tiny 6-doc corpus builder: writes `corpus.parquet`, BM25 index via `bm25s`, random-but-seeded `embeddings.npy` into `tmp_path`)

**Interfaces:**
- `Corpus.load(data_dir: Path) -> Corpus`; `.text(doc_id: str) -> str`; `.is_blank(doc_id: str) -> bool` (text strips to `""`); `.ids: list[str]` (row order).
- `BM25Retriever(index_dir: Path)`; `.search(query: str, k: int) -> list[tuple[str, float]]`.
- `DenseRetriever(embeddings_path: Path, doc_ids: list[str], embed: Callable[[str], Awaitable[np.ndarray]])`; `async .search(query: str, k: int) -> list[tuple[str, float]]`. Real `embed` = `AsyncOpenAI().embeddings.create(model="text-embedding-3-small", ...)`, wired in Task 3.
- `reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]` (port verbatim).
- `CohereReranker(client: cohere.ClientV2, model: str = "rerank-v4.0-fast")`; `.rerank(query: str, docs: list[tuple[str, str]], top_n: int) -> list[tuple[str, float]]` (docs are `(doc_id, text)`; retries `TooManyRequestsError` 3× with 2s backoff, no fixed throttle).

- [ ] **Step 1: Failing tests:**
  - `test_rrf_orders_by_summed_reciprocal_rank`: `reciprocal_rank_fusion([["a","b"],["b","c"]])[0][0] == "b"`; score of `"b"` == `1/62 + 1/61`.
  - `test_bm25_finds_exact_term`: fixture doc containing "1099-K" is rank 1 for query `"1099-K form"`.
  - `test_dense_returns_cosine_top_k`: with `embed` returning the fixture vector of doc `"d3"`, top-1 is `"d3"`, score ≈ 1.0, `len == k`.
  - `test_corpus_blank_detection`: fixture doc with `"   "` text → `is_blank` True.
  - `test_rerank_maps_indices_to_ids`: fake client returning results `[{index:1, relevance_score:0.9},{index:0,...0.2}]` → `[("id1",0.9),("id0",0.2)]`.
- [ ] **Step 2:** Run `uv run pytest tests/retrieval -v` → FAIL.
- [ ] **Step 3:** Implement the five modules, porting from `exp-python/hybrid_retrieval/utils/`. Dense pre-normalizes rows once at init.
- [ ] **Step 4:** Run → PASS; ruff clean.
- [ ] **Step 5:** Ask owner approval, then commit `Port BM25, dense, RRF and Cohere rerank retrievers`.

### Task 3: Hybrid pipeline with per-stage trace

**Files:**
- Create: `backend/app/schemas.py`, `backend/app/retrieval/pipeline.py`, `backend/tests/retrieval/test_pipeline.py`; Modify: `backend/tests/conftest.py` (`FakePipeline` returns a canned `SearchTrace`), `backend/app/main.py` (lifespan builds real pipeline via `HybridPipeline.from_settings`)

**Interfaces:**
- `schemas.py`:
  - `RankedDoc(doc_id: str, rank: int, score: float)` (rank 1-based)
  - `StageName = Literal["bm25", "dense", "rrf", "rerank"]`
  - `Stage(name: StageName, ms: float, results: list[RankedDoc])`
  - `SearchTrace(query: str, total_ms: float, stages: list[Stage])` — stages always in order bm25, dense, rrf, rerank; bm25/dense/rrf hold up to `candidate_k`, rerank holds `top_k`.
  - `Source(n: int, doc_id: str, text: str)`
  - `ChatMessage(role: Literal["user","assistant"], content: str)`, `ChatRequest(messages: list[ChatMessage])` — validator: non-empty list, last role `"user"`, last content stripped non-empty and ≤ 4000 chars.
- `HybridPipeline(corpus, bm25, dense, reranker, candidate_k, top_k, rrf_k)`; `async .search(query: str) -> SearchTrace`; `HybridPipeline.from_settings(s: Settings) -> HybridPipeline`; `.corpus` exposed.

- [ ] **Step 1: Failing tests** (fixture retrievers + fake reranker that reverses input order):
  - `test_trace_has_four_stages_in_order`: `[s.name for s in trace.stages] == ["bm25","dense","rrf","rerank"]`, each `ms >= 0`, `len(rerank.results) <= top_k`.
  - `test_blank_docs_never_reach_rerank`: blank fixture doc forced to rank 1 in fake BM25 → absent from `rrf` and `rerank` stage results (still visible in `bm25` stage is fine).
  - `test_bm25_and_dense_run_concurrently`: both fakes sleep 0.2s → `total_ms < 350`.
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement: `asyncio.gather(to_thread(bm25.search), dense.search)`, RRF over id lists, drop blank ids, take `candidate_k`, rerank via `to_thread` with corpus texts, time each stage with `time.perf_counter`.
- [ ] **Step 4:** Run → PASS.
- [ ] **Step 5:** Ask owner approval, then commit `Add hybrid pipeline that records per-stage search trace`.

### Task 4: Citation registry, marker stripping, history conversion

**Files:**
- Create: `backend/app/citations.py`, `backend/app/history.py`, `backend/tests/test_citations.py`, `backend/tests/test_history.py`

**Interfaces:**
- `SourceRegistry()`; `.add(trace: SearchTrace, corpus: Corpus) -> tuple[list[Source], list[Source]]` — returns `(this_search, newly_numbered)`. Numbers rerank results in order, continuing from the turn's last number; a doc already registered keeps its number, appears in `this_search` but not in `newly_numbered`. `.all: list[Source]`.
- `format_for_model(sources: list[Source]) -> str` — `"[n] {text}"` blocks separated by blank lines, each text truncated to 2000 chars; returns `"No relevant posts found."` when empty.
- `strip_markers(text: str) -> str` — removes `[n]`, `[n, m]`, `[n,m]` groups and collapses the double space left behind.
- `to_model_history(messages: list[ChatMessage]) -> tuple[list[ModelMessage], str]` — returns (history for all but last message, last user prompt). user → `ModelRequest(parts=[UserPromptPart(content)])`; assistant → `ModelResponse(parts=[TextPart(strip_markers(content))])`. Keeps only the last 20 messages before the prompt.

- [ ] **Step 1: Failing tests:**
  - `test_registry_numbers_continue_across_searches`: first add (docs a,b) → n 1,2; second add (docs c) → n 3.
  - `test_registry_dedupes_doc_across_searches`: second add (b, d) → `this_search` n == [2, 3], `newly_numbered` == [d as 3]; `all` has 3 sources.
  - `test_strip_markers`: `"Use CDs [1][2], or HYSA [3, 4]."` → `"Use CDs, or HYSA."`
  - `test_history_strips_old_markers_and_splits_prompt`: 3 messages → history has 2 items, `TextPart.content` has no `[1]`, prompt equals last content.
  - `test_history_caps_at_20`.
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** Run → PASS.
- [ ] **Step 5:** Ask owner approval, then commit `Add per-turn citation registry and history conversion`.

### Task 5: Pydantic AI agent and search tool

**Files:**
- Create: `backend/app/agent.py`, `backend/tests/test_agent.py`

**Interfaces:**
- `ChatDeps(pipeline: HybridPipeline, registry: SourceRegistry, on_search: Callable[[int, str], Awaitable[None]])` dataclass; `on_search(search_id, query)` lets the stream emit `search_started`.
- `build_agent(model: Model | str) -> Agent[ChatDeps, str]`.
- Tool `search_finance_posts(ctx: RunContext[ChatDeps], query: str) -> ToolReturn` — calls `on_search`, runs pipeline, registers sources, returns `ToolReturn(return_value=format_for_model(this_search), metadata={"search_id": id, "trace": trace.model_dump(), "sources": [s.model_dump() for s in newly_numbered]})`. Search ids count from 1 per turn (store counter on deps).
- `INSTRUCTIONS` (exact copy):

```
You answer personal-finance questions using posts from the FiQA forum corpus.
- For any finance question, call search_finance_posts first. You may search again with a different query if the first results are weak. Rewrite follow-up questions into standalone search queries using the conversation.
- Answer only from the returned posts. Cite every claim with the post's number in square brackets, e.g. [1] or [2][3]. Never cite a number you were not given in this turn.
- If the posts don't cover the question, say so plainly instead of guessing.
- Keep answers concise. These are forum opinions, not financial advice; say that in one short line only when giving a recommendation.
- For greetings or small talk, reply briefly without searching.
```

- [ ] **Step 1: Failing tests** with `FunctionModel` (stream function scripts a tool call, then text) and `FakePipeline`:
  - `test_finance_question_calls_search_tool` (`FakePipeline` records queries in `.calls`): run with prompt "where to keep emergency fund?" → `FakePipeline.calls == ["emergency fund savings"]` (the scripted query), tool return text starts with `"[1] "`.
  - `test_tool_metadata_carries_trace_and_sources`: `FunctionToolResultEvent.part.metadata["sources"][0]["n"] == 1` and `metadata["trace"]["stages"]` length 4.
  - `test_two_searches_get_ids_1_and_2`.
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** Run → PASS.
- [ ] **Step 5:** Ask owner approval, then commit `Add Pydantic AI agent with hybrid search tool`.

### Task 6: Streaming chat endpoint

**Files:**
- Create: `backend/app/stream.py`, `backend/tests/test_chat_stream.py`; Modify: `backend/app/main.py`

**Interfaces:**
- `async chat_events(req: ChatRequest, agent: Agent[ChatDeps, str], pipeline: HybridPipeline) -> AsyncIterator[str]` — yields NDJSON lines per the wire-format table. Uses `agent.run_stream_events(prompt, message_history=..., deps=...)` as an async context manager. Map: `on_search` → `search_started`; `FunctionToolResultEvent` → `search_result` from `part.metadata`; `PartStartEvent` with `TextPart` → `text_delta(part.content)`; `PartDeltaEvent` with `TextPartDelta` → `text_delta(content_delta)`. `search_started` events are pushed through an `asyncio.Queue` merged with the agent event stream. Any exception → log it, yield `error` with message `"Something went wrong while answering. Please retry."`, stop. Normal end → `done`.
- `POST /api/chat` → `StreamingResponse(chat_events(...), media_type="application/x-ndjson")`; 422 on invalid `ChatRequest`. Agent built once in lifespan (`app.state.agent`); `create_app(pipeline=..., model=...)` accepts a model override for tests.

- [ ] **Step 1: Failing tests** (TestClient, FunctionModel, FakePipeline):
  - `test_stream_event_order`: types == `["search_started","search_result","text_delta",...,"done"]`; joined deltas == scripted answer.
  - `test_search_result_carries_sources_and_trace`.
  - `test_reranker_failure_emits_error_and_closes`: FakePipeline raises → last event `{"type":"error",...}`, no `done`.
  - `test_empty_message_rejected`: `{"messages":[{"role":"user","content":"  "}]}` → 422.
  - `test_last_message_must_be_user` → 422.
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** Run → PASS.
- [ ] **Step 5: Manual real-API check** (needs local indexes copied from `exp-python/hybrid_retrieval/{data/fiqa,indexes}` into `backend/data/` with layout `data/corpus.parquet`, `data/bm25/`, `data/embeddings.npy`, and `.env` with both keys): `uv run uvicorn app.main:app` then `curl -N -X POST localhost:8000/api/chat -H 'content-type: application/json' -d '{"messages":[{"role":"user","content":"Where should I keep my emergency fund?"}]}'` → `search_started` within ~100ms, `search_result`, streamed deltas containing `[1]`, `done`.
- [ ] **Step 6:** Ask owner approval, then commit `Stream chat answers, search traces and sources as NDJSON`.

### Task 7: Host indexes on Hugging Face

**Files:**
- Create: `backend/scripts/upload_indexes.py`, `backend/scripts/download_indexes.py`, `backend/tests/test_download_layout.py`

**Interfaces:**
- Dataset repo: `{hf_username}/fiqa-hybrid-indexes` (public, `repo_type="dataset"`), files: `corpus.parquet`, `embeddings.npy`, `bm25/*`. Upload uses `HfApi().create_repo(..., exist_ok=True)` + `upload_folder`. Includes a dataset card noting FiQA source (BEIR) and its license.
- `download_indexes(repo: str, dest: Path) -> None` via `snapshot_download(repo_id=repo, repo_type="dataset", local_dir=dest)`; CLI `python scripts/download_indexes.py --repo X --dest data`.

- [ ] **Step 1: Failing test** `test_layout_matches_settings`: after a fake download (monkeypatched `snapshot_download` that writes the fixture files), `HybridPipeline.from_settings` paths resolve (`Corpus.load` and `BM25Retriever` succeed).
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** Run → PASS.
- [ ] **Step 5:** Ask the owner to run `! hf auth login` (write token). Read username with `hf auth whoami`. Run upload script. Verify: `download_indexes` into a temp dir → files and sizes match local (`embeddings.npy` ≈ 338 MB).
- [ ] **Step 6:** Set `hf_repo` default in `Settings`. Ask owner approval, then commit `Add scripts to publish and fetch indexes from Hugging Face`.

### Task 8: Frontend scaffold, stream reader, citation parsing

**Files:**
- Create: `frontend/` via `npm create vite@latest frontend -- --template react-ts`; add Tailwind v4 (`@tailwindcss/vite`), Vitest. Create `src/lib/types.ts`, `src/lib/stream.ts`, `src/lib/citations.ts`, `src/lib/*.test.ts`. Vite dev proxy `/api → http://localhost:8000`.

**Interfaces:**
- `types.ts`: TS mirrors of `RankedDoc`, `Stage`, `SearchTrace`, `Source`, `ChatEvent` (discriminated union on `type`, matching the wire-format table).
- `readNdjson(res: Response): AsyncGenerator<ChatEvent>` — buffers partial lines across chunks.
- `splitCitations(text: string, valid: Set<number>): Array<{kind: "text", text: string} | {kind: "cite", n: number}>` — matches `\[(\d+(?:\s*,\s*\d+)*)\]`; each number becomes its own `cite`; numbers not in `valid` are dropped (the bracket group vanishes if none are valid).
- `citedNumbers(text: string, valid: Set<number>): number[]` — unique, in first-appearance order.

- [ ] **Step 1: Failing tests:**
  - `readNdjson` splits a line delivered across two chunks into one event.
  - `splitCitations("A [1][2] B [1, 3]", new Set([1,2,3]))` → cites 1,2,1,3 in order.
  - `splitCitations("A [9].", new Set([1]))` → `[{text:"A "},{text:"."}]` (no cite).
  - `citedNumbers("x [2] y [1] z [2]", new Set([1,2]))` → `[2,1]`.
- [ ] **Step 2:** `npm test` → FAIL. **Step 3:** Implement. **Step 4:** `npm test` → PASS; `npm run build` succeeds.
- [ ] **Step 5:** Ask owner approval, then commit `Scaffold React frontend with NDJSON reader and citation parsing`.

### Task 9: Chat UI with citations, sources, errors

**Files:**
- Create: `src/hooks/useChat.ts`, `src/components/{ChatPane,Message,SourcesList,EmptyState,Composer}.tsx`, `src/hooks/useChat.test.ts`; Modify: `src/App.tsx`

**Interfaces:**
- `AssistantTurn = { id: string; text: string; searches: {id: number; query: string; trace?: SearchTrace}[]; sources: Source[]; status: "streaming" | "done" | "error" }`; `UserTurn = { id: string; text: string }`.
- `useChat() -> { turns, send(text: string), retry(turnId: string), selectedTurnId, select(turnId: string) }`. `send` posts all prior turns as `{role, content}` (assistant `content` = raw text) plus the new message; new assistant turn becomes `selectedTurnId`. `retry` drops the failed assistant turn and resends the same history.
- Citation style A: `Message` renders `splitCitations` output with cites as small blue superscript badges; under a finished answer, `SourcesList` shows only `citedNumbers` sources: `[n]` + first ~140 chars + `doc {doc_id}`, click to expand full text. Clicking a badge scrolls to/expands its source.
- `EmptyState`: one intro line and 4 clickable examples: "Where should I keep my emergency fund?", "Is it worth paying off my mortgage early?", "How are stock options taxed when I exercise them?", "Should I max out my 401(k) before investing elsewhere?"
- Error: assistant turn with `status: "error"` shows partial text, inline message from the `error` event (or "Connection lost." on fetch failure), and a Retry button.
- Layout: comfortable, centered chat column, composer pinned bottom, Enter sends / Shift+Enter newline, disabled while streaming.

- [ ] **Step 1: Failing tests** (`useChat` with mocked `fetch` returning a scripted NDJSON stream):
  - deltas accumulate into `text`; `search_result` attaches trace to the matching `search_id`; status ends `"done"`.
  - `error` event → status `"error"`, partial text kept; `retry` resends history without the failed turn.
  - request body for the second turn contains both earlier messages.
- [ ] **Step 2:** FAIL → **Step 3:** Implement → **Step 4:** PASS.
- [ ] **Step 5: Manual check** with backend running and `npm run dev`: ask an example question, then "what about CDs?"; badges match sources, follow-up gets a rewritten query.
- [ ] **Step 6:** Ask owner approval, then commit `Add chat UI with streamed answers and citation sources`.

### Task 10: Pipeline panel (stage-by-stage)

**Files:**
- Create: `src/components/PipelinePanel.tsx`, `src/components/StageCard.tsx`, `src/lib/rankJumps.ts`, `src/lib/rankJumps.test.ts`; Modify: `src/App.tsx`

**Interfaces:**
- `rankJump(trace: SearchTrace, docId: string): {from: "bm25" | "dense", delta: number} | null` — for each of bm25/dense where the doc appears, climb = that rank − rerank rank; returns the retriever with the larger climb, or null if that climb < 5 or the doc is in neither. ("▲13 vs BM25".)
- `PipelinePanel({turn}: {turn?: AssistantTurn})`: left column (≈40% width on ≥lg; pull-out drawer with toggle button below lg). Per search: header with tool call `search_finance_posts("<query>")` and `total_ms`; then four `StageCard`s (BM25, Dense, RRF fusion `k=60`, Cohere rerank `top 8`), each showing count, ms, top 5 rows `rank · doc id · score`, "show all" expander. Rows whose doc is cited in the turn are highlighted with their `[n]`; rerank rows show `rankJump` badges. While searching (`search_started`, no trace yet): skeleton cards. Turn with no searches: "No search for this message." No turn yet: one-line explanation of what the panel will show.

- [ ] **Step 1: Failing tests** for `rankJump`: doc rank 14 in BM25, 1 in dense, 1 rerank → `{from:"bm25", delta:13}`; doc ranks 2/3/1 → null; doc absent from bm25 (dense rank 9, rerank 1) → `{from:"dense", delta:8}`.
- [ ] **Step 2:** FAIL → **Step 3:** Implement → **Step 4:** PASS.
- [ ] **Step 5: Manual check:** panel fills before text starts; clicking an older answer switches trace; narrow window shows drawer.
- [ ] **Step 6:** Ask owner approval, then commit `Add stage-by-stage pipeline panel`.

### Task 11: Single image and local production run

**Files:**
- Create: `Dockerfile`, `.dockerignore`; Modify: `backend/app/main.py` (mount `frontend/dist` at `/` with SPA fallback when the directory exists; `/api/*` unaffected)

**Interfaces:**
- Dockerfile stages: (1) `node:22-slim` → `npm ci && npm run build`; (2) `python:3.12-slim` + uv → `uv sync --frozen --no-dev`, `ARG HF_REPO`, run `download_indexes.py --repo $HF_REPO --dest /app/backend/data`, copy `dist` → `/app/frontend/dist`; `CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}`.

- [ ] **Step 1: Failing test** `test_spa_fallback`: with a temp dist containing `index.html`, `GET /some/route` → 200 html; `GET /api/health` still JSON.
- [ ] **Step 2:** FAIL → **Step 3:** Implement → **Step 4:** PASS.
- [ ] **Step 5:** `docker build --build-arg HF_REPO=<user>/fiqa-hybrid-indexes -t hybrid-chat .` then `docker run -p 8000:8000 --env-file backend/.env hybrid-chat`; open `localhost:8000`, run one chat. Record `docker stats` memory peak (to size Railway).
- [ ] **Step 6:** Ask owner approval, then commit `Build single Docker image serving API and React app`.

### Task 12: Railway deploy and README

**Files:**
- Create: `railway.json` (Dockerfile builder, healthcheck `/api/health`, build arg `HF_REPO`), `README.md`

**Interfaces:**
- README: one-paragraph pitch, live link, screenshot/GIF placeholder replaced after deploy, short architecture diagram (browser → FastAPI → agent → BM25/dense → RRF → rerank → stream), local run instructions, how indexes are hosted, tests command.

- [ ] **Step 1:** Install Railway CLI (`npm i -g @railway/cli` or brew); ask owner to run `! railway login`.
- [ ] **Step 2:** Ask owner approval, then push `main` to `origin`.
- [ ] **Step 3:** `railway init` (project `hybrid-search-rag`), link GitHub repo or `railway up`; set variables `OPENAI_API_KEY`, `COHERE_API_KEY` (owner pastes them via `! railway variables --set ...` so keys never appear in chat), `HF_REPO`; memory ≥ the peak recorded in Task 11 + 30%; generate public domain.
- [ ] **Step 4: Live smoke:** `curl https://<domain>/api/health` → ok; streaming curl from Task 6 against the domain → events through `done`; open in browser, ask one question and one follow-up, check badges, sources and panel.
- [ ] **Step 5:** Put live link in README. Ask owner approval, then commit `Add Railway config and README` and push.
