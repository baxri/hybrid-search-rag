"""The chat agent: one tool that runs the hybrid search and numbers what it found."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from pydantic_ai import Agent, RunContext, ToolReturn
from pydantic_ai.models import Model

from app.citations import SourceRegistry, format_for_model
from app.retrieval.pipeline import HybridPipeline
from app.schemas import SearchTrace, Source

INSTRUCTIONS = """\
You answer personal-finance questions using posts from the FiQA forum corpus.
- For any finance question, call search_finance_posts first. You may search again with a different query if the first results are weak. Rewrite follow-up questions into standalone search queries using the conversation.
- Search queries are short: the user's question in their own words, or a few key terms. Never add words about the source ("forum", "FiQA", "posts", "discussion", "opinion", "explanation") and never pad with lists of synonyms.
- Answer only from the returned posts. Cite every claim with the post's number in square brackets, e.g. [1] or [2][3]. Never cite a number you were not given in this turn.
- Never answer from your own knowledge. If a question isn't about personal finance, say you can only answer from the finance forum posts.
- If the posts don't cover the question, say so plainly instead of guessing.
- The posts are old (roughly 2008-2018). For current, latest or today's figures, say the posts can't give them; you may add what the posts said back then, labeled as historical. If the question names a year, law or event that no post names, say the posts don't cover it; don't present posts about something else as the answer.
- Say what each post says; don't draw conclusions beyond them or invent examples and numbers they don't contain. No uncited closing summaries.
- Don't offer follow-ups you can't do: all you can do is search these posts.
- Keep answers concise. These are forum opinions, not financial advice; say that in one short line only when giving a recommendation.
- For greetings or small talk, reply briefly without searching."""


@dataclass
class ChatDeps:
    """Per-turn state: a fresh registry (numbers restart each turn) and a search counter."""

    pipeline: HybridPipeline
    registry: SourceRegistry
    on_search: Callable[[int, str], Awaitable[None]]
    searches: int = 0
    question: str | None = None  # the user's own words, only when they stand alone (first turn)


@dataclass
class SearchMetadata:
    """Rides on the tool result for the stream to forward; never sent to the model."""

    search_id: int
    trace: SearchTrace
    sources: list[Source]  # newly numbered in this search


async def search_finance_posts(ctx: RunContext[ChatDeps], query: str) -> ToolReturn:
    """Search the FiQA personal-finance forum (BM25 + embeddings, fused and reranked).

    Args:
        query: A short standalone query in the user's words; rewrite follow-ups using the
            conversation. No words about the source and no synonym lists.
    """
    deps = ctx.deps
    deps.searches += 1
    search_id = deps.searches
    await deps.on_search(search_id, query)

    trace = await deps.pipeline.search(query, question=deps.question)
    this_search, newly_numbered = deps.registry.add(trace, deps.pipeline.corpus)

    # The model sees only the numbered post text; the trace and sources ride along in
    # metadata, which is never sent to the model, for the stream to forward to the UI.
    return ToolReturn(
        return_value=format_for_model(this_search),
        metadata=SearchMetadata(search_id=search_id, trace=trace, sources=newly_numbered),
    )


def build_agent(model: Model | str) -> Agent[ChatDeps, str]:
    return Agent(
        model,
        deps_type=ChatDeps,
        output_type=str,
        instructions=INSTRUCTIONS,
        tools=[search_finance_posts],
    )
