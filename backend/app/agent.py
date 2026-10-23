"""The chat agent: one tool that runs the hybrid search and numbers what it found."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from pydantic_ai import Agent, RunContext, ToolReturn
from pydantic_ai.models import Model

from app.citations import SourceRegistry, format_for_model
from app.retrieval.pipeline import HybridPipeline

INSTRUCTIONS = """\
You answer personal-finance questions using posts from the FiQA forum corpus.
- For any finance question, call search_finance_posts first. You may search again with a different query if the first results are weak. Rewrite follow-up questions into standalone search queries using the conversation.
- Answer only from the returned posts. Cite every claim with the post's number in square brackets, e.g. [1] or [2][3]. Never cite a number you were not given in this turn.
- If the posts don't cover the question, say so plainly instead of guessing.
- Keep answers concise. These are forum opinions, not financial advice; say that in one short line only when giving a recommendation.
- For greetings or small talk, reply briefly without searching."""


@dataclass
class ChatDeps:
    """Per-turn state: a fresh registry (numbers restart each turn) and a search counter."""

    pipeline: HybridPipeline
    registry: SourceRegistry
    on_search: Callable[[int, str], Awaitable[None]]
    searches: int = 0


async def search_finance_posts(ctx: RunContext[ChatDeps], query: str) -> ToolReturn:
    """Search the FiQA personal-finance forum (BM25 + embeddings, fused and reranked).

    Args:
        query: A standalone search query; rewrite follow-ups using the conversation.
    """
    deps = ctx.deps
    deps.searches += 1
    search_id = deps.searches
    await deps.on_search(search_id, query)

    trace = await deps.pipeline.search(query)
    this_search, newly_numbered = deps.registry.add(trace, deps.pipeline.corpus)

    # The model sees only the numbered post text; the trace and sources ride along in
    # metadata, which is never sent to the model, for the stream to forward to the UI.
    return ToolReturn(
        return_value=format_for_model(this_search),
        metadata={
            "search_id": search_id,
            "trace": trace.model_dump(),
            "sources": [s.model_dump() for s in newly_numbered],
        },
    )


def build_agent(model: Model | str) -> Agent[ChatDeps, str]:
    return Agent(
        model,
        deps_type=ChatDeps,
        output_type=str,
        instructions=INSTRUCTIONS,
        tools=[search_finance_posts],
    )
