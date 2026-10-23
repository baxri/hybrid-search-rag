"""Turn one agent run into NDJSON lines for the browser, one `ChatEvent` (see schemas) per line.

Pydantic AI's own events (pydantic_ai.messages.AgentStreamEvent) are large library internals;
we map the few we need onto our five small, stable events.
"""

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.messages import (
    FunctionToolResultEvent,
    PartDeltaEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
)

from app.agent import ChatDeps, SearchMetadata
from app.citations import SourceRegistry
from app.history import to_model_history
from app.retrieval.pipeline import HybridPipeline
from app.schemas import (
    ChatRequest,
    Done,
    ErrorEvent,
    SearchResult,
    SearchStarted,
    TextDelta,
)

logger = logging.getLogger(__name__)

ERROR_MESSAGE = "Something went wrong while answering. Please retry."

WireEvent = SearchStarted | SearchResult | TextDelta | ErrorEvent | Done


def _to_wire(event: Any) -> WireEvent | None:
    """Map a Pydantic AI event to a wire event; None for events the browser doesn't need."""
    if isinstance(event, PartStartEvent) and isinstance(event.part, TextPart):
        if event.part.content:
            return TextDelta(delta=event.part.content)
    elif isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
        if event.delta.content_delta:
            return TextDelta(delta=event.delta.content_delta)
    elif isinstance(event, FunctionToolResultEvent) and isinstance(
        event.part.metadata, SearchMetadata
    ):
        metadata = event.part.metadata
        return SearchResult(
            search_id=metadata.search_id, trace=metadata.trace, sources=metadata.sources
        )
    return None


async def chat_events(
    req: ChatRequest, agent: Agent[ChatDeps, str], pipeline: HybridPipeline
) -> AsyncIterator[str]:
    # One queue carries both agent events and search_started (sent from inside the tool), so
    # the browser sees them in the order they happened.
    queue: asyncio.Queue[WireEvent | None] = asyncio.Queue()

    async def on_search(search_id: int, query: str) -> None:
        await queue.put(SearchStarted(search_id=search_id, query=query))

    history, prompt = to_model_history(req.messages)
    deps = ChatDeps(pipeline=pipeline, registry=SourceRegistry(), on_search=on_search)

    async def run_agent() -> None:
        try:
            async with agent.run_stream_events(
                prompt, message_history=history, deps=deps
            ) as stream:
                async for event in stream:
                    if (wire := _to_wire(event)) is not None:
                        await queue.put(wire)
            await queue.put(Done())
        except Exception:
            logger.exception("chat run failed")
            await queue.put(ErrorEvent(message=ERROR_MESSAGE))
        finally:
            await queue.put(None)

    task = asyncio.create_task(run_agent())
    try:
        while (event := await queue.get()) is not None:
            yield event.model_dump_json() + "\n"
    finally:
        # Client went away mid-answer: stop spending OpenAI/Cohere calls on it.
        task.cancel()
