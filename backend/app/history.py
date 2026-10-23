"""Turn the frontend's text-only conversation into Pydantic AI message history."""

from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, UserPromptPart

from app.citations import strip_markers
from app.schemas import ChatMessage

MAX_HISTORY_MESSAGES = 20


def to_model_history(messages: list[ChatMessage]) -> tuple[list[ModelMessage], str]:
    """Return (history, prompt): every message but the last as history, the last as the prompt.

    Earlier answers lose their [n] markers: numbers restart each turn, so an old [2] must never
    reach the model, where it could be matched against this turn's post #2.
    """
    *earlier, last = messages
    history: list[ModelMessage] = []
    for message in earlier[-MAX_HISTORY_MESSAGES:]:
        if message.role == "user":
            history.append(ModelRequest(parts=[UserPromptPart(content=message.content)]))
        else:
            history.append(ModelResponse(parts=[TextPart(content=strip_markers(message.content))]))
    return history, last.content
