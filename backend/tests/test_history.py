from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

from app.history import to_model_history
from app.schemas import ChatMessage


def msg(role: str, content: str) -> ChatMessage:
    return ChatMessage(role=role, content=content)


def test_history_strips_old_markers_and_splits_prompt():
    history, prompt = to_model_history(
        [
            msg("user", "Where to keep an emergency fund?"),
            msg("assistant", "A high-yield savings account [1]."),
            msg("user", "What about CDs?"),
        ]
    )

    assert prompt == "What about CDs?"
    assert len(history) == 2
    assert isinstance(history[0], ModelRequest)
    assert isinstance(history[0].parts[0], UserPromptPart)
    assert history[0].parts[0].content == "Where to keep an emergency fund?"
    assert isinstance(history[1], ModelResponse)
    assert isinstance(history[1].parts[0], TextPart)
    assert history[1].parts[0].content == "A high-yield savings account."


def test_history_caps_at_20():
    earlier = [msg("user" if i % 2 == 0 else "assistant", f"m{i}") for i in range(30)]

    history, prompt = to_model_history([*earlier, msg("user", "last")])

    assert prompt == "last"
    assert len(history) == 20
    assert history[0].parts[0].content == "m10"
    assert history[-1].parts[0].content == "m29"
