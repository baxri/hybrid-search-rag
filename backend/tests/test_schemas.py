import pytest
from pydantic import ValidationError

from app.schemas import ChatRequest


def request(*messages: tuple[str, str]) -> dict:
    return {"messages": [{"role": role, "content": content} for role, content in messages]}


def test_valid_conversation_accepted():
    req = ChatRequest.model_validate(request(("user", "hi"), ("assistant", "hello"), ("user", "q")))

    assert len(req.messages) == 3


@pytest.mark.parametrize(
    "payload",
    [
        request(),
        request(("user", "   ")),
        request(("user", "hi"), ("assistant", "hello")),
        request(("user", "x" * 4001)),
    ],
    ids=["empty", "blank-last", "last-not-user", "too-long"],
)
def test_invalid_conversation_rejected(payload):
    with pytest.raises(ValidationError):
        ChatRequest.model_validate(payload)
