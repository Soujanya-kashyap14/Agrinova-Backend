"""Text-based assistant chat models.

This reuses the exact same Gemini-backed reasoning
(services/assistant_service.py) as the voice endpoint, just
without the speech-to-text step - useful when a farmer prefers
typing, or when their microphone/audio conditions make voice
recognition unreliable.
"""

from typing import Literal

from pydantic import BaseModel


class ConversationMessage(BaseModel):
    sender: Literal["user", "assistant"]

    text: str


class TextChatRequest(BaseModel):
    question: str

    language: str | None = None

    latitude: str | None = None

    longitude: str | None = None

    location: str | None = None

    conversation: list[ConversationMessage] | None = None
