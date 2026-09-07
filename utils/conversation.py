"""Shared helper for building assistant prompts with conversation
history and location context - used by both the voice and text
chat endpoints so a fix to one applies to both."""

from typing import Any


def build_question_with_context(
    text: str,
    history: list[dict[str, Any]] | None,
    location: str | None,
) -> str:
    """`history` is an already-parsed list of {"sender", "text"}
    dicts (most recent last). Only the last 8 turns are used to
    keep the prompt compact."""

    question = text

    if history:
        history_lines = []
        for item in history[-8:]:
            if not isinstance(item, dict):
                continue
            sender = item.get("sender")
            message = str(item.get("text", "")).strip()
            if sender in {"user", "assistant"} and message:
                history_lines.append(f"{sender}: {message}")
        if history_lines:
            question = (
                "Conversation context:\n"
                + "\n".join(history_lines)
                + "\n\nNew farmer question:\n"
                + text
            )

    if location:
        question = (
            f"{question}\n\n"
            f"Farmer's current location: {location}"
        )

    return question
