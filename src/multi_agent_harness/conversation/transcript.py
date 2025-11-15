"""Transcript models for tracking conversation history."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ToolInvocationRecord:
    """Record of a tool call execution with result or error."""

    tool_name: str
    arguments: dict[str, Any]
    result: Any | None = None
    error: str | None = None


@dataclass(slots=True)
class ConversationTurn:
    """A single turn in a conversation with optional tool invocations."""

    role: str
    message: str
    tool_invocations: list[ToolInvocationRecord] = field(default_factory=list)


@dataclass(slots=True)
class ConversationTranscript:
    """A complete conversation history."""

    turns: list[ConversationTurn] = field(default_factory=list)

    def add_turn(self, turn: ConversationTurn) -> None:
        """Add a turn to the transcript."""
        self.turns.append(turn)


__all__ = [
    "ConversationTranscript",
    "ConversationTurn",
    "ToolInvocationRecord",
]
