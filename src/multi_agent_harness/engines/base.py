"""Shared engine scaffolding for the multi-agent harness.

Note: Transcript models have been moved to conversation/transcript.py.
This module is maintained for backward compatibility.
"""

from __future__ import annotations

from abc import ABC
from typing import List

from ..adapters.base import ChatMessage, ProviderAdapter
from ..config import RoleModelConfig
from ..conversation.transcript import (
    ConversationTranscript,
    ConversationTurn,
    ToolInvocationRecord,
)


class RoleEngine(ABC):
    """Base for all role engines (assistant, user-proxy, judge, etc.)."""

    role: str

    def __init__(self, role: str, role_config: RoleModelConfig, adapter: ProviderAdapter) -> None:
        self.role = role
        self.role_config = role_config
        self.adapter = adapter

    def build_system_prompts(self, prompt_texts: List[str]) -> List[ChatMessage]:
        return [ChatMessage(role="system", content=text) for text in prompt_texts]


__all__ = [
    "ConversationTranscript",
    "ConversationTurn",
    "RoleEngine",
    "ToolInvocationRecord",
]

