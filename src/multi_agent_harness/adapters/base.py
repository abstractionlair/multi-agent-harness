"""Provider adapter interfaces for the multi-agent harness."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Optional, Sequence

from ..config import RoleModelConfig

SystemRole = Literal["system", "user", "assistant", "tool"]


@dataclass(slots=True)
class ChatMessage:
    role: SystemRole
    content: Any


@dataclass(slots=True)
class ToolDefinition:
    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass(slots=True)
class ToolCall:
    """Represents a tool invocation requested by a model."""

    name: str
    arguments: dict[str, Any]
    call_id: str | None = None


@dataclass(slots=True)
class ResponseFormat:
    """Desired response formatting for providers that support JSON schemas."""

    type: Literal["default", "json_schema"] = "default"
    json_schema: Optional[dict[str, Any]] = None


@dataclass(slots=True)
class ChatResponse:
    message: ChatMessage
    tool_calls: tuple[ToolCall, ...]
    raw: Any


class ProviderAdapter(ABC):
    """Abstract base class for provider-specific adapters."""

    provider_name: str

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = api_key

    @abstractmethod
    def send_chat(
        self,
        role_config: RoleModelConfig,
        messages: Sequence[ChatMessage],
        tools: Optional[Iterable[ToolDefinition]] = None,
        response_format: Optional[ResponseFormat] = None,
        tool_choice: Optional[str] = None,
    ) -> ChatResponse:
        """Execute a chat completion request."""

    @abstractmethod
    def supports_tools(self) -> bool:
        """Return True when the provider natively supports tool/function calling."""


__all__ = [
    "ChatMessage",
    "ChatResponse",
    "ProviderAdapter",
    "ResponseFormat",
    "ToolCall",
    "ToolDefinition",
]

