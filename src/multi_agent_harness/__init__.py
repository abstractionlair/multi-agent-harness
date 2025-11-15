"""Multi-agent LLM conversation orchestration toolkit."""

from .adapters.base import (
    ChatMessage,
    ChatResponse,
    ProviderAdapter,
    ResponseFormat,
    SystemRole,
    ToolCall,
    ToolDefinition,
    ToolExecutor,
)
from .config import ParticipantConfig, RoleModelConfig
from .conversation import (
    ConversationRunner,
    ConversationTranscript,
    ConversationTurn,
    Participant,
    StopCondition,
    ToolInvocationRecord,
    TranscriptAnalyzer,
    TurnRunner,
)
from .engines.base import RoleEngine

__all__ = [
    # Configuration
    "ParticipantConfig",
    "RoleModelConfig",
    # Adapters
    "ChatMessage",
    "ChatResponse",
    "ProviderAdapter",
    "ResponseFormat",
    "SystemRole",
    "ToolCall",
    "ToolDefinition",
    "ToolExecutor",
    # Conversation
    "ConversationRunner",
    "ConversationTranscript",
    "ConversationTurn",
    "Participant",
    "StopCondition",
    "ToolInvocationRecord",
    "TranscriptAnalyzer",
    "TurnRunner",
    # Legacy (deprecated)
    "RoleEngine",
]
