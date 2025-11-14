"""Multi-agent LLM conversation orchestration toolkit."""

# Configuration
from .config import ParticipantConfig, RoleModelConfig

# Adapter primitives
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

# Conversation primitives
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

# Legacy imports (backward compatibility)
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

