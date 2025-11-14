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
    ConversationTranscript,
    ConversationTurn,
    Participant,
    ToolInvocationRecord,
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
    "ConversationTranscript",
    "ConversationTurn",
    "Participant",
    "ToolInvocationRecord",
    "TurnRunner",
    # Legacy (deprecated)
    "RoleEngine",
]

