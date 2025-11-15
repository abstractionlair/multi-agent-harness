from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(slots=True)
class RoleModelConfig:
    """Minimal model selection/config for a single role.

    Note: This is maintained for backward compatibility with the adapter interface.
    New code should use Participant class directly.
    """

    provider: str
    model: str
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int | None = None


@dataclass(slots=True)
class ParticipantConfig:
    """Configuration for a conversation participant.

    This combines model configuration with identity and behavior settings.
    """

    name: str
    provider: str
    model: str
    system_prompts: Sequence[str] = ()
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int | None = None


__all__ = ["ParticipantConfig", "RoleModelConfig"]
