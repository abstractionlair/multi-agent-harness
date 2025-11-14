"""Participant: a model + prompt + adapter representing one actor in a conversation."""

from __future__ import annotations

from typing import List, Optional, Sequence

from ..adapters.base import ChatMessage, ProviderAdapter
from ..config import RoleModelConfig


class Participant:
    """Represents one actor in a conversation with a specific model and configuration.

    A Participant combines:
    - A provider adapter (e.g., OpenAI, Anthropic)
    - Model configuration (model name, temperature, etc.)
    - System prompts that define the participant's behavior
    """

    def __init__(
        self,
        name: str,
        adapter: ProviderAdapter,
        model: str,
        system_prompts: Optional[Sequence[str]] = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize a conversation participant.

        Args:
            name: Identifier for this participant (e.g., "Alice", "Judge")
            adapter: Provider adapter for making API calls
            model: Model identifier (e.g., "gpt-4o-mini", "claude-sonnet-4")
            system_prompts: List of system prompt strings
            temperature: Sampling temperature (0.0 = deterministic)
            top_p: Nucleus sampling parameter
            seed: Random seed for deterministic sampling (if supported)
        """
        self.name = name
        self.adapter = adapter
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.seed = seed
        self._system_prompts = list(system_prompts or [])

    @property
    def system_prompts(self) -> List[ChatMessage]:
        """Get system prompts as ChatMessage objects."""
        return [ChatMessage(role="system", content=text) for text in self._system_prompts]

    @property
    def model_config(self) -> RoleModelConfig:
        """Get the model configuration for this participant.

        Returns a RoleModelConfig compatible with the current adapter interface.
        This is a transitional property during the refactoring phase.
        """
        return RoleModelConfig(
            provider=self.adapter.provider_name,
            model=self.model,
            temperature=self.temperature,
            top_p=self.top_p,
            seed=self.seed,
        )

    def __repr__(self) -> str:
        return f"Participant(name={self.name!r}, model={self.model!r})"


__all__ = ["Participant"]
