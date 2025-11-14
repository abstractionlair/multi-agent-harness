from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(slots=True)
class RoleModelConfig:
    """Minimal model selection/config for a single role."""

    provider: str
    model: str
    temperature: float = 0.0
    top_p: float = 1.0
    seed: Optional[int] = None


__all__ = ["RoleModelConfig"]

