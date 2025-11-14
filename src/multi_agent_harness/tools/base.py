"""Tool system base types.

This module re-exports tool-related types from the adapter layer for
convenience. Host projects can implement ToolExecutor to provide actual
tool execution logic.
"""

from ..adapters.base import ToolDefinition, ToolExecutor

__all__ = [
    "ToolDefinition",
    "ToolExecutor",
]
