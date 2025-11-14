"""Assistant engine that orchestrates prompts, history, and tool definitions."""

from __future__ import annotations

import json
from typing import Iterable, List, Optional, Sequence

from ..adapters.base import ChatMessage, ChatResponse, ProviderAdapter, ToolDefinition
from ..config import RoleModelConfig
from .base import RoleEngine


class AssistantEngine(RoleEngine):
    """Generic assistant role that can use tools via a router-like interface.

    The engine itself does not know how tools are executed; callers pass in a
    callable that takes (tool_name, arguments) and returns a result object.
    """

    def __init__(
        self,
        role_config: RoleModelConfig,
        adapter: ProviderAdapter,
        system_prompts: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__("assistant", role_config, adapter)
        self.system_prompts = self.build_system_prompts(list(system_prompts or []))

    def run_turn(
        self,
        history: List[ChatMessage],
        user_text: str,
        tools: Iterable[ToolDefinition] | None,
        execute_tool,
        max_steps: int = 6,
        tool_choice: str = "auto",
    ) -> ChatResponse:
        """Execute an assistant turn and resolve tool calls until completion or max_steps."""

        messages = list(self.system_prompts)
        messages.extend(history)
        messages.append(ChatMessage(role="user", content=user_text))
        tools_seq = tuple(tools or ())

        response = self.adapter.send_chat(
            role_config=self.role_config,
            messages=messages,
            tools=tools_seq,
            tool_choice=tool_choice,
        )

        steps = 0
        while response.tool_calls and steps < max_steps:
            tool_calls_payload = []
            for idx, call in enumerate(response.tool_calls):
                tool_calls_payload.append(
                    {
                        "type": "function",
                        "id": call.call_id or f"call_{idx}",
                        "function": {
                            "name": call.name,
                            "arguments": json.dumps(call.arguments),
                        },
                    }
                )
            messages.append(
                ChatMessage(role="assistant", content={"tool_calls": tool_calls_payload})
            )
            for call in response.tool_calls:
                result = execute_tool(call.name, call.arguments)
                tool_msg = ChatMessage(
                    role="tool",
                    content={
                        "tool_call_id": call.call_id or call.name,
                        "content": json.dumps(result),
                    },
                )
                messages.append(tool_msg)

            response = self.adapter.send_chat(
                role_config=self.role_config,
                messages=messages,
                tools=tools_seq,
                tool_choice=tool_choice,
            )
            steps += 1

        return response


__all__ = ["AssistantEngine"]

