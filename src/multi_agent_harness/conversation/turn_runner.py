"""TurnRunner: Execute a single model turn with tool loop."""

from __future__ import annotations

import json
from typing import Iterable, List, Optional

from ..adapters.base import (
    ChatMessage,
    ChatResponse,
    ResponseFormat,
    ToolDefinition,
    ToolExecutor,
)
from .participant import Participant


class TurnRunner:
    """Executes a single turn for a participant, handling tool calls until completion.

    This is a generic turn executor that works with any participant, regardless of
    their role. It handles the tool loop (call → execute → continue) automatically.
    """

    def __init__(
        self,
        participant: Participant,
        tools: Optional[Iterable[ToolDefinition]] = None,
        tool_executor: Optional[ToolExecutor] = None,
    ) -> None:
        """Initialize a turn runner.

        Args:
            participant: The participant who will respond
            tools: Optional tool definitions available to the participant
            tool_executor: Callable that executes tools (tool_name, args) -> result
        """
        self.participant = participant
        self.tools = tuple(tools or ())
        self.tool_executor = tool_executor

        if self.tools and not tool_executor:
            raise ValueError("tool_executor is required when tools are provided")

    def run_turn(
        self,
        history: List[ChatMessage],
        user_message: str,
        max_tool_steps: int = 6,
        tool_choice: str = "auto",
        response_format: Optional[ResponseFormat] = None,
    ) -> ChatResponse:
        """Execute a turn and resolve tool calls until completion or max_steps.

        Args:
            history: Previous conversation messages
            user_message: The new user message to respond to
            max_tool_steps: Maximum number of tool call iterations
            tool_choice: Tool selection strategy ("auto", "required", "none")
            response_format: Optional response formatting (e.g., JSON schema)

        Returns:
            ChatResponse with final message and any tool calls

        Raises:
            ValueError: If a tool call is missing required call_id
        """
        # Build message list: system prompts + history + new user message
        messages = list(self.participant.system_prompts)
        messages.extend(history)
        messages.append(ChatMessage(role="user", content=user_message))

        # Initial model call
        response = self.participant.adapter.send_chat(
            role_config=self.participant.model_config,
            messages=messages,
            tools=self.tools if self.tools else None,
            response_format=response_format,
            tool_choice=tool_choice if self.tools else None,
        )

        # Tool loop: continue until no tool calls or max steps reached
        steps = 0
        while response.tool_calls and steps < max_tool_steps:
            # Validate and prepare tool call message
            tool_calls_payload = []
            for idx, call in enumerate(response.tool_calls):
                if call.call_id is None:
                    raise ValueError(
                        f"Tool call '{call.name}' is missing required call_id. "
                        "Provider adapters must set call_id for all tool calls."
                    )
                tool_calls_payload.append(
                    {
                        "type": "function",
                        "id": call.call_id,
                        "function": {
                            "name": call.name,
                            "arguments": json.dumps(call.arguments),
                        },
                    }
                )

            # Add assistant's tool call message
            messages.append(
                ChatMessage(role="assistant", content={"tool_calls": tool_calls_payload})
            )

            # Execute each tool and add results
            for call in response.tool_calls:
                assert self.tool_executor is not None  # Type checker hint
                result = self.tool_executor(call.name, call.arguments)

                # Add tool result message
                tool_msg = ChatMessage(
                    role="tool",
                    content={
                        "tool_call_id": call.call_id,
                        "content": json.dumps(result),
                    },
                )
                messages.append(tool_msg)

            # Continue conversation with tool results
            response = self.participant.adapter.send_chat(
                role_config=self.participant.model_config,
                messages=messages,
                tools=self.tools if self.tools else None,
                response_format=response_format,
                tool_choice=tool_choice if self.tools else None,
            )
            steps += 1

        return response


__all__ = ["TurnRunner"]
