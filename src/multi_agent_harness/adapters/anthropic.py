"""Anthropic adapter implementation for the multi-agent harness.

If the Anthropic SDK is available, uses it; otherwise falls back to REST.
Reads ANTHROPIC_API_KEY from env or .env for REST fallback.
"""

from __future__ import annotations

import os
import json
from typing import Iterable, Optional, Sequence, Any, Dict, List

from .base import (
    ChatMessage,
    ChatResponse,
    ProviderAdapter,
    ResponseFormat,
    ToolDefinition,
    ToolCall,
)
from ..config import RoleModelConfig

try:
    from anthropic import Anthropic  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    Anthropic = None


class AnthropicAdapter(ProviderAdapter):
    provider_name = "anthropic"

    def __init__(self, api_key: Optional[str] = None, max_retries: int = 2) -> None:
        super().__init__(api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self._client = Anthropic(api_key=self.api_key) if Anthropic else None
        self._max_retries = max(0, max_retries)

    def send_chat(
        self,
        role_config: RoleModelConfig,
        messages: Sequence[ChatMessage],
        tools: Optional[Iterable[ToolDefinition]] = None,
        response_format: Optional[ResponseFormat] = None,
        tool_choice: Optional[str] = None,
    ) -> ChatResponse:
        # Extract system prompts from system messages
        system_prompts: List[str] = []
        converted_messages: List[Dict[str, Any]] = []

        for msg in messages:
            if msg.role == "system":
                # Extract system content
                if isinstance(msg.content, str):
                    system_prompts.append(msg.content)
                else:
                    # If structured, try to get text content
                    system_prompts.append(str(msg.content))
            else:
                converted_messages.append(self._convert_message(msg))

        payload: Dict[str, Any] = {
            "model": role_config.model,
            "messages": converted_messages,
            "max_tokens": 4096,  # Anthropic requires max_tokens
            "temperature": role_config.temperature,
            "top_p": role_config.top_p,
        }

        if system_prompts:
            payload["system"] = "\n\n".join(system_prompts)

        if tools:
            payload["tools"] = [self._convert_tool(tool) for tool in tools]
            if tool_choice:
                if tool_choice == "auto":
                    payload["tool_choice"] = {"type": "auto"}
                elif tool_choice == "required":
                    payload["tool_choice"] = {"type": "any"}
                else:
                    payload["tool_choice"] = {"type": "tool", "name": tool_choice}

        # Anthropic doesn't support structured output in the same way
        # We'll ignore response_format for now (could be added in prompts)

        if self._client:
            completion = self._call_with_retries(payload)
            return self._convert_completion_sdk(completion)
        else:
            completion = self._rest_messages_with_fallback(payload)
            return self._convert_completion_rest(completion)

    def supports_tools(self) -> bool:
        return True

    def _call_with_retries(self, payload: Dict[str, Any]):
        last_error: Optional[Exception] = None
        for attempt in range(self._max_retries + 1):
            try:
                return self._client.messages.create(**payload)
            except Exception as exc:  # pragma: no cover - requires network
                last_error = exc
                if attempt == self._max_retries:
                    break
        raise RuntimeError(f"Anthropic messages API failed after retries: {last_error}") from last_error

    @staticmethod
    def _convert_message(message: ChatMessage) -> Dict[str, Any]:
        """Convert our ChatMessage to Anthropic's format.

        Anthropic uses content blocks instead of simple strings:
        - Text content: [{"type": "text", "text": "..."}]
        - Tool use: [{"type": "tool_use", "id": "...", "name": "...", "input": {...}}]
        - Tool result: [{"type": "tool_result", "tool_use_id": "...", "content": "..."}]
        """
        # Assistant message with tool calls
        if message.role == "assistant" and isinstance(message.content, dict) and "tool_calls" in message.content:
            content_blocks: List[Dict[str, Any]] = []

            # Add text content if present
            text_content = message.content.get("content", "")
            if text_content:
                content_blocks.append({"type": "text", "text": text_content})

            # Add tool use blocks
            for tc in message.content.get("tool_calls", []):
                content_blocks.append({
                    "type": "tool_use",
                    "id": tc.get("id", tc.get("function", {}).get("name", "unknown")),
                    "name": tc.get("function", {}).get("name", ""),
                    "input": json.loads(tc.get("function", {}).get("arguments", "{}"))
                })

            return {
                "role": "assistant",
                "content": content_blocks
            }

        # Tool result message - convert to user message with tool_result blocks
        if message.role == "tool":
            if not isinstance(message.content, dict) or "tool_call_id" not in message.content:
                raise ValueError(
                    "Tool role messages must include {'tool_call_id', 'content'} payload"
                )
            return {
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": message.content["tool_call_id"],
                    "content": message.content.get("content", "")
                }]
            }

        # Simple text message
        content = message.content if isinstance(message.content, str) else str(message.content)
        return {
            "role": message.role,
            "content": content
        }

    @staticmethod
    def _convert_tool(tool: ToolDefinition) -> Dict[str, Any]:
        """Convert our ToolDefinition to Anthropic's tool format."""
        return {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.input_schema,
        }

    @staticmethod
    def _convert_completion_sdk(completion: Any) -> ChatResponse:
        """Convert Anthropic SDK response to our ChatResponse."""
        # Extract text content from content blocks
        text_parts: List[str] = []
        tool_calls: List[ToolCall] = []

        for block in completion.content:
            if hasattr(block, "type"):
                if block.type == "text":
                    text_parts.append(block.text)
                elif block.type == "tool_use":
                    tool_calls.append(ToolCall(
                        name=block.name,
                        arguments=block.input,
                        call_id=block.id
                    ))

        content = "".join(text_parts)

        return ChatResponse(
            message=ChatMessage(role="assistant", content=content),
            tool_calls=tuple(tool_calls),
            raw=completion,
        )

    @staticmethod
    def _convert_completion_rest(completion: Dict[str, Any]) -> ChatResponse:
        """Convert Anthropic REST response to our ChatResponse."""
        content_blocks = completion.get("content", [])
        text_parts: List[str] = []
        tool_calls: List[ToolCall] = []

        for block in content_blocks:
            if block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif block.get("type") == "tool_use":
                tool_calls.append(ToolCall(
                    name=block.get("name", ""),
                    arguments=block.get("input", {}),
                    call_id=block.get("id")
                ))

        content = "".join(text_parts)

        return ChatResponse(
            message=ChatMessage(role="assistant", content=content),
            tool_calls=tuple(tool_calls),
            raw=completion,
        )

    @staticmethod
    def _rest_messages(body: Dict[str, Any]) -> Dict[str, Any]:
        """Make a REST call to Anthropic's messages API."""
        import urllib.request
        import json as _json

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            AnthropicAdapter._load_dotenv()
            api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set (and .env missing)")

        data = _json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=data,
            headers={
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            },
        )
        import urllib.error

        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                payload = _json.loads(resp.read().decode("utf-8"))
            return payload
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "ignore") if hasattr(e, "read") else ""
            raise urllib.error.HTTPError(e.url, e.code, f"{e.reason}: {detail}", e.hdrs, e.fp)

    def _rest_messages_with_fallback(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """Make REST call with model fallback."""
        import urllib.error

        attempted: list[str] = []
        models = [body.get("model"), "claude-3-5-sonnet-20241022"]
        last_err: Optional[Exception] = None
        for m in models:
            if not m or m in attempted:
                continue
            attempted.append(m)
            body_mut = dict(body)
            body_mut["model"] = m
            try:
                return self._rest_messages(body_mut)
            except urllib.error.HTTPError as e:
                if e.code in (400, 404, 422):
                    last_err = e
                    continue
                raise
            except Exception as e:  # pragma: no cover
                last_err = e
                continue
        if last_err:
            raise last_err
        raise RuntimeError("Anthropic REST messages API failed with unknown error")

    @staticmethod
    def _load_dotenv() -> None:
        """Load environment variables from .env file."""
        path = os.path.abspath(os.path.join(os.getcwd(), ".env"))
        try:
            with open(path, "r", encoding="utf-8") as f:
                for raw in f:
                    line = raw.strip()
                    if not line or line.startswith("#"):
                        continue
                    if line.startswith("export "):
                        line = line[len("export ") :]
                    if "=" in line:
                        k, v = line.split("=", 1)
                        v = v.strip().strip('"').strip("'")
                        if k not in os.environ:
                            os.environ[k] = v
        except FileNotFoundError:
            pass


__all__ = ["AnthropicAdapter"]
