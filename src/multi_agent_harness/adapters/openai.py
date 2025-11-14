"""OpenAI adapter implementation for the multi-agent harness.

If the OpenAI SDK is available, uses it; otherwise falls back to REST.
Reads OPENAI_API_KEY from env or .env for REST fallback.
"""

from __future__ import annotations

import os
import json
from typing import Iterable, Optional, Sequence, Any, Dict

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
    from openai import OpenAI  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    OpenAI = None


class OpenAIAdapter(ProviderAdapter):
    provider_name = "openai"

    def __init__(self, api_key: Optional[str] = None, max_retries: int = 2) -> None:
        super().__init__(api_key or os.environ.get("OPENAI_API_KEY"))
        self._client = OpenAI(api_key=self.api_key) if OpenAI else None
        self._max_retries = max(0, max_retries)

    def send_chat(
        self,
        role_config: RoleModelConfig,
        messages: Sequence[ChatMessage],
        tools: Optional[Iterable[ToolDefinition]] = None,
        response_format: Optional[ResponseFormat] = None,
        tool_choice: Optional[str] = None,
    ) -> ChatResponse:
        payload: Dict[str, Any] = {
            "model": role_config.model,
            "messages": [self._convert_message(msg) for msg in messages],
            "temperature": role_config.temperature,
            "top_p": role_config.top_p,
        }
        if role_config.seed is not None:
            payload["seed"] = role_config.seed
        if tools:
            payload["tools"] = [self._convert_tool(tool) for tool in tools]
            payload["tool_choice"] = tool_choice or "auto"
        if response_format and response_format.type == "json_schema" and response_format.json_schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": response_format.json_schema,
            }

        if self._client:
            completion = self._call_with_retries(payload)
            return self._convert_completion_sdk(completion)
        else:
            completion = self._rest_chat_completions_with_fallback(payload)
            return self._convert_completion_rest(completion)

    def supports_tools(self) -> bool:
        return True

    def _call_with_retries(self, payload: Dict[str, Any]):
        last_error: Optional[Exception] = None
        for attempt in range(self._max_retries + 1):
            try:
                return self._client.chat.completions.create(**payload)
            except Exception as exc:  # pragma: no cover - requires network
                last_error = exc
                if attempt == self._max_retries:
                    break
        raise RuntimeError(f"OpenAI chat completion failed after retries: {last_error}") from last_error

    @staticmethod
    def _convert_message(message: ChatMessage) -> Dict[str, Any]:
        # Assistant message with explicit tool_calls (structured)
        if message.role == "assistant" and isinstance(message.content, dict) and "tool_calls" in message.content:
            return {
                "role": "assistant",
                "content": message.content.get("content", ""),
                "tool_calls": message.content["tool_calls"],
            }
        if message.role == "tool":
            if not isinstance(message.content, dict) or "tool_call_id" not in message.content:
                raise ValueError(
                    "Tool role messages must include {'tool_call_id', 'content'} payload"
                )
            return {
                "role": "tool",
                "tool_call_id": message.content["tool_call_id"],
                "content": message.content.get("content", ""),
            }
        return {
            "role": message.role,
            "content": message.content,
        }

    @staticmethod
    def _convert_tool(tool: ToolDefinition) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema,
            },
        }

    @staticmethod
    def _convert_completion_sdk(completion: Any) -> ChatResponse:
        choice = completion.choices[0]
        content = choice.message.content or ""
        if isinstance(content, list):
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        tool_calls = tuple(
            ToolCall(
                name=tc.function.name,
                arguments=json.loads(tc.function.arguments or "{}"),
                call_id=getattr(tc, "id", None),
            )
            for tc in (choice.message.tool_calls or [])
        )
        return ChatResponse(
            message=ChatMessage(role="assistant", content=content),
            tool_calls=tool_calls,
            raw=completion,
        )

    @staticmethod
    def _convert_completion_rest(completion: Dict[str, Any]) -> ChatResponse:
        choice = completion["choices"][0]
        message = choice["message"]
        content = message.get("content") or ""
        tool_calls_data = message.get("tool_calls") or []
        tool_calls: list[ToolCall] = []
        for tc in tool_calls_data:
            fn = tc.get("function", {})
            name = fn.get("name")
            args_str = fn.get("arguments")
            try:
                args = json.loads(args_str or "{}")
            except Exception:
                args = {}
            tool_calls.append(ToolCall(name=name, arguments=args, call_id=tc.get("id")))
        return ChatResponse(
            message=ChatMessage(role="assistant", content=content),
            tool_calls=tuple(tool_calls),
            raw=completion,
        )

    @staticmethod
    def _rest_chat_completions(body: Dict[str, Any]) -> Dict[str, Any]:
        import urllib.request
        import json as _json

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            OpenAIAdapter._load_dotenv()
            api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set (and .env missing)")
        data = _json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
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

    def _rest_chat_completions_with_fallback(self, body: Dict[str, Any]) -> Dict[str, Any]:
        import urllib.error

        attempted: list[str] = []
        models = [body.get("model"), "gpt-4o-mini"]
        last_err: Optional[Exception] = None
        for m in models:
            if not m or m in attempted:
                continue
            attempted.append(m)
            body_mut = dict(body)
            body_mut["model"] = m
            try:
                return self._rest_chat_completions(body_mut)
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
        raise RuntimeError("OpenAI REST chat failed with unknown error")

    @staticmethod
    def _load_dotenv() -> None:
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


__all__ = ["OpenAIAdapter"]

