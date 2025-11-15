"""Google Gemini adapter implementation for the multi-agent harness.

If the google-generativeai SDK is available, uses it; otherwise falls back to REST.
Reads GOOGLE_API_KEY from env or .env for REST fallback.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Sequence
from typing import Any, cast

from .base import (
    ChatMessage,
    ChatResponse,
    ProviderAdapter,
    ResponseFormat,
    ToolCall,
    ToolDefinition,
)
from ..config import RoleModelConfig

try:
    import google.generativeai as genai
except ImportError:  # pragma: no cover - optional dependency
    genai = None


class GeminiAdapter(ProviderAdapter):
    provider_name = "gemini"

    def __init__(self, api_key: str | None = None, max_retries: int = 2) -> None:
        super().__init__(api_key or os.environ.get("GOOGLE_API_KEY"))
        if genai and self.api_key:
            genai.configure(api_key=self.api_key)
        self._max_retries = max(0, max_retries)

    def send_chat(
        self,
        role_config: RoleModelConfig,
        messages: Sequence[ChatMessage],
        tools: Iterable[ToolDefinition] | None = None,
        response_format: ResponseFormat | None = None,
        tool_choice: str | None = None,
    ) -> ChatResponse:
        # Extract system instructions
        system_instructions: list[str] = []
        converted_messages: list[dict[str, Any]] = []

        for msg in messages:
            if msg.role == "system":
                if isinstance(msg.content, str):
                    system_instructions.append(msg.content)
                else:
                    system_instructions.append(str(msg.content))
            else:
                converted_messages.append(self._convert_message(msg))

        # Build generation config
        generation_config: dict[str, Any] = {
            "temperature": role_config.temperature,
            "top_p": role_config.top_p,
        }

        if response_format and response_format.type == "json_schema":
            # Gemini supports JSON schema in response_mime_type and response_schema
            generation_config["response_mime_type"] = "application/json"
            if response_format.json_schema:
                # Extract schema from json_schema wrapper if present
                schema = response_format.json_schema
                if "schema" in schema:
                    generation_config["response_schema"] = schema["schema"]
                else:
                    generation_config["response_schema"] = schema

        if genai:
            return self._send_with_sdk(
                model_name=role_config.model,
                messages=converted_messages,
                system_instruction="\n\n".join(system_instructions) if system_instructions else None,
                tools=list(tools) if tools else None,
                generation_config=generation_config,
            )
        else:
            return self._send_with_rest(
                model_name=role_config.model,
                messages=converted_messages,
                system_instruction="\n\n".join(system_instructions) if system_instructions else None,
                tools=list(tools) if tools else None,
                generation_config=generation_config,
            )

    def supports_tools(self) -> bool:
        return True

    def _send_with_sdk(
        self,
        model_name: str,
        messages: list[dict[str, Any]],
        system_instruction: str | None,
        tools: list[ToolDefinition] | None,
        generation_config: dict[str, Any],
    ) -> ChatResponse:
        """Send request using the Google Generative AI SDK."""
        if genai is None:  # pragma: no cover - guard for typing
            raise RuntimeError("google-generativeai SDK is not available")
        model_kwargs: dict[str, Any] = {}
        if system_instruction:
            model_kwargs["system_instruction"] = system_instruction

        model = genai.GenerativeModel(
            model_name=model_name,
            **model_kwargs
        )

        # Convert tools to Gemini format
        gemini_tools = None
        if tools:
            gemini_tools = [self._convert_tool_to_gemini(tool) for tool in tools]

        # Build contents from messages
        contents = [self._message_to_content(msg) for msg in messages]

        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                if gemini_tools:
                    response = model.generate_content(
                        contents,
                        tools=gemini_tools,
                        generation_config=generation_config,
                    )
                else:
                    response = model.generate_content(
                        contents,
                        generation_config=generation_config,
                    )
                return self._convert_sdk_response(response)
            except Exception as exc:  # pragma: no cover - requires network
                last_error = exc
                if attempt == self._max_retries:
                    break

        raise RuntimeError(f"Gemini SDK call failed after retries: {last_error}") from last_error

    def _send_with_rest(
        self,
        model_name: str,
        messages: list[dict[str, Any]],
        system_instruction: str | None,
        tools: list[ToolDefinition] | None,
        generation_config: dict[str, Any],
    ) -> ChatResponse:
        """Send request using REST API."""
        payload: dict[str, Any] = {
            "contents": [self._message_to_content(msg) for msg in messages],
            "generationConfig": generation_config,
        }

        if system_instruction:
            payload["systemInstruction"] = {
                "parts": [{"text": system_instruction}]
            }

        if tools:
            payload["tools"] = [
                {
                    "functionDeclarations": [self._convert_tool_to_rest(tool) for tool in tools]
                }
            ]

        response = self._rest_generate_content(model_name, payload)
        return self._convert_rest_response(response)

    @staticmethod
    def _convert_message(message: ChatMessage) -> dict[str, Any]:
        """Convert our ChatMessage to Gemini's internal format (before parts conversion)."""
        # Map our roles to Gemini's roles
        role_map = {
            "user": "user",
            "assistant": "model",
            "system": "user",  # System messages become user messages
            "tool": "user",  # Tool results become user messages
        }

        gemini_role = role_map.get(message.role, "user")

        return {
            "role": gemini_role,
            "content": message.content,
            "original_role": message.role,  # Keep track for special handling
        }

    @staticmethod
    def _message_to_content(message: dict[str, Any]) -> dict[str, Any]:
        """Convert message to Gemini content format with parts."""
        parts: list[dict[str, Any]] = []

        # Handle tool results
        if message.get("original_role") == "tool":
            content = message["content"]
            if isinstance(content, dict) and "tool_call_id" in content:
                # This is a function response
                parts.append(
                    {
                        "functionResponse": {
                            "name": content.get("name", "unknown"),
                            "response": {"result": content.get("content", "")},
                        }
                    }
                )
            else:
                parts.append({"text": str(content)})
        # Handle assistant messages with tool calls
        elif message.get("original_role") == "assistant" and isinstance(message["content"], dict):
            content_dict = message["content"]
            # Add text content if present
            if "content" in content_dict and content_dict["content"]:
                parts.append({"text": content_dict["content"]})
            # Add function calls
            if "tool_calls" in content_dict:
                for tc in content_dict["tool_calls"]:
                    func = tc.get("function", {})
                    parts.append(
                        {
                            "functionCall": {
                                "name": func.get("name", ""),
                                "args": json.loads(func.get("arguments", "{}")),
                            }
                        }
                    )
        # Handle simple text messages
        else:
            content = message["content"]
            if isinstance(content, str):
                parts.append({"text": content})
            else:
                parts.append({"text": str(content)})

        return {
            "role": message["role"],
            "parts": parts,
        }

    @staticmethod
    def _convert_tool_to_gemini(tool: ToolDefinition) -> Any:
        """Convert ToolDefinition to Gemini SDK format."""
        # The SDK expects actual genai types, but we'll return dict for compatibility
        return {
            "function_declarations": [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema,
                }
            ]
        }

    @staticmethod
    def _convert_tool_to_rest(tool: ToolDefinition) -> dict[str, Any]:
        """Convert ToolDefinition to Gemini REST format."""
        return {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.input_schema,
        }

    @staticmethod
    def _convert_sdk_response(response: Any) -> ChatResponse:
        """Convert Gemini SDK response to ChatResponse."""
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        # Process response parts
        for candidate in response.candidates:
            for part in candidate.content.parts:
                if hasattr(part, "text"):
                    text_parts.append(part.text)
                elif hasattr(part, "function_call"):
                    fc = part.function_call
                    tool_calls.append(
                        ToolCall(
                            name=fc.name,
                            arguments=dict(fc.args),
                            call_id=fc.name,  # Gemini doesn't provide separate IDs
                        )
                    )

        content = "".join(text_parts)

        return ChatResponse(
            message=ChatMessage(role="assistant", content=content),
            tool_calls=tuple(tool_calls),
            raw=response,
        )

    @staticmethod
    def _convert_rest_response(response: dict[str, Any]) -> ChatResponse:
        """Convert Gemini REST response to ChatResponse."""
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        candidates = response.get("candidates", [])
        if candidates:
            content = candidates[0].get("content", {})
            parts = content.get("parts", [])

            for part in parts:
                if "text" in part:
                    text_parts.append(part["text"])
                elif "functionCall" in part:
                    fc = part["functionCall"]
                    tool_calls.append(
                        ToolCall(
                            name=fc.get("name", ""),
                            arguments=fc.get("args", {}),
                            call_id=fc.get("name", ""),
                        )
                    )

        content = "".join(text_parts)

        return ChatResponse(
            message=ChatMessage(role="assistant", content=content),
            tool_calls=tuple(tool_calls),
            raw=response,
        )

    @staticmethod
    def _rest_generate_content(model_name: str, body: dict[str, Any]) -> dict[str, Any]:
        """Make a REST call to Gemini's generateContent API."""
        import urllib.request
        import json as _json

        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            GeminiAdapter._load_dotenv()
            api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY not set (and .env missing)")

        # Gemini REST API endpoint
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"

        data = _json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
            },
        )
        import urllib.error

        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                payload = cast(dict[str, Any], _json.loads(resp.read().decode("utf-8")))
            return payload
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "ignore") if hasattr(e, "read") else ""
            raise urllib.error.HTTPError(
                e.url,
                e.code,
                f"{e.reason}: {detail}",
                e.hdrs,
                e.fp,
            ) from e

    @staticmethod
    def _load_dotenv() -> None:
        """Load environment variables from .env file."""
        path = os.path.abspath(os.path.join(os.getcwd(), ".env"))
        try:
            with open(path, encoding="utf-8") as f:
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


__all__ = ["GeminiAdapter"]
