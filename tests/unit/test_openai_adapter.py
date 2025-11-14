"""Unit tests for OpenAI adapter payload conversion (no network required)."""

import json
import pytest
from typing import Any, Dict

from multi_agent_harness.adapters.base import ChatMessage, ToolDefinition, ToolCall
from multi_agent_harness.adapters.openai import OpenAIAdapter
from multi_agent_harness.config import RoleModelConfig


class TestOpenAIMessageConversion:
    """Test message conversion to OpenAI format."""

    def test_convert_simple_user_message(self) -> None:
        """Test conversion of a simple user message."""
        message = ChatMessage(role="user", content="Hello, world!")
        result = OpenAIAdapter._convert_message(message)

        assert result == {
            "role": "user",
            "content": "Hello, world!",
        }

    def test_convert_simple_assistant_message(self) -> None:
        """Test conversion of a simple assistant message."""
        message = ChatMessage(role="assistant", content="Hi there!")
        result = OpenAIAdapter._convert_message(message)

        assert result == {
            "role": "assistant",
            "content": "Hi there!",
        }

    def test_convert_system_message(self) -> None:
        """Test conversion of a system message."""
        message = ChatMessage(role="system", content="You are a helpful assistant.")
        result = OpenAIAdapter._convert_message(message)

        assert result == {
            "role": "system",
            "content": "You are a helpful assistant.",
        }

    def test_convert_assistant_message_with_tool_calls(self) -> None:
        """Test conversion of assistant message with tool calls."""
        tool_calls = [
            {
                "type": "function",
                "id": "call_123",
                "function": {
                    "name": "get_weather",
                    "arguments": json.dumps({"location": "San Francisco"}),
                },
            }
        ]
        message = ChatMessage(
            role="assistant",
            content={"tool_calls": tool_calls, "content": "Let me check the weather."},
        )
        result = OpenAIAdapter._convert_message(message)

        assert result == {
            "role": "assistant",
            "content": "Let me check the weather.",
            "tool_calls": tool_calls,
        }

    def test_convert_assistant_message_with_tool_calls_no_content(self) -> None:
        """Test conversion of assistant message with tool calls but no text content."""
        tool_calls = [
            {
                "type": "function",
                "id": "call_456",
                "function": {
                    "name": "calculate",
                    "arguments": json.dumps({"a": 5, "b": 3}),
                },
            }
        ]
        message = ChatMessage(
            role="assistant",
            content={"tool_calls": tool_calls},
        )
        result = OpenAIAdapter._convert_message(message)

        assert result == {
            "role": "assistant",
            "content": "",
            "tool_calls": tool_calls,
        }

    def test_convert_tool_message(self) -> None:
        """Test conversion of a tool result message."""
        message = ChatMessage(
            role="tool",
            content={
                "tool_call_id": "call_123",
                "content": json.dumps({"temperature": 72, "conditions": "sunny"}),
            },
        )
        result = OpenAIAdapter._convert_message(message)

        assert result == {
            "role": "tool",
            "tool_call_id": "call_123",
            "content": json.dumps({"temperature": 72, "conditions": "sunny"}),
        }

    def test_convert_tool_message_missing_tool_call_id(self) -> None:
        """Test that tool messages without tool_call_id raise ValueError."""
        message = ChatMessage(
            role="tool",
            content={"content": "some result"},
        )

        with pytest.raises(ValueError, match="must include.*tool_call_id"):
            OpenAIAdapter._convert_message(message)

    def test_convert_tool_message_with_string_content(self) -> None:
        """Test that tool messages with string content raise ValueError."""
        message = ChatMessage(
            role="tool",
            content="invalid string content",
        )

        with pytest.raises(ValueError, match="must include.*tool_call_id"):
            OpenAIAdapter._convert_message(message)


class TestOpenAIToolConversion:
    """Test tool definition conversion to OpenAI format."""

    def test_convert_simple_tool(self) -> None:
        """Test conversion of a simple tool definition."""
        tool = ToolDefinition(
            name="get_weather",
            description="Get the current weather for a location",
            input_schema={
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "The city name"},
                },
                "required": ["location"],
            },
        )
        result = OpenAIAdapter._convert_tool(tool)

        assert result == {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "The city name"},
                    },
                    "required": ["location"],
                },
            },
        }

    def test_convert_tool_with_multiple_parameters(self) -> None:
        """Test conversion of a tool with multiple parameters."""
        tool = ToolDefinition(
            name="calculate",
            description="Perform a mathematical calculation",
            input_schema={
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["add", "subtract", "multiply", "divide"],
                    },
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                },
                "required": ["operation", "a", "b"],
            },
        )
        result = OpenAIAdapter._convert_tool(tool)

        assert result["type"] == "function"
        assert result["function"]["name"] == "calculate"
        assert result["function"]["description"] == "Perform a mathematical calculation"
        assert len(result["function"]["parameters"]["required"]) == 3


class TestOpenAICompletionConversion:
    """Test conversion of OpenAI API responses to ChatResponse."""

    def test_convert_simple_sdk_completion(self) -> None:
        """Test conversion of a simple SDK completion response."""
        # Mock SDK completion object
        class MockMessage:
            content = "Hello! How can I help you?"
            tool_calls = None

        class MockChoice:
            message = MockMessage()

        class MockCompletion:
            choices = [MockChoice()]

        completion = MockCompletion()
        response = OpenAIAdapter._convert_completion_sdk(completion)

        assert response.message.role == "assistant"
        assert response.message.content == "Hello! How can I help you?"
        assert len(response.tool_calls) == 0

    def test_convert_sdk_completion_with_tool_calls(self) -> None:
        """Test conversion of SDK completion with tool calls."""
        # Mock SDK completion with tool calls
        class MockFunction:
            name = "get_weather"
            arguments = json.dumps({"location": "Boston"})

        class MockToolCall:
            id = "call_789"
            function = MockFunction()

        class MockMessage:
            content = None
            tool_calls = [MockToolCall()]

        class MockChoice:
            message = MockMessage()

        class MockCompletion:
            choices = [MockChoice()]

        completion = MockCompletion()
        response = OpenAIAdapter._convert_completion_sdk(completion)

        assert response.message.role == "assistant"
        assert response.message.content == ""
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "get_weather"
        assert response.tool_calls[0].arguments == {"location": "Boston"}
        assert response.tool_calls[0].call_id == "call_789"

    def test_convert_rest_completion(self) -> None:
        """Test conversion of REST API completion response."""
        completion = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "The answer is 42.",
                    },
                }
            ],
        }
        response = OpenAIAdapter._convert_completion_rest(completion)

        assert response.message.role == "assistant"
        assert response.message.content == "The answer is 42."
        assert len(response.tool_calls) == 0

    def test_convert_rest_completion_with_tool_calls(self) -> None:
        """Test conversion of REST completion with tool calls."""
        completion = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_abc",
                                "type": "function",
                                "function": {
                                    "name": "add",
                                    "arguments": json.dumps({"a": 10, "b": 20}),
                                },
                            }
                        ],
                    },
                }
            ],
        }
        response = OpenAIAdapter._convert_completion_rest(completion)

        assert response.message.role == "assistant"
        assert response.message.content == ""
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "add"
        assert response.tool_calls[0].arguments == {"a": 10, "b": 20}
        assert response.tool_calls[0].call_id == "call_abc"

    def test_convert_rest_completion_with_null_content(self) -> None:
        """Test conversion of REST completion with null content."""
        completion = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                    },
                }
            ],
        }
        response = OpenAIAdapter._convert_completion_rest(completion)

        assert response.message.content == ""

    def test_convert_rest_completion_with_malformed_tool_args(self) -> None:
        """Test conversion handles malformed tool arguments gracefully."""
        completion = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_xyz",
                                "type": "function",
                                "function": {
                                    "name": "broken_tool",
                                    "arguments": "not valid json{",
                                },
                            }
                        ],
                    },
                }
            ],
        }
        response = OpenAIAdapter._convert_completion_rest(completion)

        # Should gracefully handle malformed JSON with empty dict
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].arguments == {}


class TestOpenAIAdapterMethods:
    """Test OpenAIAdapter methods."""

    def test_supports_tools(self) -> None:
        """Test that OpenAI adapter reports tool support."""
        adapter = OpenAIAdapter(api_key="test_key")
        assert adapter.supports_tools() is True

    def test_provider_name(self) -> None:
        """Test that provider name is set correctly."""
        assert OpenAIAdapter.provider_name == "openai"
