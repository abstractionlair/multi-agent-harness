"""Unit tests for TurnRunner logic."""

import pytest
from typing import Any, Dict, List, Optional
from unittest.mock import Mock, MagicMock

from multi_agent_harness.adapters.base import (
    ChatMessage,
    ChatResponse,
    ProviderAdapter,
    ResponseFormat,
    ToolCall,
    ToolDefinition,
)
from multi_agent_harness.config import RoleModelConfig
from multi_agent_harness.conversation.participant import Participant
from multi_agent_harness.conversation.turn_runner import TurnRunner


class MockAdapter(ProviderAdapter):
    """Mock provider adapter for testing."""

    provider_name = "mock"

    def __init__(self, responses: Optional[List[ChatResponse]] = None) -> None:
        super().__init__()
        self.responses = responses or []
        self.call_count = 0
        self.calls: List[Dict[str, Any]] = []

    def send_chat(
        self,
        role_config: RoleModelConfig,
        messages: List[ChatMessage],
        tools: Optional[List[ToolDefinition]] = None,
        response_format: Optional[ResponseFormat] = None,
        tool_choice: Optional[str] = None,
    ) -> ChatResponse:
        # Record the call
        self.calls.append({
            "role_config": role_config,
            "messages": messages,
            "tools": tools,
            "response_format": response_format,
            "tool_choice": tool_choice,
        })

        # Return next response
        if self.call_count < len(self.responses):
            response = self.responses[self.call_count]
            self.call_count += 1
            return response

        # Default response
        return ChatResponse(
            message=ChatMessage(role="assistant", content="Default response"),
            tool_calls=(),
            raw={},
        )

    def supports_tools(self) -> bool:
        return True


class TestTurnRunnerInitialization:
    """Test TurnRunner initialization."""

    def test_init_without_tools(self) -> None:
        """Test initialization without tools."""
        adapter = MockAdapter()
        participant = Participant(
            name="Test",
            adapter=adapter,
            model="test-model",
        )
        runner = TurnRunner(participant=participant)

        assert runner.participant == participant
        assert len(runner.tools) == 0
        assert runner.tool_executor is None

    def test_init_with_tools_but_no_executor_raises(self) -> None:
        """Test that providing tools without executor raises ValueError."""
        adapter = MockAdapter()
        participant = Participant(
            name="Test",
            adapter=adapter,
            model="test-model",
        )
        tools = [
            ToolDefinition(
                name="test_tool",
                description="A test tool",
                input_schema={"type": "object"},
            )
        ]

        with pytest.raises(ValueError, match="tool_executor is required"):
            TurnRunner(participant=participant, tools=tools)

    def test_init_with_tools_and_executor(self) -> None:
        """Test initialization with tools and executor."""
        adapter = MockAdapter()
        participant = Participant(
            name="Test",
            adapter=adapter,
            model="test-model",
        )
        tools = [
            ToolDefinition(
                name="test_tool",
                description="A test tool",
                input_schema={"type": "object"},
            )
        ]

        def executor(name: str, args: Dict[str, Any]) -> Any:
            return {"result": "ok"}

        runner = TurnRunner(
            participant=participant,
            tools=tools,
            tool_executor=executor,
        )

        assert len(runner.tools) == 1
        assert runner.tool_executor is not None


class TestTurnRunnerExecution:
    """Test TurnRunner turn execution."""

    def test_simple_turn_without_tools(self) -> None:
        """Test a simple turn without tool calls."""
        # Set up mock response
        response = ChatResponse(
            message=ChatMessage(role="assistant", content="Hello!"),
            tool_calls=(),
            raw={},
        )
        adapter = MockAdapter(responses=[response])
        participant = Participant(
            name="Assistant",
            adapter=adapter,
            model="test-model",
            system_prompts=["You are a helpful assistant."],
        )
        runner = TurnRunner(participant=participant)

        # Execute turn
        result = runner.run_turn(
            history=[],
            user_message="Hi there!",
        )

        assert result.message.content == "Hello!"
        assert len(result.tool_calls) == 0

        # Verify adapter was called once
        assert adapter.call_count == 1
        call = adapter.calls[0]
        assert len(call["messages"]) == 2  # system + user
        assert call["messages"][0].role == "system"
        assert call["messages"][1].role == "user"
        assert call["messages"][1].content == "Hi there!"

    def test_turn_with_history(self) -> None:
        """Test turn execution with conversation history."""
        response = ChatResponse(
            message=ChatMessage(role="assistant", content="Sure thing!"),
            tool_calls=(),
            raw={},
        )
        adapter = MockAdapter(responses=[response])
        participant = Participant(
            name="Assistant",
            adapter=adapter,
            model="test-model",
        )
        runner = TurnRunner(participant=participant)

        history = [
            ChatMessage(role="user", content="Hello"),
            ChatMessage(role="assistant", content="Hi!"),
        ]

        result = runner.run_turn(
            history=history,
            user_message="Can you help me?",
        )

        # Verify history was included
        assert adapter.call_count == 1
        call = adapter.calls[0]
        assert len(call["messages"]) == 3  # history(2) + new user message(1)

    def test_turn_with_single_tool_call(self) -> None:
        """Test turn that makes one tool call."""
        # First response with tool call
        tool_call_response = ChatResponse(
            message=ChatMessage(role="assistant", content=""),
            tool_calls=(
                ToolCall(
                    name="get_weather",
                    arguments={"location": "Boston"},
                    call_id="call_123",
                ),
            ),
            raw={},
        )
        # Second response after tool execution
        final_response = ChatResponse(
            message=ChatMessage(role="assistant", content="The weather is sunny!"),
            tool_calls=(),
            raw={},
        )
        adapter = MockAdapter(responses=[tool_call_response, final_response])
        participant = Participant(
            name="Assistant",
            adapter=adapter,
            model="test-model",
        )

        # Mock tool executor
        tool_results = []

        def tool_executor(name: str, args: Dict[str, Any]) -> Any:
            tool_results.append({"name": name, "args": args})
            return {"temperature": 72, "conditions": "sunny"}

        tools = [
            ToolDefinition(
                name="get_weather",
                description="Get weather",
                input_schema={"type": "object"},
            )
        ]

        runner = TurnRunner(
            participant=participant,
            tools=tools,
            tool_executor=tool_executor,
        )

        result = runner.run_turn(
            history=[],
            user_message="What's the weather in Boston?",
        )

        # Verify final response
        assert result.message.content == "The weather is sunny!"

        # Verify tool was executed
        assert len(tool_results) == 1
        assert tool_results[0]["name"] == "get_weather"
        assert tool_results[0]["args"] == {"location": "Boston"}

        # Verify two adapter calls (initial + after tool)
        assert adapter.call_count == 2

    def test_turn_with_multiple_tool_calls(self) -> None:
        """Test turn that makes multiple sequential tool calls."""
        # Response 1: tool call
        response1 = ChatResponse(
            message=ChatMessage(role="assistant", content=""),
            tool_calls=(
                ToolCall(name="tool1", arguments={"x": 1}, call_id="call_1"),
            ),
            raw={},
        )
        # Response 2: another tool call
        response2 = ChatResponse(
            message=ChatMessage(role="assistant", content=""),
            tool_calls=(
                ToolCall(name="tool2", arguments={"y": 2}, call_id="call_2"),
            ),
            raw={},
        )
        # Response 3: final answer
        response3 = ChatResponse(
            message=ChatMessage(role="assistant", content="All done!"),
            tool_calls=(),
            raw={},
        )
        adapter = MockAdapter(responses=[response1, response2, response3])
        participant = Participant(
            name="Assistant",
            adapter=adapter,
            model="test-model",
        )

        executed_tools = []

        def tool_executor(name: str, args: Dict[str, Any]) -> Any:
            executed_tools.append(name)
            return {"status": "ok"}

        tools = [
            ToolDefinition(name="tool1", description="Tool 1", input_schema={}),
            ToolDefinition(name="tool2", description="Tool 2", input_schema={}),
        ]

        runner = TurnRunner(
            participant=participant,
            tools=tools,
            tool_executor=tool_executor,
        )

        result = runner.run_turn(
            history=[],
            user_message="Do the things",
            max_tool_steps=10,
        )

        assert result.message.content == "All done!"
        assert executed_tools == ["tool1", "tool2"]
        assert adapter.call_count == 3

    def test_turn_respects_max_tool_steps(self) -> None:
        """Test that turn execution respects max_tool_steps limit."""
        # Always return a tool call response
        tool_call_response = ChatResponse(
            message=ChatMessage(role="assistant", content=""),
            tool_calls=(
                ToolCall(name="tool", arguments={}, call_id="call_x"),
            ),
            raw={},
        )
        adapter = MockAdapter(
            responses=[tool_call_response] * 10  # More than max_tool_steps
        )
        participant = Participant(
            name="Assistant",
            adapter=adapter,
            model="test-model",
        )

        call_count = 0

        def tool_executor(name: str, args: Dict[str, Any]) -> Any:
            nonlocal call_count
            call_count += 1
            return {}

        tools = [
            ToolDefinition(name="tool", description="Tool", input_schema={}),
        ]

        runner = TurnRunner(
            participant=participant,
            tools=tools,
            tool_executor=tool_executor,
        )

        result = runner.run_turn(
            history=[],
            user_message="Test",
            max_tool_steps=3,
        )

        # Should stop after max_tool_steps
        assert call_count == 3
        assert adapter.call_count == 4  # Initial call + 3 tool iterations

    def test_turn_with_missing_call_id_raises(self) -> None:
        """Test that missing call_id in tool call raises ValueError."""
        tool_call_response = ChatResponse(
            message=ChatMessage(role="assistant", content=""),
            tool_calls=(
                ToolCall(name="tool", arguments={}, call_id=None),  # Missing call_id
            ),
            raw={},
        )
        adapter = MockAdapter(responses=[tool_call_response])
        participant = Participant(
            name="Assistant",
            adapter=adapter,
            model="test-model",
        )

        def tool_executor(name: str, args: Dict[str, Any]) -> Any:
            return {}

        tools = [
            ToolDefinition(name="tool", description="Tool", input_schema={}),
        ]

        runner = TurnRunner(
            participant=participant,
            tools=tools,
            tool_executor=tool_executor,
        )

        with pytest.raises(ValueError, match="missing required call_id"):
            runner.run_turn(
                history=[],
                user_message="Test",
            )

    def test_turn_with_response_format(self) -> None:
        """Test turn execution with response format."""
        response = ChatResponse(
            message=ChatMessage(role="assistant", content='{"answer": 42}'),
            tool_calls=(),
            raw={},
        )
        adapter = MockAdapter(responses=[response])
        participant = Participant(
            name="Assistant",
            adapter=adapter,
            model="test-model",
        )
        runner = TurnRunner(participant=participant)

        response_format = ResponseFormat(
            type="json_schema",
            json_schema={"type": "object", "properties": {"answer": {"type": "number"}}},
        )

        result = runner.run_turn(
            history=[],
            user_message="What is the answer?",
            response_format=response_format,
        )

        # Verify response_format was passed to adapter
        assert adapter.call_count == 1
        call = adapter.calls[0]
        assert call["response_format"] == response_format

    def test_turn_with_tool_choice(self) -> None:
        """Test turn execution with specific tool_choice."""
        response = ChatResponse(
            message=ChatMessage(role="assistant", content=""),
            tool_calls=(
                ToolCall(name="required_tool", arguments={}, call_id="call_1"),
            ),
            raw={},
        )
        final_response = ChatResponse(
            message=ChatMessage(role="assistant", content="Done"),
            tool_calls=(),
            raw={},
        )
        adapter = MockAdapter(responses=[response, final_response])
        participant = Participant(
            name="Assistant",
            adapter=adapter,
            model="test-model",
        )

        def tool_executor(name: str, args: Dict[str, Any]) -> Any:
            return {}

        tools = [
            ToolDefinition(name="required_tool", description="Tool", input_schema={}),
        ]

        runner = TurnRunner(
            participant=participant,
            tools=tools,
            tool_executor=tool_executor,
        )

        result = runner.run_turn(
            history=[],
            user_message="Test",
            tool_choice="required",
        )

        # Verify tool_choice was passed
        assert adapter.call_count == 2
        assert adapter.calls[0]["tool_choice"] == "required"
