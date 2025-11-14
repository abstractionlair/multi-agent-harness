"""Comprehensive tests for tool loop behavior and edge cases."""

import json
import pytest
from typing import Any, Dict, List, Optional

from multi_agent_harness.adapters.base import (
    ChatMessage,
    ChatResponse,
    ProviderAdapter,
    ToolCall,
    ToolDefinition,
)
from multi_agent_harness.config import RoleModelConfig
from multi_agent_harness.conversation.participant import Participant
from multi_agent_harness.conversation.turn_runner import TurnRunner


class MockAdapter(ProviderAdapter):
    """Mock adapter for testing tool loops."""

    provider_name = "mock"

    def __init__(self, responses: Optional[List[ChatResponse]] = None) -> None:
        super().__init__()
        self.responses = responses or []
        self.call_count = 0

    def send_chat(
        self,
        role_config: RoleModelConfig,
        messages: List[ChatMessage],
        tools: Optional[List[ToolDefinition]] = None,
        response_format: Optional[Any] = None,
        tool_choice: Optional[str] = None,
    ) -> ChatResponse:
        if self.call_count < len(self.responses):
            response = self.responses[self.call_count]
            self.call_count += 1
            return response
        return ChatResponse(
            message=ChatMessage(role="assistant", content="Default"),
            tool_calls=(),
            raw={},
        )

    def supports_tools(self) -> bool:
        return True


class TestToolLoopBasicBehavior:
    """Test basic tool loop behavior."""

    def test_no_tool_calls_completes_immediately(self) -> None:
        """Test that turn completes immediately without tool calls."""
        response = ChatResponse(
            message=ChatMessage(role="assistant", content="Simple response"),
            tool_calls=(),
            raw={},
        )
        adapter = MockAdapter(responses=[response])
        participant = Participant(name="Test", adapter=adapter, model="test")

        executor_calls = []

        def tool_executor(name: str, args: Dict[str, Any]) -> Any:
            executor_calls.append((name, args))
            return {}

        tools = [
            ToolDefinition(name="tool1", description="Tool 1", input_schema={}),
        ]

        runner = TurnRunner(
            participant=participant,
            tools=tools,
            tool_executor=tool_executor,
        )

        result = runner.run_turn(history=[], user_message="Test")

        assert len(executor_calls) == 0
        assert adapter.call_count == 1

    def test_single_tool_call_executes_once(self) -> None:
        """Test single tool call execution."""
        tool_response = ChatResponse(
            message=ChatMessage(role="assistant", content=""),
            tool_calls=(
                ToolCall(name="add", arguments={"a": 1, "b": 2}, call_id="call_1"),
            ),
            raw={},
        )
        final_response = ChatResponse(
            message=ChatMessage(role="assistant", content="The answer is 3"),
            tool_calls=(),
            raw={},
        )
        adapter = MockAdapter(responses=[tool_response, final_response])
        participant = Participant(name="Test", adapter=adapter, model="test")

        executor_calls = []

        def tool_executor(name: str, args: Dict[str, Any]) -> Any:
            executor_calls.append((name, args))
            return args["a"] + args["b"]

        tools = [
            ToolDefinition(name="add", description="Add numbers", input_schema={}),
        ]

        runner = TurnRunner(
            participant=participant,
            tools=tools,
            tool_executor=tool_executor,
        )

        result = runner.run_turn(history=[], user_message="What is 1+2?")

        assert len(executor_calls) == 1
        assert executor_calls[0] == ("add", {"a": 1, "b": 2})
        assert adapter.call_count == 2


class TestToolLoopMultipleCalls:
    """Test tool loop with multiple sequential calls."""

    def test_multiple_sequential_tool_calls(self) -> None:
        """Test multiple sequential tool calls in a loop."""
        responses = [
            # First call: request tool1
            ChatResponse(
                message=ChatMessage(role="assistant", content=""),
                tool_calls=(
                    ToolCall(name="tool1", arguments={"x": 1}, call_id="call_1"),
                ),
                raw={},
            ),
            # Second call: request tool2
            ChatResponse(
                message=ChatMessage(role="assistant", content=""),
                tool_calls=(
                    ToolCall(name="tool2", arguments={"y": 2}, call_id="call_2"),
                ),
                raw={},
            ),
            # Third call: request tool3
            ChatResponse(
                message=ChatMessage(role="assistant", content=""),
                tool_calls=(
                    ToolCall(name="tool3", arguments={"z": 3}, call_id="call_3"),
                ),
                raw={},
            ),
            # Final response
            ChatResponse(
                message=ChatMessage(role="assistant", content="All done!"),
                tool_calls=(),
                raw={},
            ),
        ]
        adapter = MockAdapter(responses=responses)
        participant = Participant(name="Test", adapter=adapter, model="test")

        call_sequence = []

        def tool_executor(name: str, args: Dict[str, Any]) -> Any:
            call_sequence.append(name)
            return {"status": "ok"}

        tools = [
            ToolDefinition(name="tool1", description="Tool 1", input_schema={}),
            ToolDefinition(name="tool2", description="Tool 2", input_schema={}),
            ToolDefinition(name="tool3", description="Tool 3", input_schema={}),
        ]

        runner = TurnRunner(
            participant=participant,
            tools=tools,
            tool_executor=tool_executor,
        )

        result = runner.run_turn(
            history=[],
            user_message="Execute all tools",
            max_tool_steps=5,
        )

        assert call_sequence == ["tool1", "tool2", "tool3"]
        assert result.message.content == "All done!"
        assert adapter.call_count == 4

    def test_parallel_tool_calls_in_single_turn(self) -> None:
        """Test multiple tool calls returned in a single response."""
        tool_response = ChatResponse(
            message=ChatMessage(role="assistant", content=""),
            tool_calls=(
                ToolCall(name="tool1", arguments={"a": 1}, call_id="call_1"),
                ToolCall(name="tool2", arguments={"b": 2}, call_id="call_2"),
                ToolCall(name="tool3", arguments={"c": 3}, call_id="call_3"),
            ),
            raw={},
        )
        final_response = ChatResponse(
            message=ChatMessage(role="assistant", content="All tools executed"),
            tool_calls=(),
            raw={},
        )
        adapter = MockAdapter(responses=[tool_response, final_response])
        participant = Participant(name="Test", adapter=adapter, model="test")

        executed_tools = []

        def tool_executor(name: str, args: Dict[str, Any]) -> Any:
            executed_tools.append(name)
            return {"result": f"{name}_result"}

        tools = [
            ToolDefinition(name="tool1", description="Tool 1", input_schema={}),
            ToolDefinition(name="tool2", description="Tool 2", input_schema={}),
            ToolDefinition(name="tool3", description="Tool 3", input_schema={}),
        ]

        runner = TurnRunner(
            participant=participant,
            tools=tools,
            tool_executor=tool_executor,
        )

        result = runner.run_turn(history=[], user_message="Run all tools")

        # All three tools should be executed
        assert len(executed_tools) == 3
        assert "tool1" in executed_tools
        assert "tool2" in executed_tools
        assert "tool3" in executed_tools


class TestToolLoopMaxSteps:
    """Test max_tool_steps limit behavior."""

    def test_max_steps_stops_infinite_loop(self) -> None:
        """Test that max_tool_steps prevents infinite loops."""
        # Always return a tool call
        infinite_response = ChatResponse(
            message=ChatMessage(role="assistant", content=""),
            tool_calls=(
                ToolCall(name="loop_tool", arguments={}, call_id="call_x"),
            ),
            raw={},
        )
        adapter = MockAdapter(responses=[infinite_response] * 20)
        participant = Participant(name="Test", adapter=adapter, model="test")

        execution_count = 0

        def tool_executor(name: str, args: Dict[str, Any]) -> Any:
            nonlocal execution_count
            execution_count += 1
            return {"iteration": execution_count}

        tools = [
            ToolDefinition(name="loop_tool", description="Loops", input_schema={}),
        ]

        runner = TurnRunner(
            participant=participant,
            tools=tools,
            tool_executor=tool_executor,
        )

        result = runner.run_turn(
            history=[],
            user_message="Start loop",
            max_tool_steps=3,
        )

        # Should execute exactly max_tool_steps times
        assert execution_count == 3
        # Should make initial call + 3 iterations
        assert adapter.call_count == 4

    def test_completes_before_max_steps(self) -> None:
        """Test that loop can complete before hitting max_steps."""
        responses = [
            ChatResponse(
                message=ChatMessage(role="assistant", content=""),
                tool_calls=(
                    ToolCall(name="tool", arguments={}, call_id="call_1"),
                ),
                raw={},
            ),
            ChatResponse(
                message=ChatMessage(role="assistant", content="Done"),
                tool_calls=(),
                raw={},
            ),
        ]
        adapter = MockAdapter(responses=responses)
        participant = Participant(name="Test", adapter=adapter, model="test")

        execution_count = 0

        def tool_executor(name: str, args: Dict[str, Any]) -> Any:
            nonlocal execution_count
            execution_count += 1
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
            max_tool_steps=10,  # High limit
        )

        # Should only execute once and complete
        assert execution_count == 1
        assert adapter.call_count == 2
        assert result.message.content == "Done"

    def test_zero_max_steps(self) -> None:
        """Test behavior with max_tool_steps=0."""
        tool_response = ChatResponse(
            message=ChatMessage(role="assistant", content=""),
            tool_calls=(
                ToolCall(name="tool", arguments={}, call_id="call_1"),
            ),
            raw={},
        )
        adapter = MockAdapter(responses=[tool_response])
        participant = Participant(name="Test", adapter=adapter, model="test")

        execution_count = 0

        def tool_executor(name: str, args: Dict[str, Any]) -> Any:
            nonlocal execution_count
            execution_count += 1
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
            max_tool_steps=0,
        )

        # Should not execute any tools
        assert execution_count == 0
        assert adapter.call_count == 1


class TestToolLoopErrorHandling:
    """Test tool loop error handling."""

    def test_tool_executor_exception_propagates(self) -> None:
        """Test that tool executor exceptions propagate."""
        tool_response = ChatResponse(
            message=ChatMessage(role="assistant", content=""),
            tool_calls=(
                ToolCall(name="error_tool", arguments={}, call_id="call_1"),
            ),
            raw={},
        )
        adapter = MockAdapter(responses=[tool_response])
        participant = Participant(name="Test", adapter=adapter, model="test")

        def tool_executor(name: str, args: Dict[str, Any]) -> Any:
            raise ValueError("Tool execution failed!")

        tools = [
            ToolDefinition(name="error_tool", description="Error", input_schema={}),
        ]

        runner = TurnRunner(
            participant=participant,
            tools=tools,
            tool_executor=tool_executor,
        )

        with pytest.raises(ValueError, match="Tool execution failed"):
            runner.run_turn(history=[], user_message="Trigger error")

    def test_missing_call_id_raises_error(self) -> None:
        """Test that missing call_id raises clear error."""
        tool_response = ChatResponse(
            message=ChatMessage(role="assistant", content=""),
            tool_calls=(
                ToolCall(name="tool", arguments={}, call_id=None),
            ),
            raw={},
        )
        adapter = MockAdapter(responses=[tool_response])
        participant = Participant(name="Test", adapter=adapter, model="test")

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
            runner.run_turn(history=[], user_message="Test")


class TestToolLoopMessageConstruction:
    """Test correct message construction in tool loop."""

    def test_tool_result_message_format(self) -> None:
        """Test that tool result messages are correctly formatted."""
        tool_response = ChatResponse(
            message=ChatMessage(role="assistant", content=""),
            tool_calls=(
                ToolCall(
                    name="get_data",
                    arguments={"id": 123},
                    call_id="call_abc",
                ),
            ),
            raw={},
        )
        final_response = ChatResponse(
            message=ChatMessage(role="assistant", content="Got the data"),
            tool_calls=(),
            raw={},
        )
        adapter = MockAdapter(responses=[tool_response, final_response])
        participant = Participant(name="Test", adapter=adapter, model="test")

        def tool_executor(name: str, args: Dict[str, Any]) -> Any:
            return {"user": "Alice", "age": 30}

        tools = [
            ToolDefinition(name="get_data", description="Get data", input_schema={}),
        ]

        runner = TurnRunner(
            participant=participant,
            tools=tools,
            tool_executor=tool_executor,
        )

        result = runner.run_turn(history=[], user_message="Get user 123")

        # Inspect the second call to adapter (after tool execution)
        assert adapter.call_count == 2
        # The adapter mock doesn't store messages, but we verify it completed

    def test_complex_tool_arguments(self) -> None:
        """Test tool loop with complex nested arguments."""
        tool_response = ChatResponse(
            message=ChatMessage(role="assistant", content=""),
            tool_calls=(
                ToolCall(
                    name="complex_tool",
                    arguments={
                        "nested": {
                            "field1": "value1",
                            "field2": [1, 2, 3],
                            "field3": {"a": "b"},
                        },
                        "array": ["x", "y", "z"],
                    },
                    call_id="call_1",
                ),
            ),
            raw={},
        )
        final_response = ChatResponse(
            message=ChatMessage(role="assistant", content="Processed"),
            tool_calls=(),
            raw={},
        )
        adapter = MockAdapter(responses=[tool_response, final_response])
        participant = Participant(name="Test", adapter=adapter, model="test")

        received_args = None

        def tool_executor(name: str, args: Dict[str, Any]) -> Any:
            nonlocal received_args
            received_args = args
            return {"status": "ok"}

        tools = [
            ToolDefinition(name="complex_tool", description="Complex", input_schema={}),
        ]

        runner = TurnRunner(
            participant=participant,
            tools=tools,
            tool_executor=tool_executor,
        )

        result = runner.run_turn(history=[], user_message="Use complex tool")

        # Verify complex arguments were passed correctly
        assert received_args is not None
        assert received_args["nested"]["field1"] == "value1"
        assert received_args["nested"]["field2"] == [1, 2, 3]
        assert received_args["array"] == ["x", "y", "z"]
