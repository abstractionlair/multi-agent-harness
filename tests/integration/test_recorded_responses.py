"""Integration tests using recorded API responses.

These tests use pre-recorded API responses to verify end-to-end behavior
without making actual network calls.
"""

import json
from typing import Any

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
from multi_agent_harness.conversation.conversation_runner import ConversationRunner


class RecordedAdapter(ProviderAdapter):
    """Adapter that replays recorded API responses."""

    provider_name = "recorded"

    def __init__(self, recorded_responses: list[dict[str, Any]]) -> None:
        super().__init__()
        self.recorded_responses = recorded_responses
        self.call_index = 0

    def send_chat(
        self,
        role_config: RoleModelConfig,
        messages: list[ChatMessage],
        tools: list[ToolDefinition] | None = None,
        response_format: ResponseFormat | None = None,
        tool_choice: str | None = None,
    ) -> ChatResponse:
        if self.call_index >= len(self.recorded_responses):
            raise RuntimeError(
                f"No more recorded responses (call {self.call_index + 1})"
            )

        response_data = self.recorded_responses[self.call_index]
        self.call_index += 1

        # Parse recorded response
        message_content = response_data.get("content", "")
        tool_calls_data = response_data.get("tool_calls", [])

        tool_calls = tuple(
            ToolCall(
                name=tc["name"],
                arguments=tc["arguments"],
                call_id=tc.get("call_id"),
            )
            for tc in tool_calls_data
        )

        return ChatResponse(
            message=ChatMessage(role="assistant", content=message_content),
            tool_calls=tool_calls,
            raw=response_data,
        )

    def supports_tools(self) -> bool:
        return True


class TestRecordedSimpleConversation:
    """Test simple conversation with recorded responses."""

    def test_single_turn_conversation(self) -> None:
        """Test a single turn conversation with recorded response."""
        # Recorded response from a real API call
        recorded_responses = [
            {
                "content": "Hello! I'm doing great, thank you for asking. How can I help you today?",
                "tool_calls": [],
            }
        ]

        adapter = RecordedAdapter(recorded_responses)
        participant = Participant(
            name="Assistant",
            adapter=adapter,
            model="gpt-4o-mini",
            system_prompts=["You are a helpful assistant."],
        )

        runner = TurnRunner(participant=participant)
        response = runner.run_turn(
            history=[],
            user_message="Hello! How are you?",
        )

        assert "Hello!" in response.message.content
        assert len(response.tool_calls) == 0

    def test_multi_turn_conversation(self) -> None:
        """Test multi-turn conversation with recorded responses."""
        # Simulate a back-and-forth conversation
        alice_responses = [
            {"content": "Hi Bob! The weather is nice today.", "tool_calls": []},
            {"content": "I like sunny days the best!", "tool_calls": []},
        ]
        bob_responses = [
            {"content": "Hi Alice! Yes, it's beautiful out.", "tool_calls": []},
            {"content": "Me too! Perfect for a walk.", "tool_calls": []},
        ]

        alice_adapter = RecordedAdapter(alice_responses)
        bob_adapter = RecordedAdapter(bob_responses)

        alice = Participant(
            name="Alice",
            adapter=alice_adapter,
            model="gpt-4o-mini",
        )
        bob = Participant(
            name="Bob",
            adapter=bob_adapter,
            model="gpt-4o-mini",
        )

        runner = ConversationRunner(participants=[alice, bob])
        transcript = runner.run(
            starting_message="Start conversation",
            max_turns=4,
        )

        assert len(transcript.turns) == 4
        assert "weather" in transcript.turns[0].message.lower()
        assert "beautiful" in transcript.turns[1].message.lower()


class TestRecordedToolCalling:
    """Test tool calling with recorded responses."""

    def test_simple_tool_call_and_response(self) -> None:
        """Test a simple tool call with recorded responses."""
        # Recorded responses: tool call, then final answer
        recorded_responses = [
            {
                "content": "",
                "tool_calls": [
                    {
                        "name": "get_weather",
                        "arguments": {"location": "San Francisco"},
                        "call_id": "call_abc123",
                    }
                ],
            },
            {
                "content": "The weather in San Francisco is currently 68°F and sunny.",
                "tool_calls": [],
            },
        ]

        adapter = RecordedAdapter(recorded_responses)
        participant = Participant(
            name="Assistant",
            adapter=adapter,
            model="gpt-4o-mini",
            system_prompts=["You are a weather assistant."],
        )

        # Mock tool executor
        def tool_executor(name: str, args: dict[str, Any]) -> Any:
            if name == "get_weather":
                return {
                    "temperature": 68,
                    "conditions": "sunny",
                    "location": args["location"],
                }
            raise ValueError(f"Unknown tool: {name}")

        tools = [
            ToolDefinition(
                name="get_weather",
                description="Get current weather",
                input_schema={
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"},
                    },
                    "required": ["location"],
                },
            )
        ]

        runner = TurnRunner(
            participant=participant,
            tools=tools,
            tool_executor=tool_executor,
        )

        response = runner.run_turn(
            history=[],
            user_message="What's the weather in San Francisco?",
        )

        assert "68" in response.message.content
        assert "sunny" in response.message.content.lower()

    def test_multiple_sequential_tool_calls(self) -> None:
        """Test multiple sequential tool calls with recorded responses."""
        # Recorded: call tool1, call tool2, final answer
        recorded_responses = [
            {
                "content": "",
                "tool_calls": [
                    {
                        "name": "calculate",
                        "arguments": {"operation": "add", "a": 15, "b": 27},
                        "call_id": "call_1",
                    }
                ],
            },
            {
                "content": "",
                "tool_calls": [
                    {
                        "name": "calculate",
                        "arguments": {"operation": "multiply", "a": 42, "b": 2},
                        "call_id": "call_2",
                    }
                ],
            },
            {
                "content": "First, 15 + 27 = 42. Then, 42 × 2 = 84.",
                "tool_calls": [],
            },
        ]

        adapter = RecordedAdapter(recorded_responses)
        participant = Participant(
            name="Calculator",
            adapter=adapter,
            model="gpt-4o-mini",
        )

        def tool_executor(name: str, args: dict[str, Any]) -> Any:
            if name == "calculate":
                op = args["operation"]
                a = args["a"]
                b = args["b"]
                if op == "add":
                    return {"result": a + b}
                elif op == "multiply":
                    return {"result": a * b}
            raise ValueError("Unknown operation")

        tools = [
            ToolDefinition(
                name="calculate",
                description="Perform calculation",
                input_schema={"type": "object"},
            )
        ]

        runner = TurnRunner(
            participant=participant,
            tools=tools,
            tool_executor=tool_executor,
        )

        response = runner.run_turn(
            history=[],
            user_message="What is (15 + 27) * 2?",
        )

        assert "42" in response.message.content
        assert "84" in response.message.content

    def test_parallel_tool_calls(self) -> None:
        """Test multiple parallel tool calls in one response."""
        # Recorded: multiple tools in one call, then final answer
        recorded_responses = [
            {
                "content": "",
                "tool_calls": [
                    {
                        "name": "get_weather",
                        "arguments": {"location": "New York"},
                        "call_id": "call_1",
                    },
                    {
                        "name": "get_weather",
                        "arguments": {"location": "London"},
                        "call_id": "call_2",
                    },
                    {
                        "name": "get_weather",
                        "arguments": {"location": "Tokyo"},
                        "call_id": "call_3",
                    },
                ],
            },
            {
                "content": "New York: 72°F, London: 15°C, Tokyo: 25°C",
                "tool_calls": [],
            },
        ]

        adapter = RecordedAdapter(recorded_responses)
        participant = Participant(
            name="WeatherBot",
            adapter=adapter,
            model="gpt-4o-mini",
        )

        weather_data = {
            "New York": {"temp": 72, "unit": "F"},
            "London": {"temp": 15, "unit": "C"},
            "Tokyo": {"temp": 25, "unit": "C"},
        }

        def tool_executor(name: str, args: dict[str, Any]) -> Any:
            location = args["location"]
            return weather_data.get(location, {"temp": 0, "unit": "C"})

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

        response = runner.run_turn(
            history=[],
            user_message="What's the weather in New York, London, and Tokyo?",
        )

        assert "New York" in response.message.content
        assert "London" in response.message.content
        assert "Tokyo" in response.message.content


class TestRecordedComplexScenarios:
    """Test complex scenarios with recorded responses."""

    def test_conversation_with_tools(self) -> None:
        """Test conversation between two participants with tools."""
        # Alice asks a question
        alice_responses = [
            {"content": "Can you check the weather in Paris for me?", "tool_calls": []}
        ]

        # Bob uses tools and responds
        bob_responses = [
            {
                "content": "",
                "tool_calls": [
                    {
                        "name": "get_weather",
                        "arguments": {"location": "Paris"},
                        "call_id": "call_x",
                    }
                ],
            },
            {
                "content": "The weather in Paris is 18°C and partly cloudy.",
                "tool_calls": [],
            },
        ]

        alice_adapter = RecordedAdapter(alice_responses)
        bob_adapter = RecordedAdapter(bob_responses)

        alice = Participant(name="Alice", adapter=alice_adapter, model="gpt-4o-mini")
        bob = Participant(name="Bob", adapter=bob_adapter, model="gpt-4o-mini")

        def tool_executor(name: str, args: dict[str, Any]) -> Any:
            return {"temperature": 18, "conditions": "partly cloudy"}

        tools = [
            ToolDefinition(
                name="get_weather",
                description="Get weather",
                input_schema={"type": "object"},
            )
        ]

        runner = ConversationRunner(
            participants=[alice, bob],
            tools=tools,
            tool_executor=tool_executor,
        )

        transcript = runner.run(
            starting_message="Hello",
            max_turns=2,
        )

        assert len(transcript.turns) == 2
        assert "Paris" in transcript.turns[0].message
        assert "18" in transcript.turns[1].message

    def test_json_response_format(self) -> None:
        """Test structured JSON response with recorded data."""
        recorded_responses = [
            {
                "content": '{"score": 85, "feedback": "Good work!", "areas_to_improve": ["testing", "documentation"]}',
                "tool_calls": [],
            }
        ]

        adapter = RecordedAdapter(recorded_responses)
        participant = Participant(
            name="Judge",
            adapter=adapter,
            model="gpt-4o-mini",
            system_prompts=["You are a code reviewer. Return JSON."],
        )

        runner = TurnRunner(participant=participant)

        response_format = ResponseFormat(
            type="json_schema",
            json_schema={
                "type": "object",
                "properties": {
                    "score": {"type": "number"},
                    "feedback": {"type": "string"},
                    "areas_to_improve": {"type": "array", "items": {"type": "string"}},
                },
            },
        )

        response = runner.run_turn(
            history=[],
            user_message="Review this code: def add(a, b): return a + b",
            response_format=response_format,
        )

        # Verify JSON response
        data = json.loads(response.message.content)
        assert data["score"] == 85
        assert "Good work!" in data["feedback"]
        assert "testing" in data["areas_to_improve"]
