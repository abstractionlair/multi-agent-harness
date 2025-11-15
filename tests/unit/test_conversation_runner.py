"""Unit tests for ConversationRunner state management."""

from __future__ import annotations

from typing import Any

import pytest

from multi_agent_harness.adapters.base import (
    ChatMessage,
    ChatResponse,
    ProviderAdapter,
    ResponseFormat,
    ToolCall,
    ToolDefinition,
)
from multi_agent_harness.config import RoleModelConfig
from multi_agent_harness.conversation.conversation_runner import ConversationRunner
from multi_agent_harness.conversation.participant import Participant
from multi_agent_harness.conversation.transcript import (
    ConversationTranscript,
    ConversationTurn,
)


class MockAdapter(ProviderAdapter):
    """Mock provider adapter for testing."""

    provider_name = "mock"

    def __init__(self, responses: list[ChatResponse] | None = None) -> None:
        super().__init__()
        self.responses = responses or []
        self.call_count = 0
        self.calls: list[dict[str, Any]] = []

    def send_chat(
        self,
        role_config: RoleModelConfig,
        messages: list[ChatMessage],
        tools: list[ToolDefinition] | None = None,
        response_format: ResponseFormat | None = None,
        tool_choice: str | None = None,
    ) -> ChatResponse:
        self.calls.append(
            {
                "role_config": role_config,
                "messages": messages,
                "tools": tools,
                "response_format": response_format,
                "tool_choice": tool_choice,
            }
        )

        if self.call_count < len(self.responses):
            response = self.responses[self.call_count]
            self.call_count += 1
            return response

        return ChatResponse(
            message=ChatMessage(role="assistant", content=f"Response {self.call_count}"),
            tool_calls=(),
            raw={},
        )

    def supports_tools(self) -> bool:
        return True


class TestConversationRunnerInitialization:
    """Test ConversationRunner initialization."""

    def test_init_with_two_participants(self) -> None:
        """Test initialization with minimum required participants."""
        adapter1 = MockAdapter()
        adapter2 = MockAdapter()
        participant1 = Participant(name="Alice", adapter=adapter1, model="model1")
        participant2 = Participant(name="Bob", adapter=adapter2, model="model2")

        runner = ConversationRunner(participants=[participant1, participant2])

        assert len(runner.participants) == 2
        assert len(runner._turn_runners) == 2

    def test_init_with_single_participant_raises(self) -> None:
        """Test that initialization with fewer than 2 participants raises ValueError."""
        adapter = MockAdapter()
        participant = Participant(name="Alice", adapter=adapter, model="model1")

        with pytest.raises(ValueError, match="at least 2 participants"):
            ConversationRunner(participants=[participant])

    def test_init_with_empty_participants_raises(self) -> None:
        """Test that initialization with empty participants raises ValueError."""
        with pytest.raises(ValueError, match="at least 2 participants"):
            ConversationRunner(participants=[])

    def test_init_with_tools_but_no_executor_raises(self) -> None:
        """Test that providing tools without executor raises ValueError."""
        adapter1 = MockAdapter()
        adapter2 = MockAdapter()
        participant1 = Participant(name="Alice", adapter=adapter1, model="model1")
        participant2 = Participant(name="Bob", adapter=adapter2, model="model2")

        tools = [
            ToolDefinition(name="test_tool", description="Test", input_schema={}),
        ]

        with pytest.raises(ValueError, match="tool_executor is required"):
            ConversationRunner(participants=[participant1, participant2], tools=tools)

    def test_init_with_tools_and_executor(self) -> None:
        """Test initialization with tools and executor."""
        adapter1 = MockAdapter()
        adapter2 = MockAdapter()
        participant1 = Participant(name="Alice", adapter=adapter1, model="model1")
        participant2 = Participant(name="Bob", adapter=adapter2, model="model2")

        tools = [
            ToolDefinition(name="test_tool", description="Test", input_schema={}),
        ]

        def executor(name: str, args: dict[str, Any]) -> Any:
            return {"result": "ok"}

        runner = ConversationRunner(
            participants=[participant1, participant2],
            tools=tools,
            tool_executor=executor,
        )

        assert len(runner.tools) == 1
        assert runner.tool_executor is not None


class TestConversationRunnerExecution:
    """Test ConversationRunner execution and state management."""

    def test_simple_two_turn_conversation(self) -> None:
        """Test a simple two-turn conversation."""
        responses = [
            ChatResponse(
                message=ChatMessage(role="assistant", content="Hello Bob!"),
                tool_calls=(),
                raw={},
            ),
            ChatResponse(
                message=ChatMessage(role="assistant", content="Hello Alice!"),
                tool_calls=(),
                raw={},
            ),
        ]
        adapter1 = MockAdapter(responses=[responses[0]])
        adapter2 = MockAdapter(responses=[responses[1]])
        participant1 = Participant(name="Alice", adapter=adapter1, model="model1")
        participant2 = Participant(name="Bob", adapter=adapter2, model="model2")

        runner = ConversationRunner(participants=[participant1, participant2])

        transcript = runner.run(
            starting_message="Start the conversation",
            max_turns=2,
        )

        assert len(transcript.turns) == 2
        assert transcript.turns[0].role == "Alice"
        assert transcript.turns[0].message == "Hello Bob!"
        assert transcript.turns[1].role == "Bob"
        assert transcript.turns[1].message == "Hello Alice!"

    def test_conversation_with_starting_participant(self) -> None:
        """Test conversation starting with specific participant."""
        responses = [
            ChatResponse(
                message=ChatMessage(role="assistant", content="Bob starts"),
                tool_calls=(),
                raw={},
            ),
            ChatResponse(
                message=ChatMessage(role="assistant", content="Alice responds"),
                tool_calls=(),
                raw={},
            ),
        ]
        adapter1 = MockAdapter(responses=[responses[1]])
        adapter2 = MockAdapter(responses=[responses[0]])
        participant1 = Participant(name="Alice", adapter=adapter1, model="model1")
        participant2 = Participant(name="Bob", adapter=adapter2, model="model2")

        runner = ConversationRunner(participants=[participant1, participant2])

        transcript = runner.run(
            starting_message="Go!",
            starting_participant=participant2,  # Bob starts
            max_turns=2,
        )

        assert len(transcript.turns) == 2
        assert transcript.turns[0].role == "Bob"
        assert transcript.turns[1].role == "Alice"

    def test_conversation_with_invalid_starting_participant_raises(self) -> None:
        """Test that invalid starting participant raises ValueError."""
        adapter1 = MockAdapter()
        adapter2 = MockAdapter()
        adapter3 = MockAdapter()
        participant1 = Participant(name="Alice", adapter=adapter1, model="model1")
        participant2 = Participant(name="Bob", adapter=adapter2, model="model2")
        participant3 = Participant(name="Charlie", adapter=adapter3, model="model3")

        runner = ConversationRunner(participants=[participant1, participant2])

        with pytest.raises(ValueError, match="not found in participants"):
            runner.run(
                starting_message="Go!",
                starting_participant=participant3,  # Not in runner's participants
                max_turns=1,
            )

    def test_conversation_with_max_turns(self) -> None:
        """Test that conversation stops at max_turns."""
        # Create enough responses for many turns
        responses_alice = [
            ChatResponse(
                message=ChatMessage(role="assistant", content=f"Alice {i}"),
                tool_calls=(),
                raw={},
            )
            for i in range(10)
        ]
        responses_bob = [
            ChatResponse(
                message=ChatMessage(role="assistant", content=f"Bob {i}"),
                tool_calls=(),
                raw={},
            )
            for i in range(10)
        ]

        adapter1 = MockAdapter(responses=responses_alice)
        adapter2 = MockAdapter(responses=responses_bob)
        participant1 = Participant(name="Alice", adapter=adapter1, model="model1")
        participant2 = Participant(name="Bob", adapter=adapter2, model="model2")

        runner = ConversationRunner(participants=[participant1, participant2])

        transcript = runner.run(
            starting_message="Start",
            max_turns=5,
        )

        assert len(transcript.turns) == 5

    def test_conversation_with_stop_condition(self) -> None:
        """Test conversation stopping on custom condition."""
        responses_alice = [
            ChatResponse(
                message=ChatMessage(role="assistant", content="Turn 1"),
                tool_calls=(),
                raw={},
            ),
            ChatResponse(
                message=ChatMessage(role="assistant", content="Turn 3"),
                tool_calls=(),
                raw={},
            ),
        ]
        responses_bob = [
            ChatResponse(
                message=ChatMessage(role="assistant", content="STOP"),
                tool_calls=(),
                raw={},
            ),
        ]

        adapter1 = MockAdapter(responses=responses_alice)
        adapter2 = MockAdapter(responses=responses_bob)
        participant1 = Participant(name="Alice", adapter=adapter1, model="model1")
        participant2 = Participant(name="Bob", adapter=adapter2, model="model2")

        runner = ConversationRunner(participants=[participant1, participant2])

        def stop_condition(transcript: ConversationTranscript) -> bool:
            # Stop if any message contains "STOP"
            return any("STOP" in turn.message for turn in transcript.turns)

        transcript = runner.run(
            starting_message="Start",
            max_turns=10,
            stop_condition=stop_condition,
        )

        # Should stop after Bob says "STOP"
        assert len(transcript.turns) == 2
        assert transcript.turns[1].message == "STOP"

    def test_conversation_with_initial_transcript(self) -> None:
        """Test continuing conversation from existing transcript."""
        # Create initial transcript
        initial_transcript = ConversationTranscript()
        initial_transcript.add_turn(
            ConversationTurn(role="Alice", message="Previous message 1", tool_invocations=[])
        )
        initial_transcript.add_turn(
            ConversationTurn(role="Bob", message="Previous message 2", tool_invocations=[])
        )

        responses = [
            ChatResponse(
                message=ChatMessage(role="assistant", content="New message"),
                tool_calls=(),
                raw={},
            ),
        ]
        adapter1 = MockAdapter(responses=responses)
        adapter2 = MockAdapter()
        participant1 = Participant(name="Alice", adapter=adapter1, model="model1")
        participant2 = Participant(name="Bob", adapter=adapter2, model="model2")

        runner = ConversationRunner(participants=[participant1, participant2])

        transcript = runner.run(
            starting_message="Continue",
            initial_transcript=initial_transcript,
            max_turns=1,
        )

        # Should have initial 2 turns + 1 new turn
        assert len(transcript.turns) == 3
        assert transcript.turns[0].message == "Previous message 1"
        assert transcript.turns[1].message == "Previous message 2"
        assert transcript.turns[2].message == "New message"

    def test_three_participant_conversation(self) -> None:
        """Test conversation with three participants rotating."""
        responses_alice = [
            ChatResponse(
                message=ChatMessage(role="assistant", content="Alice turn 1"),
                tool_calls=(),
                raw={},
            ),
        ]
        responses_bob = [
            ChatResponse(
                message=ChatMessage(role="assistant", content="Bob turn 1"),
                tool_calls=(),
                raw={},
            ),
        ]
        responses_charlie = [
            ChatResponse(
                message=ChatMessage(role="assistant", content="Charlie turn 1"),
                tool_calls=(),
                raw={},
            ),
        ]

        adapter1 = MockAdapter(responses=responses_alice)
        adapter2 = MockAdapter(responses=responses_bob)
        adapter3 = MockAdapter(responses=responses_charlie)
        participant1 = Participant(name="Alice", adapter=adapter1, model="model1")
        participant2 = Participant(name="Bob", adapter=adapter2, model="model2")
        participant3 = Participant(name="Charlie", adapter=adapter3, model="model3")

        runner = ConversationRunner(
            participants=[participant1, participant2, participant3]
        )

        transcript = runner.run(
            starting_message="Start",
            max_turns=3,
        )

        assert len(transcript.turns) == 3
        assert transcript.turns[0].role == "Alice"
        assert transcript.turns[1].role == "Bob"
        assert transcript.turns[2].role == "Charlie"

    def test_conversation_history_building(self) -> None:
        """Test that conversation history is correctly built for each participant."""
        responses = [
            ChatResponse(
                message=ChatMessage(role="assistant", content="Alice 1"),
                tool_calls=(),
                raw={},
            ),
            ChatResponse(
                message=ChatMessage(role="assistant", content="Bob 1"),
                tool_calls=(),
                raw={},
            ),
            ChatResponse(
                message=ChatMessage(role="assistant", content="Alice 2"),
                tool_calls=(),
                raw={},
            ),
        ]
        adapter1 = MockAdapter(responses=[responses[0], responses[2]])
        adapter2 = MockAdapter(responses=[responses[1]])
        participant1 = Participant(
            name="Alice",
            adapter=adapter1,
            model="model1",
            system_prompts=["You are Alice"],
        )
        participant2 = Participant(
            name="Bob",
            adapter=adapter2,
            model="model2",
            system_prompts=["You are Bob"],
        )

        runner = ConversationRunner(participants=[participant1, participant2])

        runner.run(
            starting_message="Start",
            max_turns=3,
        )

        # Check that Alice's second call includes history
        assert adapter1.call_count == 2
        alice_second_call = adapter1.calls[1]
        messages = alice_second_call["messages"]

        # Should have: system prompt + starting message + Alice 1 + Bob 1 + new user message
        assert len(messages) == 5
        assert messages[0].role == "system"
        assert messages[1].role == "user"  # starting message
        assert messages[2].role == "user"  # Alice 1 (as user from Bob's perspective)
        assert messages[3].role == "assistant"  # Bob 1
        assert messages[4].role == "user"  # Bob 1 (as user for Alice's turn)

    def test_conversation_with_tools_records_invocations(self) -> None:
        """Test that tool invocations are recorded in transcript."""
        tool_call_response = ChatResponse(
            message=ChatMessage(role="assistant", content=""),
            tool_calls=(
                ToolCall(name="test_tool", arguments={"x": 1}, call_id="call_1"),
            ),
            raw={},
        )
        final_response = ChatResponse(
            message=ChatMessage(role="assistant", content="Done with tool"),
            tool_calls=(),
            raw={},
        )

        adapter1 = MockAdapter(responses=[tool_call_response, final_response])
        adapter2 = MockAdapter()
        participant1 = Participant(name="Alice", adapter=adapter1, model="model1")
        participant2 = Participant(name="Bob", adapter=adapter2, model="model2")

        def tool_executor(name: str, args: dict[str, Any]) -> Any:
            return {"result": "ok"}

        tools = [
            ToolDefinition(name="test_tool", description="Test", input_schema={}),
        ]

        runner = ConversationRunner(
            participants=[participant1, participant2],
            tools=tools,
            tool_executor=tool_executor,
        )

        transcript = runner.run(
            starting_message="Use the tool",
            max_turns=1,
        )

        # Alice's turn should have tool invocations recorded
        assert len(transcript.turns) == 1
        assert len(transcript.turns[0].tool_invocations) == 1
        assert transcript.turns[0].tool_invocations[0].tool_name == "test_tool"
        assert transcript.turns[0].tool_invocations[0].arguments == {"x": 1}

    def test_unlimited_turns_with_stop_condition(self) -> None:
        """Test conversation with no max_turns but with stop condition."""
        responses_alice = [
            ChatResponse(
                message=ChatMessage(role="assistant", content=f"Message {i}"),
                tool_calls=(),
                raw={},
            )
            for i in range(3)
        ]
        responses_bob = [
            ChatResponse(
                message=ChatMessage(role="assistant", content=f"Message {i}"),
                tool_calls=(),
                raw={},
            )
            for i in range(2)
        ] + [
            ChatResponse(
                message=ChatMessage(role="assistant", content="DONE"),
                tool_calls=(),
                raw={},
            ),
        ]

        adapter1 = MockAdapter(responses=responses_alice)
        adapter2 = MockAdapter(responses=responses_bob)
        participant1 = Participant(name="Alice", adapter=adapter1, model="model1")
        participant2 = Participant(name="Bob", adapter=adapter2, model="model2")

        runner = ConversationRunner(participants=[participant1, participant2])

        def stop_condition(transcript: ConversationTranscript) -> bool:
            return any("DONE" in turn.message for turn in transcript.turns)

        transcript = runner.run(
            starting_message="Start",
            max_turns=None,  # Unlimited
            stop_condition=stop_condition,
        )

        # Should continue until Bob says "DONE"
        assert len(transcript.turns) == 6  # 3 Alice + 3 Bob
        assert transcript.turns[-1].message == "DONE"
