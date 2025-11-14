"""ConversationRunner: Orchestrate multi-turn conversations between participants."""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence

from ..adapters.base import ChatMessage, ResponseFormat, ToolDefinition, ToolExecutor
from .participant import Participant
from .transcript import ConversationTranscript, ConversationTurn, ToolInvocationRecord
from .turn_runner import TurnRunner


# Type alias for stop condition callback
StopCondition = Callable[[ConversationTranscript], bool]


class ConversationRunner:
    """Orchestrates multi-turn conversations between 2 or more participants.

    This class manages the back-and-forth interaction between participants,
    handling turn-taking, conversation state, and termination conditions.
    """

    def __init__(
        self,
        participants: Sequence[Participant],
        tools: Optional[Sequence[ToolDefinition]] = None,
        tool_executor: Optional[ToolExecutor] = None,
    ) -> None:
        """Initialize a conversation runner.

        Args:
            participants: List of participants (must have at least 2)
            tools: Optional tool definitions available to all participants
            tool_executor: Callable that executes tools (tool_name, args) -> result

        Raises:
            ValueError: If fewer than 2 participants are provided
        """
        if len(participants) < 2:
            raise ValueError(
                f"ConversationRunner requires at least 2 participants, got {len(participants)}"
            )

        self.participants = list(participants)
        self.tools = tuple(tools or ())
        self.tool_executor = tool_executor

        if self.tools and not tool_executor:
            raise ValueError("tool_executor is required when tools are provided")

        # Create turn runners for each participant
        self._turn_runners: dict[str, TurnRunner] = {}
        for participant in self.participants:
            self._turn_runners[participant.name] = TurnRunner(
                participant=participant,
                tools=self.tools,
                tool_executor=self.tool_executor,
            )

    def run(
        self,
        starting_message: str,
        starting_participant: Optional[Participant] = None,
        max_turns: Optional[int] = None,
        stop_condition: Optional[StopCondition] = None,
        initial_transcript: Optional[ConversationTranscript] = None,
        tool_choice: str = "auto",
        response_format: Optional[ResponseFormat] = None,
    ) -> ConversationTranscript:
        """Run a multi-turn conversation between participants.

        Args:
            starting_message: The initial message to start the conversation
            starting_participant: Which participant responds first (defaults to first participant)
            max_turns: Maximum number of turns (None for unlimited)
            stop_condition: Optional callback to check if conversation should stop
            initial_transcript: Optional existing transcript to continue from
            tool_choice: Tool selection strategy ("auto", "required", "none")
            response_format: Optional response formatting (e.g., JSON schema)

        Returns:
            ConversationTranscript with complete conversation history

        Raises:
            ValueError: If starting_participant is not in participants list
        """
        # Initialize or continue from existing transcript
        transcript = initial_transcript or ConversationTranscript()

        # Determine starting participant
        if starting_participant is None:
            current_participant_idx = 0
        else:
            try:
                current_participant_idx = self.participants.index(starting_participant)
            except ValueError:
                raise ValueError(
                    f"starting_participant '{starting_participant.name}' not found in participants"
                )

        # Build conversation history as simple user/assistant pairs
        # Each participant sees previous messages as context
        current_message = starting_message
        turn_count = 0

        while True:
            # Check termination conditions
            if max_turns is not None and turn_count >= max_turns:
                break

            if stop_condition and stop_condition(transcript):
                break

            # Get current participant
            current_participant = self.participants[current_participant_idx]
            turn_runner = self._turn_runners[current_participant.name]

            # Build conversation history for this participant
            # They see all previous turns as user-assistant exchanges
            history = self._build_history_for_participant(transcript)

            # Execute turn
            response = turn_runner.run_turn(
                history=history,
                user_message=current_message,
                max_tool_steps=6,
                tool_choice=tool_choice,
                response_format=response_format,
            )

            # Extract message content
            message_content = response.message.content
            if isinstance(message_content, dict):
                # Handle structured content (e.g., tool calls)
                message_text = str(message_content)
            else:
                message_text = message_content

            # Record tool invocations
            tool_invocations = []
            for tool_call in response.tool_calls:
                # Note: tool results are already handled by turn_runner
                # We're just recording them in the transcript
                tool_invocations.append(
                    ToolInvocationRecord(
                        tool_name=tool_call.name,
                        arguments=tool_call.arguments,
                        result=None,  # Results are embedded in turn execution
                    )
                )

            # Add turn to transcript
            turn = ConversationTurn(
                role=current_participant.name,
                message=message_text,
                tool_invocations=tool_invocations,
            )
            transcript.add_turn(turn)

            # The response becomes the message for the next participant
            current_message = message_text

            # Move to next participant
            current_participant_idx = (current_participant_idx + 1) % len(self.participants)
            turn_count += 1

        return transcript

    def _build_history_for_participant(
        self, transcript: ConversationTranscript
    ) -> List[ChatMessage]:
        """Build conversation history from a participant's perspective.

        Each participant sees the conversation as a series of user-assistant exchanges.
        Previous turns are presented as alternating user/assistant messages.

        Args:
            transcript: The current transcript

        Returns:
            List of ChatMessage objects representing the conversation history
        """
        messages: List[ChatMessage] = []

        for idx, turn in enumerate(transcript.turns):
            # Alternate between user and assistant roles
            # Even indices (0, 2, 4...) are user messages
            # Odd indices (1, 3, 5...) are assistant messages
            role = "user" if idx % 2 == 0 else "assistant"
            messages.append(ChatMessage(role=role, content=turn.message))

        return messages


__all__ = ["ConversationRunner", "StopCondition"]
