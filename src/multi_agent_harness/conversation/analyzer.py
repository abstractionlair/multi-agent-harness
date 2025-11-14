"""TranscriptAnalyzer: Helper for analyzing conversation transcripts with a model."""

from __future__ import annotations

from typing import Optional

from ..adapters.base import ChatMessage, ChatResponse, ResponseFormat
from .participant import Participant
from .transcript import ConversationTranscript


class TranscriptAnalyzer:
    """Analyzes conversation transcripts using a participant (model + prompt).

    This class enables patterns where a model examines a conversation transcript
    and produces analysis, scoring, summarization, or follow-up questions.

    Common use cases:
    - Judge: Score conversation quality, helpfulness, correctness
    - Summarizer: Extract key points from a conversation
    - Interrogator: Generate follow-up questions based on transcript
    - Moderator: Identify disagreements or issues in multi-party conversations
    """

    def __init__(self, participant: Participant) -> None:
        """Initialize a transcript analyzer.

        Args:
            participant: The participant (model + prompts) that will analyze transcripts
        """
        self.participant = participant

    def analyze(
        self,
        transcript: ConversationTranscript,
        analysis_prompt: Optional[str] = None,
        response_format: Optional[ResponseFormat] = None,
    ) -> ChatResponse:
        """Analyze a conversation transcript and produce a response.

        Args:
            transcript: The conversation transcript to analyze
            analysis_prompt: Optional specific prompt for this analysis
                           (in addition to participant's system prompts)
            response_format: Optional response formatting (e.g., JSON schema for structured output)

        Returns:
            ChatResponse containing the analysis

        Example:
            ```python
            judge = Participant(
                name="Judge",
                adapter=OpenAIAdapter(),
                model="gpt-4o",
                system_prompts=[
                    "You are an impartial judge.",
                    "Score conversations on helpfulness (1-10).",
                    "Output JSON: {\"score\": <number>, \"reasoning\": \"<text>\"}"
                ]
            )

            analyzer = TranscriptAnalyzer(participant=judge)
            verdict = analyzer.analyze(
                transcript=conversation_transcript,
                response_format=ResponseFormat(
                    type="json_schema",
                    json_schema={
                        "type": "object",
                        "properties": {
                            "score": {"type": "number"},
                            "reasoning": {"type": "string"}
                        },
                        "required": ["score", "reasoning"]
                    }
                )
            )

            import json
            result = json.loads(verdict.message.content)
            print(f"Score: {result['score']}, Reasoning: {result['reasoning']}")
            ```
        """
        # Build the transcript as a formatted string
        transcript_text = self._format_transcript(transcript)

        # Build message list for the analysis
        messages = list(self.participant.system_prompts)

        # Add the transcript as context
        if analysis_prompt:
            # Combine transcript with specific analysis prompt
            combined_prompt = f"{analysis_prompt}\n\n--- CONVERSATION TRANSCRIPT ---\n{transcript_text}"
            messages.append(ChatMessage(role="user", content=combined_prompt))
        else:
            # Just provide the transcript
            messages.append(
                ChatMessage(
                    role="user",
                    content=f"Please analyze the following conversation:\n\n{transcript_text}",
                )
            )

        # Send to model
        response = self.participant.adapter.send_chat(
            role_config=self.participant.model_config,
            messages=messages,
            response_format=response_format,
        )

        return response

    def _format_transcript(self, transcript: ConversationTranscript) -> str:
        """Format a transcript as a readable string.

        Args:
            transcript: The transcript to format

        Returns:
            Formatted transcript string with turn numbers and participant names
        """
        lines = []

        for idx, turn in enumerate(transcript.turns, 1):
            lines.append(f"Turn {idx} ({turn.role}):")
            lines.append(f"  {turn.message}")

            # Include tool invocations if present
            if turn.tool_invocations:
                lines.append("  Tool Calls:")
                for tool_inv in turn.tool_invocations:
                    lines.append(f"    - {tool_inv.tool_name}: {tool_inv.arguments}")
                    if tool_inv.result is not None:
                        lines.append(f"      Result: {tool_inv.result}")
                    if tool_inv.error is not None:
                        lines.append(f"      Error: {tool_inv.error}")

            lines.append("")  # Blank line between turns

        return "\n".join(lines)


__all__ = ["TranscriptAnalyzer"]
