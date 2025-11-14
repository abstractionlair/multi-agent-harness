#!/usr/bin/env python3
"""Example usage patterns for Phase 3 orchestration layer.

This demonstrates the intended API design for:
1. Multi-participant conversations (debate pattern)
2. Transcript analysis (judge pattern)
3. Stop conditions
4. Participant injection

Note: These examples show the API but won't make real API calls without valid keys.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def example_debate_pattern():
    """Example: Two models debate for N rounds."""
    print("=" * 60)
    print("Example 1: Debate Pattern (Model A ↔ Model B)")
    print("=" * 60)

    from multi_agent_harness import ConversationRunner, Participant
    from multi_agent_harness.adapters.openai import OpenAIAdapter

    # Create participants with opposing viewpoints
    alice = Participant(
        name="Alice",
        adapter=OpenAIAdapter(api_key="your-api-key"),
        model="gpt-4o-mini",
        system_prompts=["You believe AI will be beneficial to humanity."],
    )

    bob = Participant(
        name="Bob",
        adapter=OpenAIAdapter(api_key="your-api-key"),
        model="gpt-4o-mini",
        system_prompts=["You are skeptical about AI's impact on humanity."],
    )

    # Run debate
    runner = ConversationRunner(participants=[alice, bob])

    # This would run the conversation (commented out to avoid API calls)
    # transcript = runner.run(
    #     starting_message="What will be the impact of AI on humanity?",
    #     starting_participant=alice,
    #     max_turns=4  # 4 rounds of back-and-forth
    # )

    print("Setup complete:")
    print(f"  - Alice: {alice.system_prompts[0].content}")
    print(f"  - Bob: {bob.system_prompts[0].content}")
    print("  - Would run 4 turns of debate")
    print()


def example_judge_pattern():
    """Example: Judge analyzes and scores a transcript."""
    print("=" * 60)
    print("Example 2: Judge Pattern (Model C analyzes transcript)")
    print("=" * 60)

    from multi_agent_harness import (
        ConversationTranscript,
        ConversationTurn,
        Participant,
        ResponseFormat,
        TranscriptAnalyzer,
    )
    from multi_agent_harness.adapters.openai import OpenAIAdapter

    # Create a judge participant
    judge = Participant(
        name="Judge",
        adapter=OpenAIAdapter(api_key="your-api-key"),
        model="gpt-4o",
        system_prompts=[
            "You are an impartial judge.",
            "Score conversations on helpfulness (1-10).",
            'Output JSON: {"score": <number>, "reasoning": "<text>"}',
        ],
    )

    # Create mock transcript
    transcript = ConversationTranscript()
    transcript.add_turn(
        ConversationTurn(
            role="Assistant", message="The capital of France is Paris. Is there anything else?"
        )
    )
    transcript.add_turn(ConversationTurn(role="User", message="Thank you!"))

    # Create analyzer
    analyzer = TranscriptAnalyzer(participant=judge)

    # This would analyze the transcript (commented out to avoid API calls)
    # verdict = analyzer.analyze(
    #     transcript=transcript,
    #     response_format=ResponseFormat(
    #         type="json_schema",
    #         json_schema={
    #             "type": "object",
    #             "properties": {
    #                 "score": {"type": "number"},
    #                 "reasoning": {"type": "string"}
    #             },
    #             "required": ["score", "reasoning"]
    #         }
    #     )
    # )

    print("Setup complete:")
    print(f"  - Judge prompts: {judge.system_prompts[0].content}")
    print(f"  - Transcript has {len(transcript.turns)} turns")
    print("  - Would return JSON with score and reasoning")
    print()


def example_stop_condition():
    """Example: Custom stop condition for conversation."""
    print("=" * 60)
    print("Example 3: Custom Stop Condition")
    print("=" * 60)

    from multi_agent_harness import ConversationRunner, ConversationTranscript, Participant
    from multi_agent_harness.adapters.openai import OpenAIAdapter

    def stop_when_agreement(transcript: ConversationTranscript) -> bool:
        """Stop when participants agree."""
        if len(transcript.turns) < 2:
            return False

        # Check if last turn contains "I agree"
        last_message = transcript.turns[-1].message.lower()
        return "i agree" in last_message or "you're right" in last_message

    alice = Participant(
        name="Alice",
        adapter=OpenAIAdapter(api_key="your-api-key"),
        model="gpt-4o-mini",
        system_prompts=["You are Alice."],
    )

    bob = Participant(
        name="Bob",
        adapter=OpenAIAdapter(api_key="your-api-key"),
        model="gpt-4o-mini",
        system_prompts=["You are Bob."],
    )

    runner = ConversationRunner(participants=[alice, bob])

    # This would run until agreement (commented out to avoid API calls)
    # transcript = runner.run(
    #     starting_message="Should we use tabs or spaces?",
    #     starting_participant=alice,
    #     max_turns=10,  # Safety limit
    #     stop_condition=stop_when_agreement
    # )

    print("Setup complete:")
    print("  - Stop condition: will stop when participants agree")
    print("  - Max turns: 10 (safety limit)")
    print()


def example_participant_injection():
    """Example: Inject new participant mid-conversation."""
    print("=" * 60)
    print("Example 4: Participant Injection")
    print("=" * 60)

    from multi_agent_harness import (
        ConversationRunner,
        ConversationTranscript,
        ConversationTurn,
        Participant,
        TranscriptAnalyzer,
    )
    from multi_agent_harness.adapters.openai import OpenAIAdapter

    # First conversation: Alice and Bob
    alice = Participant(
        name="Alice",
        adapter=OpenAIAdapter(api_key="your-api-key"),
        model="gpt-4o-mini",
        system_prompts=["You are Alice."],
    )

    bob = Participant(
        name="Bob",
        adapter=OpenAIAdapter(api_key="your-api-key"),
        model="gpt-4o-mini",
        system_prompts=["You are Bob."],
    )

    # Simulate first conversation
    initial_transcript = ConversationTranscript()
    initial_transcript.add_turn(ConversationTurn(role="Alice", message="I think X is better."))
    initial_transcript.add_turn(ConversationTurn(role="Bob", message="I disagree, Y is better."))
    initial_transcript.add_turn(ConversationTurn(role="Alice", message="But X has advantages."))

    # Charlie analyzes the conversation
    charlie = Participant(
        name="Charlie",
        adapter=OpenAIAdapter(api_key="your-api-key"),
        model="gpt-4o",
        system_prompts=["You are a moderator. Summarize key disagreements."],
    )

    analyzer = TranscriptAnalyzer(participant=charlie)

    # This would analyze (commented out to avoid API calls)
    # summary = analyzer.analyze(initial_transcript)

    # Charlie replaces Bob and continues
    runner2 = ConversationRunner(participants=[alice, charlie])

    # This would continue conversation (commented out to avoid API calls)
    # continued = runner2.run(
    #     starting_message="Let me summarize the discussion so far...",
    #     starting_participant=charlie,
    #     initial_transcript=initial_transcript,  # Carry forward history
    #     max_turns=3
    # )

    print("Setup complete:")
    print("  - Initial conversation: Alice ↔ Bob (3 turns)")
    print("  - Charlie analyzes transcript")
    print("  - Charlie replaces Bob and continues conversation")
    print("  - Carries forward initial transcript as context")
    print()


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("Phase 3 Orchestration Layer - Example Patterns")
    print("=" * 60)
    print()

    example_debate_pattern()
    example_judge_pattern()
    example_stop_condition()
    example_participant_injection()

    print("=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
    print()
    print("Note: These examples demonstrate API usage but don't make")
    print("real API calls. To run with real models, add valid API keys.")


if __name__ == "__main__":
    main()
