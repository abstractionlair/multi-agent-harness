#!/usr/bin/env python3
"""Verification script for Phase 3: Orchestration Layer.

This script demonstrates the new ConversationRunner and TranscriptAnalyzer functionality.
It doesn't make real API calls but verifies imports and basic instantiation.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def verify_imports():
    """Verify all new Phase 3 classes can be imported."""
    print("🔍 Verifying imports...")

    try:
        from multi_agent_harness import (
            ConversationRunner,
            ConversationTranscript,
            Participant,
            StopCondition,
            TranscriptAnalyzer,
            TurnRunner,
        )
        from multi_agent_harness.adapters.openai import OpenAIAdapter

        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def verify_conversation_runner():
    """Verify ConversationRunner can be instantiated."""
    print("\n🔍 Verifying ConversationRunner...")

    try:
        from multi_agent_harness import ConversationRunner, Participant
        from multi_agent_harness.adapters.openai import OpenAIAdapter

        # Create mock participants (no real API calls)
        adapter = OpenAIAdapter(api_key="test-key")
        alice = Participant(
            name="Alice",
            adapter=adapter,
            model="gpt-4o-mini",
            system_prompts=["You are a helpful assistant named Alice."],
        )

        bob = Participant(
            name="Bob",
            adapter=adapter,
            model="gpt-4o-mini",
            system_prompts=["You are Bob. You like to ask questions."],
        )

        # Create runner
        runner = ConversationRunner(participants=[alice, bob])

        print("✅ ConversationRunner instantiated successfully")
        print(f"   - Participants: {[p.name for p in runner.participants]}")
        print(f"   - Turn runners: {list(runner._turn_runners.keys())}")
        return True
    except Exception as e:
        print(f"❌ ConversationRunner verification failed: {e}")
        return False


def verify_transcript_analyzer():
    """Verify TranscriptAnalyzer can be instantiated."""
    print("\n🔍 Verifying TranscriptAnalyzer...")

    try:
        from multi_agent_harness import (
            ConversationTranscript,
            ConversationTurn,
            Participant,
            TranscriptAnalyzer,
        )
        from multi_agent_harness.adapters.openai import OpenAIAdapter

        # Create mock participant
        adapter = OpenAIAdapter(api_key="test-key")
        judge = Participant(
            name="Judge",
            adapter=adapter,
            model="gpt-4o",
            system_prompts=[
                "You are an impartial judge.",
                "Score conversations on helpfulness (1-10).",
            ],
        )

        # Create analyzer
        analyzer = TranscriptAnalyzer(participant=judge)

        # Create a mock transcript
        transcript = ConversationTranscript()
        transcript.add_turn(
            ConversationTurn(role="Alice", message="What's the capital of France?")
        )
        transcript.add_turn(ConversationTurn(role="Bob", message="Paris!"))

        # Test transcript formatting
        formatted = analyzer._format_transcript(transcript)

        print("✅ TranscriptAnalyzer instantiated successfully")
        print(f"   - Participant: {analyzer.participant.name}")
        print(f"   - Formatted transcript preview:")
        for line in formatted.split("\n")[:5]:
            print(f"     {line}")
        return True
    except Exception as e:
        print(f"❌ TranscriptAnalyzer verification failed: {e}")
        return False


def verify_error_handling():
    """Verify error handling in ConversationRunner."""
    print("\n🔍 Verifying error handling...")

    try:
        from multi_agent_harness import ConversationRunner, Participant
        from multi_agent_harness.adapters.openai import OpenAIAdapter

        adapter = OpenAIAdapter(api_key="test-key")
        alice = Participant(
            name="Alice", adapter=adapter, model="gpt-4o-mini", system_prompts=[]
        )

        # Test: fewer than 2 participants
        try:
            runner = ConversationRunner(participants=[alice])
            print("❌ Should have raised ValueError for < 2 participants")
            return False
        except ValueError as e:
            print(f"✅ Correctly raised ValueError: {e}")

        return True
    except Exception as e:
        print(f"❌ Error handling verification failed: {e}")
        return False


def verify_stop_condition():
    """Verify StopCondition type can be used."""
    print("\n🔍 Verifying StopCondition type...")

    try:
        from multi_agent_harness import ConversationTranscript, StopCondition

        # Define a simple stop condition
        def stop_after_3_turns(transcript: ConversationTranscript) -> bool:
            return len(transcript.turns) >= 3

        # Verify it matches the type
        condition: StopCondition = stop_after_3_turns

        print("✅ StopCondition type works correctly")
        print(f"   - Type: {StopCondition}")
        return True
    except Exception as e:
        print(f"❌ StopCondition verification failed: {e}")
        return False


def main():
    """Run all verification checks."""
    print("=" * 60)
    print("Phase 3: Orchestration Layer Verification")
    print("=" * 60)

    results = [
        verify_imports(),
        verify_conversation_runner(),
        verify_transcript_analyzer(),
        verify_error_handling(),
        verify_stop_condition(),
    ]

    print("\n" + "=" * 60)
    if all(results):
        print("✅ All verification checks passed!")
        print("=" * 60)
        return 0
    else:
        print("❌ Some verification checks failed")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
