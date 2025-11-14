"""Example: Two-participant debate pattern.

This example demonstrates how to use ConversationRunner to orchestrate a
multi-turn conversation between two participants with opposing viewpoints.
It shows:
- Setting up participants with different personas/viewpoints
- Running fixed-round debates
- Using stop conditions to terminate conversations early
- Displaying conversation transcripts

This pattern is useful for:
- Exploring different perspectives on a topic
- Red-teaming or adversarial testing
- Generating diverse responses
- Multi-agent deliberation
"""

from multi_agent_harness import ConversationRunner, ConversationTranscript, Participant
from multi_agent_harness.adapters.openai import OpenAIAdapter


def example_simple_debate():
    """Example: Simple 4-round debate between two viewpoints."""
    print("=" * 60)
    print("Example 1: Simple Debate (Fixed Rounds)")
    print("=" * 60)
    print()

    # Create participants with opposing viewpoints
    alice = Participant(
        name="Alice",
        adapter=OpenAIAdapter(),
        model="gpt-4o-mini",
        system_prompts=[
            "You are Alice, a technology optimist.",
            "You believe AI will be beneficial to humanity.",
            "Make concise, compelling arguments for your position.",
        ],
        temperature=0.7,  # More creative responses
    )

    bob = Participant(
        name="Bob",
        adapter=OpenAIAdapter(),
        model="gpt-4o-mini",
        system_prompts=[
            "You are Bob, a technology skeptic.",
            "You are concerned about AI's potential risks to humanity.",
            "Make concise, compelling arguments for your position.",
        ],
        temperature=0.7,
    )

    # Create conversation runner
    runner = ConversationRunner(participants=[alice, bob])

    print("Setup:")
    print(f"  Participant 1: {alice.name} - Technology optimist")
    print(f"  Participant 2: {bob.name} - Technology skeptic")
    print(f"  Topic: Impact of AI on humanity")
    print(f"  Rounds: 4")
    print()

    # This would run the debate (commented to avoid API calls)
    # transcript = runner.run(
    #     starting_message="What will be the impact of AI on humanity?",
    #     starting_participant=alice,  # Alice goes first
    #     max_turns=4,  # 4 turns total (2 rounds of back-and-forth)
    # )

    # Display transcript
    # print("Debate Transcript:")
    # print("-" * 60)
    # for turn in transcript.turns:
    #     print(f"{turn.role}: {turn.message}")
    #     print("-" * 60)
    # print()

    print("(Would execute 4 turns of debate)")
    print()
    print("=" * 60)
    print()


def example_debate_with_stop_condition():
    """Example: Debate that stops when participants reach agreement."""
    print("=" * 60)
    print("Example 2: Debate with Stop Condition")
    print("=" * 60)
    print()

    def stop_when_agreement(transcript: ConversationTranscript) -> bool:
        """Stop when participants agree or acknowledge each other's points.

        Args:
            transcript: Current conversation transcript

        Returns:
            True if conversation should stop, False otherwise
        """
        if len(transcript.turns) < 2:
            return False

        # Check last message for agreement indicators
        last_message = transcript.turns[-1].message.lower()
        agreement_phrases = [
            "i agree",
            "you're right",
            "good point",
            "i concede",
            "fair enough",
            "you've convinced me",
        ]

        return any(phrase in last_message for phrase in agreement_phrases)

    # Create participants
    alice = Participant(
        name="Alice",
        adapter=OpenAIAdapter(),
        model="gpt-4o-mini",
        system_prompts=[
            "You are Alice. Discuss topics thoughtfully.",
            "If your opponent makes a compelling point, acknowledge it.",
        ],
        temperature=0.8,
    )

    bob = Participant(
        name="Bob",
        adapter=OpenAIAdapter(),
        model="gpt-4o-mini",
        system_prompts=[
            "You are Bob. Discuss topics thoughtfully.",
            "If your opponent makes a compelling point, acknowledge it.",
        ],
        temperature=0.8,
    )

    runner = ConversationRunner(participants=[alice, bob])

    print("Setup:")
    print(f"  Participant 1: {alice.name}")
    print(f"  Participant 2: {bob.name}")
    print(f"  Stop condition: Conversation ends when agreement is reached")
    print(f"  Safety limit: 10 turns")
    print()

    # This would run until agreement (commented to avoid API calls)
    # transcript = runner.run(
    #     starting_message="Should Python use tabs or spaces for indentation?",
    #     starting_participant=alice,
    #     max_turns=10,  # Safety limit
    #     stop_condition=stop_when_agreement,
    # )

    # print(f"Debate ended after {len(transcript.turns)} turns")
    # print()

    print("(Would run until participants agree, max 10 turns)")
    print()
    print("=" * 60)
    print()


def example_multi_round_debate():
    """Example: Extended debate with multiple topics."""
    print("=" * 60)
    print("Example 3: Multi-Topic Debate")
    print("=" * 60)
    print()

    # Create participants with expertise
    expert_a = Participant(
        name="ExpertA",
        adapter=OpenAIAdapter(),
        model="gpt-4o-mini",
        system_prompts=[
            "You are an expert in software engineering.",
            "You prefer static typing and compiled languages.",
        ],
    )

    expert_b = Participant(
        name="ExpertB",
        adapter=OpenAIAdapter(),
        model="gpt-4o-mini",
        system_prompts=[
            "You are an expert in software engineering.",
            "You prefer dynamic typing and interpreted languages.",
        ],
    )

    runner = ConversationRunner(participants=[expert_a, expert_b])

    topics = [
        "What are the advantages of your preferred typing system?",
        "How does your approach handle large-scale projects?",
        "What about developer productivity?",
    ]

    print("Setup:")
    print(f"  Participant 1: {expert_a.name} - Static typing advocate")
    print(f"  Participant 2: {expert_b.name} - Dynamic typing advocate")
    print(f"  Topics: {len(topics)}")
    print()

    for idx, topic in enumerate(topics, 1):
        print(f"Topic {idx}: {topic}")

        # Each topic gets 2 rounds (4 turns)
        # transcript = runner.run(
        #     starting_message=topic,
        #     starting_participant=expert_a if idx % 2 == 1 else expert_b,
        #     max_turns=4,
        # )

        # Print summary
        # print(f"  Completed {len(transcript.turns)} turns")
        print(f"  (Would run 4 turns on this topic)")
        print()

    print("=" * 60)
    print()


def example_three_way_discussion():
    """Example: Three participants discussing a topic."""
    print("=" * 60)
    print("Example 4: Three-Way Discussion")
    print("=" * 60)
    print()

    # Create three participants with different perspectives
    optimist = Participant(
        name="Optimist",
        adapter=OpenAIAdapter(),
        model="gpt-4o-mini",
        system_prompts=["You always see the bright side of things."],
    )

    pessimist = Participant(
        name="Pessimist",
        adapter=OpenAIAdapter(),
        model="gpt-4o-mini",
        system_prompts=["You focus on potential problems and risks."],
    )

    realist = Participant(
        name="Realist",
        adapter=OpenAIAdapter(),
        model="gpt-4o-mini",
        system_prompts=["You take a balanced, practical view."],
    )

    # Create runner with 3 participants (round-robin turn-taking)
    runner = ConversationRunner(participants=[optimist, pessimist, realist])

    print("Setup:")
    print(f"  Participant 1: {optimist.name}")
    print(f"  Participant 2: {pessimist.name}")
    print(f"  Participant 3: {realist.name}")
    print(f"  Turn order: Round-robin (Optimist → Pessimist → Realist → ...)")
    print(f"  Turns: 6 (2 complete rounds)")
    print()

    # This would run the discussion (commented to avoid API calls)
    # transcript = runner.run(
    #     starting_message="Should we adopt this new technology?",
    #     max_turns=6,  # 2 complete rounds
    # )

    # print("Discussion transcript:")
    # for turn in transcript.turns:
    #     print(f"  {turn.role}: {turn.message[:80]}...")
    # print()

    print("(Would execute 6 turns in round-robin order)")
    print()
    print("=" * 60)
    print()


def main():
    """Run all debate examples."""
    print("\n" + "=" * 60)
    print("Debate Examples - Multi-Participant Conversations")
    print("=" * 60)
    print()

    example_simple_debate()
    example_debate_with_stop_condition()
    example_multi_round_debate()
    example_three_way_discussion()

    print("=" * 60)
    print("Examples complete!")
    print()
    print("To run with real API calls:")
    print("1. Set OPENAI_API_KEY environment variable")
    print("2. Uncomment the runner.run() calls")
    print("3. Run: python -m multi_agent_harness.examples.debate")
    print("=" * 60)


if __name__ == "__main__":
    main()
