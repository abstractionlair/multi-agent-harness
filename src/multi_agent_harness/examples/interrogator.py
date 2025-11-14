"""Example: Interrogator pattern for follow-up questioning.

This example demonstrates how to use TranscriptAnalyzer to generate follow-up
questions based on conversation analysis, and then inject a new participant
into an ongoing conversation. It shows:
- Analyzing transcripts to identify gaps or areas for clarification
- Generating targeted follow-up questions
- Participant injection mid-conversation
- Continuing conversations with additional context

This pattern is useful for:
- Investigative/interview scenarios
- Quality assurance (asking clarifying questions)
- Filling knowledge gaps
- Active listening and engagement
- Moderating discussions
"""

import json

from multi_agent_harness import (
    ConversationRunner,
    ConversationTranscript,
    ConversationTurn,
    Participant,
    ResponseFormat,
    TranscriptAnalyzer,
)
from multi_agent_harness.adapters.openai import OpenAIAdapter


def create_initial_conversation() -> ConversationTranscript:
    """Create a sample conversation with potential gaps.

    Returns:
        ConversationTranscript with a conversation that could use follow-up
    """
    transcript = ConversationTranscript()

    transcript.add_turn(
        ConversationTurn(
            role="User",
            message="I'm thinking about switching to a new programming language.",
        )
    )

    transcript.add_turn(
        ConversationTurn(
            role="Assistant",
            message="That's interesting! There are many great options available. What's motivating this change?",
        )
    )

    transcript.add_turn(
        ConversationTurn(
            role="User",
            message="My current projects are getting slower and I need better performance.",
        )
    )

    transcript.add_turn(
        ConversationTurn(
            role="Assistant",
            message="Performance is definitely important. Languages like Rust, C++, and Go are known for excellent performance.",
        )
    )

    return transcript


def example_question_generation():
    """Example: Generate follow-up questions from transcript analysis."""
    print("=" * 60)
    print("Example 1: Generating Follow-Up Questions")
    print("=" * 60)
    print()

    # Create interrogator participant
    interrogator = Participant(
        name="Interrogator",
        adapter=OpenAIAdapter(),
        model="gpt-4o",
        system_prompts=[
            "You are an expert interviewer analyzing conversations.",
            "Identify important details that are missing or unclear.",
            "Generate 2-4 specific follow-up questions to gather more information.",
            'Output JSON: {"questions": ["<question1>", "<question2>", ...], "reasoning": "<why these questions>"}',
        ],
        temperature=0.3,
    )

    analyzer = TranscriptAnalyzer(participant=interrogator)
    transcript = create_initial_conversation()

    print("Initial Conversation:")
    print("-" * 60)
    for turn in transcript.turns:
        print(f"{turn.role}: {turn.message}")
    print("-" * 60)
    print()

    # Define schema for follow-up questions
    questions_schema = {
        "name": "follow_up_questions",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "questions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of follow-up questions",
                },
                "reasoning": {
                    "type": "string",
                    "description": "Why these questions are important",
                },
            },
            "required": ["questions", "reasoning"],
            "additionalProperties": False,
        },
    }

    # This would generate questions (commented to avoid API calls)
    # response = analyzer.analyze(
    #     transcript=transcript,
    #     analysis_prompt="What important information is missing? What follow-up questions should be asked?",
    #     response_format=ResponseFormat(
    #         type="json_schema",
    #         json_schema=questions_schema,
    #     ),
    # )

    # result = json.loads(response.message.content)
    # print("Generated Follow-Up Questions:")
    # print(f"Reasoning: {result['reasoning']}")
    # print()
    # print("Questions:")
    # for idx, question in enumerate(result['questions'], 1):
    #     print(f"  {idx}. {question}")
    # print()

    print("(Would generate 2-4 targeted follow-up questions)")
    print("Example questions might include:")
    print("  1. What programming language are you currently using?")
    print("  2. What type of performance issues are you experiencing?")
    print("  3. What is your team's experience level?")
    print()
    print("=" * 60)
    print()


def example_participant_injection():
    """Example: Inject interrogator into ongoing conversation."""
    print("=" * 60)
    print("Example 2: Participant Injection")
    print("=" * 60)
    print()

    # Initial participants
    user_sim = Participant(
        name="User",
        adapter=OpenAIAdapter(),
        model="gpt-4o-mini",
        system_prompts=[
            "You are a developer asking for advice about programming languages.",
        ],
    )

    assistant = Participant(
        name="Assistant",
        adapter=OpenAIAdapter(),
        model="gpt-4o-mini",
        system_prompts=[
            "You are a helpful programming advisor.",
        ],
    )

    # Run initial conversation
    initial_runner = ConversationRunner(participants=[user_sim, assistant])

    print("Phase 1: Initial conversation (User ↔ Assistant)")
    print()

    # This would run initial conversation (commented to avoid API calls)
    # initial_transcript = initial_runner.run(
    #     starting_message="I'm thinking about switching programming languages.",
    #     starting_participant=user_sim,
    #     max_turns=4,
    # )

    # Simulate the transcript we created earlier
    initial_transcript = create_initial_conversation()

    print("After 4 turns of conversation...")
    print()

    # Create interrogator to analyze the conversation
    interrogator = Participant(
        name="Interrogator",
        adapter=OpenAIAdapter(),
        model="gpt-4o",
        system_prompts=[
            "You are a technical interviewer.",
            "Ask clarifying questions to understand requirements better.",
            "Be concise and direct.",
        ],
    )

    analyzer = TranscriptAnalyzer(participant=interrogator)

    print("Phase 2: Interrogator analyzes conversation")
    print()

    # This would generate follow-up (commented to avoid API calls)
    # analysis = analyzer.analyze(
    #     transcript=initial_transcript,
    #     analysis_prompt="Based on this conversation, what's the most important question to ask next?",
    # )

    # interrogator_question = analysis.message.content
    interrogator_question = "What programming language are you currently using, and what specific performance metrics are you concerned about?"

    print(f"Interrogator identified key question: {interrogator_question}")
    print()

    # Phase 3: Interrogator joins the conversation
    print("Phase 3: Interrogator joins conversation")
    print()

    # Create new runner with interrogator and original assistant
    continued_runner = ConversationRunner(participants=[interrogator, assistant])

    # This would continue with interrogator (commented to avoid API calls)
    # continued_transcript = continued_runner.run(
    #     starting_message=interrogator_question,
    #     starting_participant=interrogator,
    #     initial_transcript=initial_transcript,  # Carry forward context
    #     max_turns=4,
    # )

    # print("Continued conversation with interrogator:")
    # for turn in continued_transcript.turns[len(initial_transcript.turns):]:
    #     print(f"{turn.role}: {turn.message}")
    #     print()

    print("(Would continue conversation with interrogator asking follow-ups)")
    print()
    print("=" * 60)
    print()


def example_clarification_loop():
    """Example: Iterative clarification with interrogator."""
    print("=" * 60)
    print("Example 3: Iterative Clarification Loop")
    print("=" * 60)
    print()

    # Create participants
    subject = Participant(
        name="Subject",
        adapter=OpenAIAdapter(),
        model="gpt-4o-mini",
        system_prompts=["You are explaining a technical concept."],
    )

    interrogator = Participant(
        name="Interrogator",
        adapter=OpenAIAdapter(),
        model="gpt-4o-mini",
        system_prompts=[
            "You ask clarifying questions to ensure complete understanding.",
            "Focus on one aspect at a time.",
            "Ask concise, specific questions.",
        ],
    )

    analyzer = TranscriptAnalyzer(participant=interrogator)

    print("Pattern: Iterative question-and-answer clarification")
    print()

    # Simulate multiple rounds of clarification
    transcript = ConversationTranscript()
    transcript.add_turn(
        ConversationTurn(
            role="Subject",
            message="We use microservices architecture.",
        )
    )

    num_rounds = 3
    print(f"Running {num_rounds} rounds of clarification:")
    print()

    for round_num in range(1, num_rounds + 1):
        print(f"Round {round_num}:")

        # Interrogator analyzes and asks follow-up
        # response = analyzer.analyze(
        #     transcript=transcript,
        #     analysis_prompt="Ask one specific follow-up question to clarify.",
        # )
        # question = response.message.content

        # Simulated questions
        questions = [
            "How many microservices does your system have?",
            "What communication protocol do you use between services?",
            "How do you handle service discovery?",
        ]
        question = questions[round_num - 1] if round_num <= len(questions) else "Tell me more."

        print(f"  Interrogator: {question}")

        # Add to transcript
        transcript.add_turn(ConversationTurn(role="Interrogator", message=question))

        # Subject responds (in real scenario, would use TurnRunner)
        # response = subject_runner.run_turn(...)
        # answer = response.message.content

        answer = f"Response to: {question}"
        print(f"  Subject: {answer}")

        transcript.add_turn(ConversationTurn(role="Subject", message=answer))
        print()

    print(f"After {num_rounds} rounds, gathered comprehensive information")
    print()
    print("=" * 60)
    print()


def example_gap_identification():
    """Example: Identify and address information gaps."""
    print("=" * 60)
    print("Example 4: Gap Identification and Filling")
    print("=" * 60)
    print()

    # Create gap analyzer
    gap_analyzer_participant = Participant(
        name="GapAnalyzer",
        adapter=OpenAIAdapter(),
        model="gpt-4o",
        system_prompts=[
            "You analyze conversations to find information gaps.",
            "Identify what key information is missing for a complete understanding.",
            'Output JSON: {"gaps": [{"topic": "<area>", "missing": "<what\'s missing>"}], "priority": "<which gap to address first>"}',
        ],
        temperature=0.0,
    )

    analyzer = TranscriptAnalyzer(participant=gap_analyzer_participant)
    transcript = create_initial_conversation()

    print("Analyzing conversation for information gaps...")
    print()

    gap_schema = {
        "name": "gaps",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "gaps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "topic": {"type": "string"},
                            "missing": {"type": "string"},
                        },
                        "required": ["topic", "missing"],
                        "additionalProperties": False,
                    },
                },
                "priority": {"type": "string"},
            },
            "required": ["gaps", "priority"],
            "additionalProperties": False,
        },
    }

    # This would identify gaps (commented to avoid API calls)
    # response = analyzer.analyze(
    #     transcript=transcript,
    #     response_format=ResponseFormat(
    #         type="json_schema",
    #         json_schema=gap_schema,
    #     ),
    # )

    # gaps = json.loads(response.message.content)
    # print("Identified Information Gaps:")
    # for gap in gaps['gaps']:
    #     print(f"  - {gap['topic']}: {gap['missing']}")
    # print()
    # print(f"Priority to address: {gaps['priority']}")
    # print()

    print("(Would identify gaps like:)")
    print("  - Current language: not specified")
    print("  - Performance metrics: vague 'slower'")
    print("  - Project constraints: team size, timeline, budget")
    print()
    print("=" * 60)
    print()


def main():
    """Run all interrogator examples."""
    print("\n" + "=" * 60)
    print("Interrogator Examples - Follow-Up Questioning Patterns")
    print("=" * 60)
    print()

    example_question_generation()
    example_participant_injection()
    example_clarification_loop()
    example_gap_identification()

    print("=" * 60)
    print("Examples complete!")
    print()
    print("To run with real API calls:")
    print("1. Set OPENAI_API_KEY environment variable")
    print("2. Uncomment the API call sections")
    print("3. Run: python -m multi_agent_harness.examples.interrogator")
    print()
    print("Key Concepts:")
    print("  - Transcript analysis: Identify what's missing")
    print("  - Follow-up questions: Generate targeted inquiries")
    print("  - Participant injection: Add new participants mid-conversation")
    print("  - Context preservation: Carry forward conversation history")
    print("  - Iterative clarification: Progressive information gathering")
    print("=" * 60)


if __name__ == "__main__":
    main()
