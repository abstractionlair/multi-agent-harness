"""Example: Judge pattern for transcript analysis.

This example demonstrates how to use TranscriptAnalyzer to evaluate conversation
transcripts. It shows:
- Creating a judge participant with scoring criteria
- Analyzing transcripts with structured output (JSON)
- Different judging criteria (helpfulness, accuracy, tone)
- Multi-dimensional scoring

This pattern is useful for:
- Quality evaluation of AI conversations
- Automated testing and benchmarking
- Content moderation
- Performance monitoring
"""

import json

from multi_agent_harness import (
    ConversationTranscript,
    ConversationTurn,
    Participant,
    ResponseFormat,
    TranscriptAnalyzer,
)
from multi_agent_harness.adapters.openai import OpenAIAdapter


def create_sample_transcript() -> ConversationTranscript:
    """Create a sample conversation transcript for testing.

    Returns:
        ConversationTranscript with a sample assistant-user conversation
    """
    transcript = ConversationTranscript()

    transcript.add_turn(
        ConversationTurn(
            role="User",
            message="What is the capital of France?",
        )
    )

    transcript.add_turn(
        ConversationTurn(
            role="Assistant",
            message="The capital of France is Paris. It is the largest city in France and has been the capital since 987 AD. Is there anything else you'd like to know?",
        )
    )

    transcript.add_turn(
        ConversationTurn(
            role="User",
            message="Thank you! That's very helpful.",
        )
    )

    return transcript


def example_simple_judge():
    """Example: Simple judge that scores helpfulness."""
    print("=" * 60)
    print("Example 1: Simple Helpfulness Judge")
    print("=" * 60)
    print()

    # Create a judge participant
    judge = Participant(
        name="Judge",
        adapter=OpenAIAdapter(),
        model="gpt-4o",
        system_prompts=[
            "You are an impartial judge evaluating AI assistant conversations.",
            "Score the assistant's response on helpfulness from 1-10.",
            "Consider: accuracy, completeness, clarity, and tone.",
            'Output your verdict as JSON: {"score": <number>, "reasoning": "<text>"}',
        ],
        temperature=0.0,  # Deterministic scoring
    )

    # Create analyzer
    analyzer = TranscriptAnalyzer(participant=judge)

    # Create sample transcript
    transcript = create_sample_transcript()

    print("Transcript to judge:")
    print("-" * 60)
    for turn in transcript.turns:
        print(f"{turn.role}: {turn.message}")
    print("-" * 60)
    print()

    # Define JSON schema for structured output
    verdict_schema = {
        "name": "verdict",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "score": {
                    "type": "number",
                    "description": "Helpfulness score from 1-10",
                },
                "reasoning": {
                    "type": "string",
                    "description": "Explanation for the score",
                },
            },
            "required": ["score", "reasoning"],
            "additionalProperties": False,
        },
    }

    # This would analyze the transcript (commented to avoid API calls)
    # response = analyzer.analyze(
    #     transcript=transcript,
    #     response_format=ResponseFormat(
    #         type="json_schema",
    #         json_schema=verdict_schema,
    #     ),
    # )

    # Parse and display verdict
    # verdict = json.loads(response.message.content)
    # print(f"Judge's Verdict:")
    # print(f"  Score: {verdict['score']}/10")
    # print(f"  Reasoning: {verdict['reasoning']}")
    # print()

    print("(Would produce JSON verdict with score and reasoning)")
    print()
    print("=" * 60)
    print()


def example_multi_criteria_judge():
    """Example: Judge that evaluates multiple criteria."""
    print("=" * 60)
    print("Example 2: Multi-Criteria Judge")
    print("=" * 60)
    print()

    # Create a judge with multiple evaluation dimensions
    judge = Participant(
        name="MultiCriteriaJudge",
        adapter=OpenAIAdapter(),
        model="gpt-4o",
        system_prompts=[
            "You are an expert evaluator of AI assistant conversations.",
            "Evaluate conversations on multiple criteria:",
            "  - Accuracy: Is the information correct?",
            "  - Completeness: Does it fully answer the question?",
            "  - Clarity: Is it easy to understand?",
            "  - Tone: Is it professional and friendly?",
            "Score each criterion from 1-10 and provide overall score.",
            'Output JSON with structure: {"criteria": {"accuracy": <score>, ...}, "overall": <score>, "summary": "<text>"}',
        ],
        temperature=0.0,
    )

    analyzer = TranscriptAnalyzer(participant=judge)
    transcript = create_sample_transcript()

    # Define detailed schema
    evaluation_schema = {
        "name": "evaluation",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "criteria": {
                    "type": "object",
                    "properties": {
                        "accuracy": {"type": "number"},
                        "completeness": {"type": "number"},
                        "clarity": {"type": "number"},
                        "tone": {"type": "number"},
                    },
                    "required": ["accuracy", "completeness", "clarity", "tone"],
                    "additionalProperties": False,
                },
                "overall": {
                    "type": "number",
                    "description": "Overall score",
                },
                "summary": {
                    "type": "string",
                    "description": "Brief summary of evaluation",
                },
            },
            "required": ["criteria", "overall", "summary"],
            "additionalProperties": False,
        },
    }

    print("Evaluating on multiple criteria:")
    print("  - Accuracy")
    print("  - Completeness")
    print("  - Clarity")
    print("  - Tone")
    print()

    # This would analyze with multiple criteria (commented to avoid API calls)
    # response = analyzer.analyze(
    #     transcript=transcript,
    #     response_format=ResponseFormat(
    #         type="json_schema",
    #         json_schema=evaluation_schema,
    #     ),
    # )

    # evaluation = json.loads(response.message.content)
    # print("Multi-Criteria Evaluation:")
    # print("  Criteria Scores:")
    # for criterion, score in evaluation["criteria"].items():
    #     print(f"    {criterion.capitalize()}: {score}/10")
    # print(f"  Overall: {evaluation['overall']}/10")
    # print(f"  Summary: {evaluation['summary']}")
    # print()

    print("(Would produce detailed multi-criteria evaluation)")
    print()
    print("=" * 60)
    print()


def example_custom_analysis():
    """Example: Custom analysis prompt for specific evaluation."""
    print("=" * 60)
    print("Example 3: Custom Analysis Prompt")
    print("=" * 60)
    print()

    # Create a versatile judge
    judge = Participant(
        name="Analyst",
        adapter=OpenAIAdapter(),
        model="gpt-4o",
        system_prompts=[
            "You are a conversation analyst.",
            "Follow the specific analysis instructions provided.",
        ],
        temperature=0.2,
    )

    analyzer = TranscriptAnalyzer(participant=judge)
    transcript = create_sample_transcript()

    # Different analysis prompts for different purposes
    analysis_prompts = [
        "Identify any factual errors or inaccuracies in the assistant's responses.",
        "Evaluate the tone and professionalism of the conversation.",
        "Suggest improvements the assistant could make.",
    ]

    print("Running multiple analyses on the same transcript:")
    print()

    for idx, prompt in enumerate(analysis_prompts, 1):
        print(f"Analysis {idx}: {prompt}")

        # This would run the analysis (commented to avoid API calls)
        # response = analyzer.analyze(
        #     transcript=transcript,
        #     analysis_prompt=prompt,
        # )
        # print(f"Result: {response.message.content}")
        # print()

        print("  (Would provide analysis based on this prompt)")
        print()

    print("=" * 60)
    print()


def example_comparative_judge():
    """Example: Judge comparing two different transcripts."""
    print("=" * 60)
    print("Example 4: Comparative Judging")
    print("=" * 60)
    print()

    # Create two transcripts to compare
    transcript_a = ConversationTranscript()
    transcript_a.add_turn(ConversationTurn(role="User", message="What is Python?"))
    transcript_a.add_turn(
        ConversationTurn(
            role="Assistant",
            message="Python is a programming language.",
        )
    )

    transcript_b = ConversationTranscript()
    transcript_b.add_turn(ConversationTurn(role="User", message="What is Python?"))
    transcript_b.add_turn(
        ConversationTurn(
            role="Assistant",
            message="Python is a high-level, interpreted programming language known for its simplicity and readability. Created by Guido van Rossum in 1991, it supports multiple programming paradigms and has a comprehensive standard library.",
        )
    )

    judge = Participant(
        name="ComparativeJudge",
        adapter=OpenAIAdapter(),
        model="gpt-4o",
        system_prompts=[
            "You are comparing two assistant responses to the same question.",
            "Determine which response is better and explain why.",
            'Output JSON: {"better": "A" or "B", "reasoning": "<explanation>"}',
        ],
        temperature=0.0,
    )

    analyzer = TranscriptAnalyzer(participant=judge)

    print("Comparing two responses:")
    print()
    print("Response A:", transcript_a.turns[1].message)
    print()
    print("Response B:", transcript_b.turns[1].message)
    print()

    # For comparative judging, we could format both transcripts in the prompt
    # comparison_prompt = f"""
    # RESPONSE A:
    # {analyzer._format_transcript(transcript_a)}
    #
    # RESPONSE B:
    # {analyzer._format_transcript(transcript_b)}
    #
    # Which response is better? Explain your reasoning.
    # """

    # response = analyzer.analyze(
    #     transcript=ConversationTranscript(),  # Empty, we're using custom prompt
    #     analysis_prompt=comparison_prompt,
    #     response_format=ResponseFormat(
    #         type="json_schema",
    #         json_schema={...}
    #     )
    # )

    print("(Would compare and determine which response is better)")
    print()
    print("=" * 60)
    print()


def main():
    """Run all judge examples."""
    print("\n" + "=" * 60)
    print("Judge Examples - Transcript Analysis Patterns")
    print("=" * 60)
    print()

    example_simple_judge()
    example_multi_criteria_judge()
    example_custom_analysis()
    example_comparative_judge()

    print("=" * 60)
    print("Examples complete!")
    print()
    print("To run with real API calls:")
    print("1. Set OPENAI_API_KEY environment variable")
    print("2. Uncomment the analyzer.analyze() calls")
    print("3. Run: python -m multi_agent_harness.examples.judge")
    print()
    print("Key Concepts:")
    print("  - TranscriptAnalyzer: Helper for analyzing conversations")
    print("  - ResponseFormat: Structured JSON output for consistent scoring")
    print("  - Custom prompts: Tailor analysis to specific criteria")
    print("  - Multi-criteria: Evaluate multiple dimensions simultaneously")
    print("=" * 60)


if __name__ == "__main__":
    main()
