# Usage Guide

This guide covers installation, basic concepts, and common usage patterns for the multi-agent harness library.

---

## Installation

### From Source

```bash
git clone https://github.com/yourusername/multi-agent-harness.git
cd multi-agent-harness
pip install -e .
```

### Dependencies

**Core dependencies (required):**
```bash
pip install openai>=1.0.0
```

**Optional provider dependencies:**
```bash
# For Anthropic
pip install anthropic>=0.18.0

# For Google Gemini
pip install google-generativeai>=0.3.0

# For xAI (uses OpenAI-compatible API)
# No additional dependencies needed
```

**Development dependencies:**
```bash
pip install pytest>=7.0 mypy>=1.0 black ruff
```

---

## Quick Start

### 1. Simple Single-Turn Interaction

```python
from multi_agent_harness import Participant
from multi_agent_harness.adapters import OpenAIAdapter

# Create a participant
assistant = Participant(
    name="Assistant",
    adapter=OpenAIAdapter(api_key="your-api-key"),
    model="gpt-4o-mini",
    system_prompts=["You are a helpful assistant."]
)

# Send a message
from multi_agent_harness import TurnRunner

runner = TurnRunner(participant=assistant)
response = runner.run_turn(
    history=[],
    user_message="What's the capital of France?"
)

print(response.message.content)  # "The capital of France is Paris."
```

### 2. Multi-Turn Conversation (Two Models)

```python
from multi_agent_harness import Participant, ConversationRunner
from multi_agent_harness.adapters import OpenAIAdapter

# Define two participants
alice = Participant(
    name="Alice",
    adapter=OpenAIAdapter(),
    model="gpt-4o-mini",
    system_prompts=["You are Alice, a curious student who asks questions."]
)

bob = Participant(
    name="Bob",
    adapter=OpenAIAdapter(),
    model="gpt-4o-mini",
    system_prompts=["You are Bob, a knowledgeable teacher who explains concepts."]
)

# Run conversation
runner = ConversationRunner(participants=[alice, bob])
transcript = runner.run(
    starting_message="I'm interested in learning about black holes.",
    starting_participant=alice,  # Alice speaks first
    max_turns=6  # 3 exchanges
)

# Print conversation
for idx, turn in enumerate(transcript.turns, 1):
    print(f"Turn {idx} ({turn.role}): {turn.message}\n")
```

### 3. Using Different Providers

```python
from multi_agent_harness import Participant
from multi_agent_harness.adapters import (
    OpenAIAdapter,
    AnthropicAdapter,
    XAIAdapter,
    GeminiAdapter
)

# OpenAI
openai_participant = Participant(
    name="GPT",
    adapter=OpenAIAdapter(api_key="..."),
    model="gpt-4o"
)

# Anthropic Claude
claude_participant = Participant(
    name="Claude",
    adapter=AnthropicAdapter(api_key="..."),
    model="claude-sonnet-4-5-20250929"
)

# xAI Grok
grok_participant = Participant(
    name="Grok",
    adapter=XAIAdapter(api_key="..."),
    model="grok-beta"
)

# Google Gemini
gemini_participant = Participant(
    name="Gemini",
    adapter=GeminiAdapter(api_key="..."),
    model="gemini-1.5-pro"
)

# Mix and match in conversations!
runner = ConversationRunner(participants=[openai_participant, claude_participant])
```

---

## Working with Tools

### Basic Tool Usage

```python
from multi_agent_harness import Participant, TurnRunner, ToolDefinition

# Define tools
def execute_tool(name: str, args: dict):
    if name == "add":
        return {"result": args["a"] + args["b"]}
    elif name == "multiply":
        return {"result": args["a"] * args["b"]}
    raise ValueError(f"Unknown tool: {name}")

tools = [
    ToolDefinition(
        name="add",
        description="Add two numbers",
        input_schema={
            "type": "object",
            "properties": {
                "a": {"type": "number", "description": "First number"},
                "b": {"type": "number", "description": "Second number"}
            },
            "required": ["a", "b"]
        }
    ),
    ToolDefinition(
        name="multiply",
        description="Multiply two numbers",
        input_schema={
            "type": "object",
            "properties": {
                "a": {"type": "number"},
                "b": {"type": "number"}
            },
            "required": ["a", "b"]
        }
    )
]

# Create participant with tools
calculator = Participant(
    name="Calculator",
    adapter=OpenAIAdapter(),
    model="gpt-4o-mini",
    system_prompts=["You are a calculator assistant. Use tools to compute results."]
)

runner = TurnRunner(
    participant=calculator,
    tools=tools,
    tool_executor=execute_tool
)

response = runner.run_turn(
    history=[],
    user_message="What is (5 + 3) * 4?",
    max_tool_steps=5
)

print(response.message.content)  # "32"
```

### Tools in Multi-Turn Conversations

```python
# Tools are available to all participants in a ConversationRunner
runner = ConversationRunner(
    participants=[alice, bob],
    tools=tools,
    tool_executor=execute_tool
)

transcript = runner.run(
    starting_message="Can you help me calculate some numbers?",
    max_turns=4
)
```

### Controlling Tool Usage

```python
# Force tool use
response = runner.run_turn(
    history=[],
    user_message="Add 5 and 3",
    tool_choice="required"  # Model must call a tool
)

# Disable tools for this turn
response = runner.run_turn(
    history=[],
    user_message="Just chat with me",
    tool_choice="none"  # Model cannot call tools
)

# Auto (default): model decides
response = runner.run_turn(
    history=[],
    user_message="What's 2 + 2?",
    tool_choice="auto"
)
```

---

## Common Patterns

### Pattern 1: Assistant with Tools

Full example in [`examples/assistant.py`](../src/multi_agent_harness/examples/assistant.py)

```python
from multi_agent_harness import Participant, TurnRunner

assistant = Participant(
    name="Assistant",
    adapter=OpenAIAdapter(),
    model="gpt-4o-mini",
    system_prompts=[
        "You are a helpful assistant with access to tools.",
        "Use tools when appropriate to answer questions accurately."
    ]
)

runner = TurnRunner(participant=assistant, tools=my_tools, tool_executor=executor)

# Accumulate conversation history
history = []
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    response = runner.run_turn(history=history, user_message=user_input)

    # Update history
    from multi_agent_harness import ChatMessage
    history.append(ChatMessage(role="user", content=user_input))
    history.append(response.message)

    print(f"Assistant: {response.message.content}")
```

### Pattern 2: Multi-Model Debate

Full example in [`examples/debate.py`](../src/multi_agent_harness/examples/debate.py)

```python
from multi_agent_harness import ConversationRunner

# Two models with opposing viewpoints
pro = Participant(
    name="Pro",
    adapter=OpenAIAdapter(),
    model="gpt-4o-mini",
    system_prompts=["Argue in favor of the given proposition."]
)

con = Participant(
    name="Con",
    adapter=OpenAIAdapter(),
    model="gpt-4o-mini",
    system_prompts=["Argue against the given proposition."]
)

runner = ConversationRunner(participants=[pro, con])
transcript = runner.run(
    starting_message="Remote work is better than office work.",
    starting_participant=pro,
    max_turns=8  # 4 rounds of debate
)

# Analyze the debate
for turn in transcript.turns:
    print(f"{turn.role}: {turn.message}\n")
```

### Pattern 3: Judge/Evaluator

Full example in [`examples/judge.py`](../src/multi_agent_harness/examples/judge.py)

```python
from multi_agent_harness import TranscriptAnalyzer, ResponseFormat

# First, run a conversation
transcript = runner.run(...)

# Then have a judge evaluate it
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
    transcript=transcript,
    response_format=ResponseFormat(
        type="json_schema",
        json_schema={
            "name": "evaluation",
            "schema": {
                "type": "object",
                "properties": {
                    "score": {"type": "number", "minimum": 1, "maximum": 10},
                    "reasoning": {"type": "string"}
                },
                "required": ["score", "reasoning"],
                "additionalProperties": False
            },
            "strict": True
        }
    )
)

import json
result = json.loads(verdict.message.content)
print(f"Score: {result['score']}/10")
print(f"Reasoning: {result['reasoning']}")
```

### Pattern 4: Interrogator (Follow-up Questions)

Full example in [`examples/interrogator.py`](../src/multi_agent_harness/examples/interrogator.py)

```python
from multi_agent_harness import TranscriptAnalyzer, ConversationRunner

# Initial conversation
transcript = initial_runner.run(...)

# Interrogator analyzes and generates questions
interrogator = Participant(
    name="Interrogator",
    adapter=OpenAIAdapter(),
    model="gpt-4o",
    system_prompts=[
        "Analyze the conversation and identify gaps or unclear points.",
        "Generate 3 clarifying questions."
    ]
)

analyzer = TranscriptAnalyzer(participant=interrogator)
questions = analyzer.analyze(transcript=transcript)

# Continue conversation with the questions
continued_runner = ConversationRunner(participants=[original_participant])
continued_transcript = continued_runner.run(
    starting_message=questions.message.content,
    initial_transcript=transcript,  # Continue from existing transcript
    max_turns=3
)
```

### Pattern 5: Custom Stop Conditions

```python
from multi_agent_harness import ConversationTranscript

def stop_on_agreement(transcript: ConversationTranscript) -> bool:
    """Stop when both participants agree."""
    if len(transcript.turns) < 2:
        return False

    last_turn = transcript.turns[-1].message.lower()
    return "i agree" in last_turn or "you're right" in last_turn

def stop_on_max_tokens(max_tokens: int):
    """Factory for token-based stop condition."""
    def condition(transcript: ConversationTranscript) -> bool:
        total = sum(len(turn.message.split()) for turn in transcript.turns)
        return total * 1.3 > max_tokens  # Rough estimate: 1.3 words per token
    return condition

runner = ConversationRunner(participants=[alice, bob])
transcript = runner.run(
    starting_message="Let's discuss...",
    stop_condition=stop_on_agreement,
    max_turns=20  # Safety limit
)
```

---

## Configuration

### Participant Configuration

```python
from multi_agent_harness import ParticipantConfig

config = ParticipantConfig(
    model_name="gpt-4o",
    temperature=0.7,
    max_tokens=1000,
    top_p=0.9,
    # ... other provider-specific params
)

participant = Participant(
    name="Assistant",
    adapter=OpenAIAdapter(),
    model=config.model_name,
    system_prompts=["..."],
    model_config=config
)
```

### Response Formatting

```python
from multi_agent_harness import ResponseFormat

# JSON mode (unstructured)
response = runner.run_turn(
    history=[],
    user_message="List three colors in JSON",
    response_format=ResponseFormat(type="json_object")
)

# JSON schema (structured)
response = runner.run_turn(
    history=[],
    user_message="Analyze this text",
    response_format=ResponseFormat(
        type="json_schema",
        json_schema={
            "name": "analysis",
            "schema": {
                "type": "object",
                "properties": {
                    "sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]},
                    "confidence": {"type": "number"}
                },
                "required": ["sentiment", "confidence"]
            }
        }
    )
)
```

### System Prompts

```python
# Single system prompt
participant = Participant(
    name="Helper",
    adapter=OpenAIAdapter(),
    model="gpt-4o-mini",
    system_prompts=["You are a helpful assistant."]
)

# Multiple system prompts (concatenated)
participant = Participant(
    name="Specialist",
    adapter=OpenAIAdapter(),
    model="gpt-4o-mini",
    system_prompts=[
        "You are a Python expert.",
        "Always provide working code examples.",
        "Explain your reasoning step by step."
    ]
)
```

---

## Error Handling

### Adapter Errors

```python
from openai import OpenAIError

try:
    response = runner.run_turn(history=[], user_message="Hello")
except OpenAIError as e:
    print(f"OpenAI API error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

### Tool Execution Errors

```python
def safe_tool_executor(name: str, args: dict):
    try:
        if name == "risky_operation":
            return perform_operation(args)
    except Exception as e:
        # Return error as result (model will see it)
        return {"error": str(e)}

    raise ValueError(f"Unknown tool: {name}")

# Tool errors are automatically captured in ToolInvocationRecord
runner = TurnRunner(participant=assistant, tools=tools, tool_executor=safe_tool_executor)
response = runner.run_turn(...)

# Check if tools succeeded
for tool_call in response.tool_calls:
    if hasattr(tool_call, 'error'):
        print(f"Tool {tool_call.name} failed: {tool_call.error}")
```

### Conversation Errors

```python
try:
    runner = ConversationRunner(participants=[alice])  # Need at least 2!
except ValueError as e:
    print(f"Configuration error: {e}")

try:
    transcript = runner.run(
        starting_message="...",
        starting_participant=unknown_participant  # Not in participants list
    )
except ValueError as e:
    print(f"Runtime error: {e}")
```

---

## Best Practices

### 1. Always Set API Keys via Environment

```python
import os

# Better: Use environment variables
adapter = OpenAIAdapter(api_key=os.getenv("OPENAI_API_KEY"))

# Avoid: Hardcoding keys
adapter = OpenAIAdapter(api_key="sk-...")  # ❌ Don't do this
```

### 2. Manage Conversation History Length

```python
# Keep history bounded to avoid token limits
MAX_HISTORY = 20

history = []
while True:
    user_input = input("You: ")
    response = runner.run_turn(history=history[-MAX_HISTORY:], user_message=user_input)

    history.append(ChatMessage(role="user", content=user_input))
    history.append(response.message)
```

### 3. Use Type Hints

```python
from multi_agent_harness import Participant, ConversationTranscript

def process_conversation(
    participants: list[Participant],
    topic: str
) -> ConversationTranscript:
    runner = ConversationRunner(participants=participants)
    return runner.run(starting_message=topic, max_turns=10)
```

### 4. Test with Mock Adapters

```python
# For testing, create a mock adapter
class MockAdapter(ProviderAdapter):
    def send_chat(self, role_config, messages, **kwargs):
        return ChatResponse(
            message=ChatMessage(role="assistant", content="Mock response"),
            tool_calls=[],
            stop_reason="complete"
        )

# Use in tests
test_participant = Participant(
    name="Test",
    adapter=MockAdapter(),
    model="mock-model"
)
```

### 5. Log Conversations for Debugging

```python
def log_transcript(transcript: ConversationTranscript, filename: str):
    with open(filename, "w") as f:
        for idx, turn in enumerate(transcript.turns, 1):
            f.write(f"Turn {idx} ({turn.role}):\n")
            f.write(f"{turn.message}\n\n")

            if turn.tool_invocations:
                f.write("Tools:\n")
                for tool in turn.tool_invocations:
                    f.write(f"  - {tool.tool_name}({tool.arguments})\n")
                f.write("\n")

transcript = runner.run(...)
log_transcript(transcript, "conversation.log")
```

---

## Next Steps

- **Architecture:** See [`ARCHITECTURE.md`](ARCHITECTURE.md) for design philosophy and component relationships
- **Extending:** See [`EXTENDING.md`](EXTENDING.md) for implementing new providers
- **Examples:** Check `examples/` directory for complete working examples:
  - `assistant.py` — Interactive assistant with tools
  - `debate.py` — Two models arguing
  - `judge.py` — Third model scoring conversations
  - `interrogator.py` — Follow-up question generation

---

## Troubleshooting

### "No module named 'multi_agent_harness'"

```bash
# Make sure you installed the package
pip install -e .

# Or add src to Python path
export PYTHONPATH="${PYTHONPATH}:/path/to/multi-agent-harness/src"
```

### "API key not found"

```bash
# Set environment variables
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export XAI_API_KEY="xai-..."
export GOOGLE_API_KEY="..."
```

### "Tool call missing required call_id"

Some providers (like OpenAI) require tool calls to have unique IDs. This is validated automatically:

```python
# The library handles this - you don't need to do anything
# If you see this error, the provider's response is malformed
```

### "ConversationRunner requires at least 2 participants"

```python
# ❌ Wrong
runner = ConversationRunner(participants=[alice])

# ✅ Correct
runner = ConversationRunner(participants=[alice, bob])
```

### "Maximum tool steps reached"

The tool loop has a safety limit:

```python
# Increase max_tool_steps if needed
response = runner.run_turn(
    history=[],
    user_message="Complex task requiring many tool calls",
    max_tool_steps=10  # Default is usually 5-6
)
```

---

## Additional Resources

- **GitHub:** [https://github.com/yourusername/multi-agent-harness](https://github.com/yourusername/multi-agent-harness)
- **Issues:** [https://github.com/yourusername/multi-agent-harness/issues](https://github.com/yourusername/multi-agent-harness/issues)
- **API Reference:** (Coming soon)
