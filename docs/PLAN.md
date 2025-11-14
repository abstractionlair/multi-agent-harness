# Plan: Multi-Agent LLM Harness Library

Status: Draft (initial extraction → redesign)
Last Updated: 2025-11-14

## Vision

A **conversation orchestration toolkit** for building multi-model LLM interactions. Rather than prescriptive "engines," provide flexible primitives that let you:

- Wire up conversations between (Model A, Prompt A) ↔ (Model B, Prompt B)
- Have (Model C, Prompt C) analyze transcripts and produce judgments
  - Including various captured logs, etc.
- Inject new participants mid-conversation
  - Optionally having seen things like judge outputs and captured logs, etc.
- Run fixed rounds or open-ended interactions
- Support tools/function calling provider-agnostically

Host projects (like the GTD test harness) compose these primitives into domain-specific workflows.

---

## Core Architecture

### Primitives (What the Library Provides)

1. **Provider Adapters** — Normalize different LLM APIs
   - `ProviderAdapter` base class with `send_chat()` method
   - Production implementations: OpenAI, Anthropic, xAI, Google Gemini
   - Handles provider-specific quirks (tool calling formats, message roles, etc.)

2. **Conversation State** — Track interactions
   - `ChatMessage` — normalized message with role + content
   - `ConversationTranscript` — sequence of turns
   - `ToolInvocationRecord` — executed tool calls with results/errors

3. **Participant** — A model + prompt + adapter
   - Represents one "actor" in a conversation
   - Configurable: model name, temperature, system prompts
   - Can have tool access or not

4. **Orchestrators** — Wire up interaction patterns
   - `TurnRunner` — Execute a single model turn with tool loop
   - `ConversationRunner` — Manage back-and-forth between participants
   - `TranscriptAnalyzer` — Let a model process conversation history

5. **Tool System** — Optional function calling
   - `ToolDefinition` — JSON Schema-based tool spec
   - `ToolExecutor` protocol — Host provides actual execution
   - Library handles tool loop (call → execute → continue)

### What's NOT in the Library

- Domain-specific prompts or test scenarios
- MCP client implementation (but supports MCP via ToolExecutor)
- Graph memory, ontology, or any domain logic
- Opinionated "assistant" or "judge" roles (those are examples)

---

## Package Layout (Revised)

```text
multi_agent_harness/
  __init__.py
  config.py                # ParticipantConfig, ProviderConfig

  adapters/
    __init__.py
    base.py                # ProviderAdapter + core types
    openai.py              # OpenAI implementation
    anthropic.py           # Anthropic implementation
    xai.py                 # xAI implementation
    gemini.py              # Google Gemini implementation

  conversation/
    __init__.py
    participant.py         # Participant class
    transcript.py          # Transcript + turn models
    turn_runner.py         # Single turn execution with tools
    conversation_runner.py # Multi-turn orchestration
    analyzer.py            # Transcript analysis utilities

  tools/
    __init__.py
    base.py                # ToolDefinition, ToolExecutor protocol

  examples/
    __init__.py
    assistant.py           # Example: assistant with tools
    judge.py               # Example: transcript evaluator
    debate.py              # Example: two models arguing
    interrogator.py        # Example: follow-up questioner

docs/
  ARCHITECTURE.md          # Design philosophy + diagrams
  USAGE.md                 # Getting started guide
  EXTENDING.md             # How to add new providers
  EXAMPLES.md              # Cookbook of patterns
```

---

## Implementation Roadmap

### Phase 1: Core Primitives ✅ (Partially Done)

**Status:** OpenAI adapter + basic types exist, needs refactoring

- [x] `adapters/base.py` — ChatMessage, ToolDefinition, ToolCall, ProviderAdapter
- [x] `adapters/openai.py` — OpenAI implementation
- [x] `config.py` — ParticipantConfig (currently RoleModelConfig)
- [ ] Refactor `assistant.py` into `turn_runner.py` (generic, not role-specific)
- [ ] Create `conversation/participant.py`
- [ ] Create `conversation/transcript.py` (extract from engines/base.py)

**Issues to Fix:**
1. **Type safety:** `ChatMessage.content: Any` → Union[str, dict] with proper structure
2. **Tool executor typing:** Add `Callable[[str, dict], Any]` type alias
3. **Tool call ID validation:** Don't fall back to `call.name` as ID
4. **Error handling:** Let errors propagate naturally; don't wrap unnecessarily

### Phase 2: Additional Providers

- [ ] `adapters/anthropic.py`
  - Handle their message format (content blocks)
  - Tool use → tool_result round trip
  - System prompt injection

- [ ] `adapters/xai.py`
  - OpenAI-compatible API (likely similar to openai.py)

- [ ] `adapters/gemini.py`
  - Google's content/parts format
  - Function calling via generateContent

**Testing:** Mock provider responses, validate payload conversion

### Phase 3: Orchestration Layer

- [ ] `conversation/turn_runner.py`
  - Extract from current `assistant.py`
  - Generic: takes Participant + tools + history → response
  - Handles tool loop until completion or max_steps

- [ ] `conversation/conversation_runner.py`
  - New: manage (Participant A ↔ Participant B) interaction
  - Options:
    - `max_turns: int` — fixed rounds
    - `stop_condition: Callable` — custom termination
    - `starting_message: str` — who starts and with what

- [ ] `conversation/analyzer.py`
  - New: helper for "model sees transcript, produces output"
  - Use case: judge, summarizer, interrogator
  - Input: transcript, analysis prompt, participant
  - Output: structured or unstructured response

### Phase 4: Examples & Documentation

- [ ] `examples/assistant.py`
  - Current assistant.py behavior as an example
  - Shows: system prompts + tools + turn runner

- [ ] `examples/debate.py`
  - Two models argue for N rounds
  - Demonstrates ConversationRunner

- [ ] `examples/judge.py`
  - Third model scores a transcript
  - Demonstrates TranscriptAnalyzer

- [ ] `examples/interrogator.py`
  - Model asks follow-up questions after seeing transcript
  - Demonstrates: analyzer → inject into conversation

- [ ] `docs/ARCHITECTURE.md`
  - Design philosophy (primitives, not prescriptions)
  - Component diagram
  - Extension points

- [ ] `docs/USAGE.md`
  - Installation
  - Quick start (simple conversation)
  - Tool usage
  - Multi-model patterns

- [ ] `docs/EXTENDING.md`
  - How to implement ProviderAdapter
  - Testing adapters
  - Contributing guidelines

### Phase 5: Testing & Infrastructure

- [ ] Add dependencies to pyproject.toml:
  ```toml
  dependencies = [
    "openai>=1.0.0",
    # Optional deps for other providers
  ]
  [project.optional-dependencies]
  anthropic = ["anthropic>=0.18.0"]
  google = ["google-generativeai>=0.3.0"]
  dev = ["pytest>=7.0", "mypy>=1.0", "black", "ruff"]
  ```

- [ ] Create .gitignore
- [ ] Add mypy.ini with strict type checking
- [ ] Write unit tests:
  - Adapter payload conversion (no network)
  - TurnRunner logic
  - ConversationRunner state management
  - Tool loop behavior

- [ ] Write integration tests (with recorded responses)
- [ ] Add GitHub Actions CI
- [ ] Add pre-commit hooks (black, ruff, mypy)

---

## Usage Examples (Target API)

### Example 1: Simple Conversation

```python
from multi_agent_harness import Participant, ConversationRunner
from multi_agent_harness.adapters import OpenAIAdapter

# Define participants
alice = Participant(
    name="Alice",
    adapter=OpenAIAdapter(),
    model="gpt-4o-mini",
    system_prompts=["You are a helpful assistant named Alice."]
)

bob = Participant(
    name="Bob",
    adapter=OpenAIAdapter(),
    model="gpt-4o-mini",
    system_prompts=["You are Bob. You like to ask clarifying questions."]
)

# Run conversation
runner = ConversationRunner(participants=[alice, bob])
transcript = runner.run(
    starting_message="What's the capital of France?",
    starting_participant=alice,
    max_turns=4
)

# transcript.turns contains full conversation history
```

### Example 2: Assistant with Tools

```python
from multi_agent_harness import Participant, TurnRunner, ToolDefinition

def execute_tool(name: str, args: dict) -> dict:
    if name == "add":
        return {"result": args["a"] + args["b"]}
    raise ValueError(f"Unknown tool: {name}")

tools = [
    ToolDefinition(
        name="add",
        description="Add two numbers",
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

assistant = Participant(
    name="Assistant",
    adapter=OpenAIAdapter(),
    model="gpt-4o-mini",
    system_prompts=["You are a calculator assistant."]
)

runner = TurnRunner(participant=assistant, tools=tools, tool_executor=execute_tool)
response = runner.run_turn(
    history=[],
    user_message="What is 15 + 27?",
    max_tool_steps=5
)

print(response.message.content)  # "42"
```

### Example 3: Judge Pattern

```python
from multi_agent_harness import TranscriptAnalyzer

judge = Participant(
    name="Judge",
    adapter=OpenAIAdapter(),
    model="gpt-4o",
    system_prompts=[
        "You are an impartial judge. Score the assistant's helpfulness from 1-10.",
        "Output JSON: {\"score\": <number>, \"reasoning\": \"<text>\"}"
    ]
)

analyzer = TranscriptAnalyzer(participant=judge)
verdict = analyzer.analyze(
    transcript=transcript,
    response_format={"type": "json_schema", "json_schema": {...}}
)

print(verdict.score)
```

### Example 4: Participant Injection

```python
# After Alice and Bob talk for 3 rounds...
transcript = runner.run(participants=[alice, bob], max_turns=3)

# Charlie analyzes the conversation
charlie = Participant(
    name="Charlie",
    adapter=OpenAIAdapter(),
    model="gpt-4o",
    system_prompts=["You are a moderator. Summarize key disagreements."]
)

summary = analyzer.analyze(transcript, charlie)

# Charlie replaces Bob and continues
runner2 = ConversationRunner(participants=[alice, charlie])
continued = runner2.run(
    starting_message=f"Based on this summary: {summary}, let's continue.",
    initial_transcript=transcript,  # Carry forward history
    max_turns=3
)
```

---

## Technical Considerations

### Type Safety

- Replace `content: Any` with proper Union types
- `ToolExecutor = Callable[[str, dict[str, Any]], Any]`
- Strict mypy checking with `--strict` flag

### Error Handling Philosophy

**Don't wrap errors unnecessarily.** Let them propagate:

```python
# ❌ BAD - wrapping adds no value
try:
    result = execute_tool(name, args)
except Exception as e:
    raise RuntimeError(f"Tool execution failed: {e}") from e

# ✅ GOOD - just let it raise
result = execute_tool(name, args)
```

Only catch and handle when you can meaningfully recover or add context.

### Tool Call IDs

OpenAI requires tool_call_id for tool result messages. Validate at API boundary:

```python
if call.call_id is None:
    raise ValueError(f"Tool call {call.name} missing required call_id")
```

### Adapter Testing

Each adapter should have:
1. Unit tests for message/tool conversion (no network)
2. Recorded response tests (replay real API responses)
3. Live integration tests (marked as slow/optional)

### Multi-Provider Consistency

Adapters must normalize:
- Message roles (assistant, user, system, tool)
- Tool call request/response format
- Error conditions
- Stop reasons

---

## Open Questions

1. **Streaming support?** Not in v1, but design should allow it
2. **Async/await?** Could add async variants later
3. **Token counting/budgets?** Useful but maybe separate concern
4. **Caching/memoization?** Host project responsibility
5. **Logging/observability?** Provide hooks but don't prescribe

---

## Success Criteria

Library is "done" when:

- ✅ All 4 providers work (OpenAI, Anthropic, xAI, Gemini)
- ✅ Can run (Model A ↔ Model B) conversation with N rounds
- ✅ Can analyze transcripts with Model C
- ✅ Can inject new participant mid-conversation
- ✅ Tools work across all providers
- ✅ 80%+ test coverage
- ✅ Complete docs (Architecture, Usage, Extending, Examples)
- ✅ Type-checks with mypy --strict
- ✅ GTD project successfully migrates to use this library

---

## Migration from Current Code

1. Rename `RoleModelConfig` → `ParticipantConfig`
2. Extract `assistant.py` generic logic → `turn_runner.py`
3. Move engines/base.py transcript models → `conversation/transcript.py`
4. Delete `RoleEngine` base class (not needed with new design)
5. Convert `assistant.py` to `examples/assistant.py`
6. Update GTD repo imports once library stabilizes
