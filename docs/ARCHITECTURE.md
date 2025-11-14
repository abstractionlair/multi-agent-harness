# Architecture

## Design Philosophy

The multi-agent harness is built on a **primitives-first** philosophy: we provide flexible, composable building blocks rather than prescriptive frameworks.

### Primitives Over Frameworks

**What we provide:**
- Normalized adapters for different LLM providers
- Conversation state management (transcripts, turns, messages)
- Participant abstraction (model + prompts + adapter)
- Orchestration utilities (turn execution, multi-turn conversations, analysis)
- Tool system protocol (definition + execution interface)

**What we don't provide:**
- Domain-specific prompts or scenarios
- Opinionated "assistant" or "judge" roles
- Graph memory or ontology systems
- MCP client implementation (though we support MCP via `ToolExecutor`)

This design lets host projects compose primitives into domain-specific workflows while keeping the library focused and reusable.

---

## Core Components

### 1. Provider Adapters (`adapters/`)

**Purpose:** Normalize different LLM API formats into a common interface.

```
ProviderAdapter (base.py)
    ├── send_chat(config, messages, tools, ...) -> ChatResponse
    └── Implementations:
        ├── OpenAIAdapter (openai.py)
        ├── AnthropicAdapter (anthropic.py)
        ├── XAIAdapter (xai.py)
        └── GeminiAdapter (gemini.py)
```

**Responsibilities:**
- Convert normalized `ChatMessage` → provider-specific format
- Convert provider response → normalized `ChatResponse`
- Handle provider-specific quirks (tool calling, message roles, content blocks)
- Manage API authentication and requests

**Key Types:**
- `ChatMessage` — normalized message with role + content
- `ChatResponse` — normalized response with message + tool calls + metadata
- `ToolDefinition` — JSON Schema-based tool specification
- `ToolCall` — tool invocation request from model

### 2. Conversation State (`conversation/`)

**Purpose:** Track and manage conversation history.

```
Transcript System
    ├── ChatMessage — single message (role + content)
    ├── ConversationTurn — one participant's turn (message + tool calls)
    ├── ConversationTranscript — sequence of turns
    └── ToolInvocationRecord — executed tool call (name + args + result/error)
```

**Design Notes:**
- `ChatMessage` is the adapter-level primitive (what providers see)
- `ConversationTurn` is the orchestration-level primitive (what transcripts track)
- Tool results are embedded in the conversation flow, not stored separately
- Transcripts are immutable-ish (add turns, don't modify existing ones)

### 3. Participant (`conversation/participant.py`)

**Purpose:** Represent one "actor" in a conversation (model + prompts + adapter).

```python
Participant
    ├── name: str
    ├── adapter: ProviderAdapter
    ├── model: str
    ├── system_prompts: List[ChatMessage]
    └── model_config: ParticipantConfig (temperature, max_tokens, etc.)
```

**Key Insight:** A participant is configuration, not behavior. Behavior comes from orchestrators (TurnRunner, ConversationRunner, etc.).

### 4. Orchestrators (`conversation/`)

**Purpose:** Wire up interaction patterns between participants and models.

#### TurnRunner (`turn_runner.py`)

Executes a single model turn with tool loop:

```
run_turn(history, user_message, tools) -> ChatResponse
    1. Build message list (system + history + user_message)
    2. Call adapter.send_chat()
    3. If tool calls → execute → add results → repeat
    4. Return final response
```

**Features:**
- Tool loop with configurable max steps
- Error handling for tool execution
- Tool choice control ("auto", "required", "none")
- Response format support (JSON schema, etc.)

#### ConversationRunner (`conversation_runner.py`)

Manages multi-turn conversations between 2+ participants:

```
run(starting_message, max_turns, stop_condition) -> Transcript
    1. Initialize with starting message
    2. Loop: Current participant responds
    3. Response becomes message for next participant
    4. Round-robin until max_turns or stop_condition
    5. Return complete transcript
```

**Features:**
- Participant injection (continue with different participants)
- Custom stop conditions
- Initial transcript support (continue existing conversations)
- Tool support for all participants

#### TranscriptAnalyzer (`analyzer.py`)

Enables "model sees transcript → produces analysis" pattern:

```
analyze(transcript, analysis_prompt, response_format) -> ChatResponse
    1. Format transcript as readable text
    2. Build message with system prompts + transcript
    3. Send to participant's model
    4. Return analysis response
```

**Use Cases:**
- Judge: Score conversation quality
- Summarizer: Extract key points
- Interrogator: Generate follow-up questions
- Moderator: Identify disagreements

---

## Component Relationships

### Data Flow: Single Turn with Tools

```
User Message
    ↓
TurnRunner.run_turn()
    ↓
Build message list (system + history + user)
    ↓
Participant → Adapter.send_chat()
    ↓
Convert to provider format
    ↓
[Provider API Call]
    ↓
Convert response → ChatResponse
    ↓
Tool calls?
    ├─ No → Return response
    └─ Yes → Execute tools via ToolExecutor
           ↓
        Add tool results to messages
           ↓
        Loop back to Adapter.send_chat()
           ↓
        Max steps reached? Return final response
```

### Data Flow: Multi-Turn Conversation

```
Starting Message
    ↓
ConversationRunner.run()
    ↓
Loop:
    ├─ Get current participant
    ├─ Build history from transcript
    ├─ TurnRunner.run_turn(history, message)
    ├─ Add turn to transcript
    ├─ Response → message for next participant
    ├─ Move to next participant (round-robin)
    └─ Check stop conditions
    ↓
Return ConversationTranscript
```

### Data Flow: Transcript Analysis

```
ConversationTranscript
    ↓
TranscriptAnalyzer.analyze()
    ↓
Format transcript as text (Turn 1: ..., Turn 2: ...)
    ↓
Build analysis message (system + prompt + transcript)
    ↓
Participant → Adapter.send_chat()
    ↓
Return analysis ChatResponse
```

---

## Extension Points

### 1. Adding New Providers

Implement `ProviderAdapter` interface:

```python
class MyProviderAdapter(ProviderAdapter):
    def send_chat(
        self,
        role_config: ParticipantConfig,
        messages: Sequence[ChatMessage],
        tools: Optional[Sequence[ToolDefinition]] = None,
        tool_choice: str = "auto",
        response_format: Optional[ResponseFormat] = None,
    ) -> ChatResponse:
        # 1. Convert messages → provider format
        # 2. Make API call
        # 3. Convert response → ChatResponse
        # 4. Return
```

See [`EXTENDING.md`](EXTENDING.md) for detailed guide.

### 2. Custom Orchestrators

Create new orchestration patterns by composing primitives:

```python
class DebateOrchestrator:
    """Two models argue, third model judges."""

    def __init__(self, pro: Participant, con: Participant, judge: Participant):
        self.conversation_runner = ConversationRunner([pro, con])
        self.analyzer = TranscriptAnalyzer(judge)

    def run(self, topic: str, rounds: int):
        transcript = self.conversation_runner.run(
            starting_message=f"Argue for: {topic}",
            max_turns=rounds * 2
        )

        verdict = self.analyzer.analyze(
            transcript=transcript,
            analysis_prompt="Who made the stronger argument?"
        )

        return transcript, verdict
```

### 3. Custom Tool Executors

Implement the `ToolExecutor` protocol:

```python
ToolExecutor = Callable[[str, dict[str, Any]], Any]

def my_tool_executor(tool_name: str, arguments: dict[str, Any]) -> Any:
    if tool_name == "search":
        return search_engine.query(arguments["query"])
    elif tool_name == "calculate":
        return eval(arguments["expression"])  # Don't actually do this!
    else:
        raise ValueError(f"Unknown tool: {tool_name}")
```

Can wrap MCP servers, function registries, API clients, etc.

### 4. Custom Stop Conditions

Use `StopCondition` type for conversation termination logic:

```python
StopCondition = Callable[[ConversationTranscript], bool]

def stop_when_agreement_reached(transcript: ConversationTranscript) -> bool:
    """Stop when both parties say 'I agree'."""
    if len(transcript.turns) < 2:
        return False

    last_two = transcript.turns[-2:]
    return all("agree" in turn.message.lower() for turn in last_two)

# Use in ConversationRunner
runner.run(
    starting_message="...",
    stop_condition=stop_when_agreement_reached
)
```

---

## Design Decisions

### Why Not Use Langchain/LlamaIndex?

These are excellent frameworks, but they:
- Are opinionated about application structure
- Have large dependency trees
- Mix orchestration with domain logic
- Are harder to test and debug

We wanted **minimal, testable primitives** that compose cleanly.

### Why Separate Adapters from Orchestrators?

**Separation of concerns:**
- Adapters handle provider-specific details (API format, auth, quirks)
- Orchestrators handle conversation logic (turn-taking, tool loops, analysis)

This makes both easier to test and extend independently.

### Why Not Include Prompts/Examples?

Domain-specific prompts belong in host projects. The library provides:
- Example patterns in `examples/` (assistant, judge, debate, interrogator)
- But NOT production prompts for specific domains

This keeps the library focused and prevents prompt bloat.

### Why Synchronous (Not Async)?

**Simplicity.** Async adds complexity:
- Harder to test
- Harder to debug
- Harder to integrate with synchronous code

We can add async variants later (e.g., `async_send_chat()`) without breaking the sync API.

### Why No Streaming?

**v1 scope control.** Streaming requires:
- Different return types (AsyncIterator vs. ChatResponse)
- More complex tool loop logic
- Provider-specific chunking/buffering

The architecture supports streaming (just change return types), but we defer to v2.

### Error Handling Philosophy

**Don't wrap unnecessarily.** Let errors propagate:

```python
# ❌ BAD - adds no value
try:
    result = execute_tool(name, args)
except Exception as e:
    raise RuntimeError(f"Tool execution failed: {e}") from e

# ✅ GOOD - just let it raise
result = execute_tool(name, args)
```

Only catch/handle when you can:
1. Recover gracefully
2. Add meaningful context
3. Transform error into expected format

---

## Testing Strategy

### Adapter Tests

1. **Unit tests:** Message conversion (no network)
   ```python
   def test_openai_message_conversion():
       adapter = OpenAIAdapter()
       messages = [ChatMessage(role="user", content="Hello")]
       payload = adapter._build_messages(messages)
       assert payload == [{"role": "user", "content": "Hello"}]
   ```

2. **Integration tests:** Recorded responses
   ```python
   @mock.patch("openai.Client.chat.completions.create")
   def test_openai_response_parsing(mock_create):
       mock_create.return_value = load_fixture("openai_response.json")
       response = adapter.send_chat(...)
       assert response.message.content == "Expected response"
   ```

3. **Live tests:** Real API calls (marked as slow/optional)

### Orchestrator Tests

1. **TurnRunner:** Mock adapter, verify tool loop logic
2. **ConversationRunner:** Mock participants, verify turn sequencing
3. **TranscriptAnalyzer:** Mock adapter, verify transcript formatting

### Integration Tests

End-to-end scenarios with multiple components (using mocks or recorded responses).

---

## Future Considerations

### Potential v2 Features

- **Streaming support:** `async_send_chat()` returning `AsyncIterator[ChatChunk]`
- **Token budgets:** Track and limit token usage per conversation
- **Caching:** Memoize responses for deterministic scenarios
- **Observability:** Built-in logging/tracing hooks
- **Parallel tool execution:** Execute independent tools concurrently
- **Multi-modal inputs:** Image/audio message support

### Non-Goals

These belong in host projects, not the library:

- Graph memory / knowledge bases
- Specific evaluation metrics
- Production prompt management
- Rate limiting / retry logic (use tenacity or similar)
- Cost tracking (use provider-specific tools)

---

## Summary

The multi-agent harness provides **composable primitives** for LLM orchestration:

1. **Adapters** normalize provider APIs
2. **Participants** represent model + prompts
3. **Orchestrators** wire up interaction patterns
4. **Tool system** enables function calling

Host projects compose these into domain-specific workflows. The library stays focused, testable, and extensible.

**Next steps:**
- See [`USAGE.md`](USAGE.md) for getting started
- See [`EXTENDING.md`](EXTENDING.md) for adding providers
- See `examples/` for common patterns
