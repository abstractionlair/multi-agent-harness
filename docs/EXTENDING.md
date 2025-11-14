# Extending the Library

This guide covers how to add new provider adapters, contribute to the library, and extend functionality.

---

## Table of Contents

1. [Implementing a Provider Adapter](#implementing-a-provider-adapter)
2. [Testing Your Adapter](#testing-your-adapter)
3. [Contributing Guidelines](#contributing-guidelines)
4. [Adding New Orchestrators](#adding-new-orchestrators)
5. [Advanced Customization](#advanced-customization)

---

## Implementing a Provider Adapter

Provider adapters normalize different LLM APIs into a common interface. Here's how to implement one.

### Step 1: Understand the Interface

All adapters must implement the `ProviderAdapter` abstract base class:

```python
from multi_agent_harness.adapters.base import ProviderAdapter, ChatResponse

class ProviderAdapter(ABC):
    """Abstract base class for provider-specific adapters."""

    provider_name: str  # Class variable

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = api_key

    @abstractmethod
    def send_chat(
        self,
        role_config: RoleModelConfig,
        messages: Sequence[ChatMessage],
        tools: Optional[Iterable[ToolDefinition]] = None,
        response_format: Optional[ResponseFormat] = None,
        tool_choice: Optional[str] = None,
    ) -> ChatResponse:
        """Execute a chat completion request."""

    @abstractmethod
    def supports_tools(self) -> bool:
        """Return True if provider supports function/tool calling."""
```

### Step 2: Create Your Adapter File

Create a new file in `src/multi_agent_harness/adapters/`:

```bash
touch src/multi_agent_harness/adapters/my_provider.py
```

### Step 3: Implement the Adapter

Here's a complete example for a hypothetical provider:

```python
"""My Provider adapter implementation."""

from __future__ import annotations

import os
import json
from typing import Iterable, Optional, Sequence, Any, Dict

from .base import (
    ChatMessage,
    ChatResponse,
    ProviderAdapter,
    ResponseFormat,
    ToolDefinition,
    ToolCall,
)
from ..config import RoleModelConfig

try:
    # Try importing the provider's SDK
    from my_provider_sdk import MyProviderClient
except ImportError:
    MyProviderClient = None


class MyProviderAdapter(ProviderAdapter):
    """Adapter for My Provider's LLM API."""

    provider_name = "my_provider"

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ) -> None:
        """Initialize the adapter.

        Args:
            api_key: API key (defaults to MY_PROVIDER_API_KEY env var)
            base_url: Optional custom API endpoint
        """
        super().__init__(api_key or os.environ.get("MY_PROVIDER_API_KEY"))

        if not self.api_key:
            raise ValueError(
                "API key required. Set MY_PROVIDER_API_KEY env var or pass api_key"
            )

        # Initialize SDK client if available
        self._client = MyProviderClient(
            api_key=self.api_key,
            base_url=base_url
        ) if MyProviderClient else None

    def send_chat(
        self,
        role_config: RoleModelConfig,
        messages: Sequence[ChatMessage],
        tools: Optional[Iterable[ToolDefinition]] = None,
        response_format: Optional[ResponseFormat] = None,
        tool_choice: Optional[str] = None,
    ) -> ChatResponse:
        """Send a chat completion request to My Provider.

        Args:
            role_config: Model configuration (temperature, etc.)
            messages: Conversation history
            tools: Optional tool definitions
            response_format: Optional response formatting
            tool_choice: Tool selection strategy ("auto", "required", "none")

        Returns:
            ChatResponse with normalized message and tool calls
        """
        # 1. Build the request payload
        payload = self._build_payload(
            role_config, messages, tools, response_format, tool_choice
        )

        # 2. Make the API call
        if self._client:
            # Use SDK
            response = self._client.chat.create(**payload)
            return self._convert_sdk_response(response)
        else:
            # Fallback to REST API
            response = self._make_rest_request(payload)
            return self._convert_rest_response(response)

    def supports_tools(self) -> bool:
        """My Provider supports function calling."""
        return True

    # ─── Private Helper Methods ───────────────────────────────────────

    def _build_payload(
        self,
        role_config: RoleModelConfig,
        messages: Sequence[ChatMessage],
        tools: Optional[Iterable[ToolDefinition]],
        response_format: Optional[ResponseFormat],
        tool_choice: Optional[str],
    ) -> Dict[str, Any]:
        """Build the API request payload."""
        payload: Dict[str, Any] = {
            "model": role_config.model,
            "messages": [self._convert_message(msg) for msg in messages],
            "temperature": role_config.temperature,
        }

        # Add optional parameters
        if role_config.max_tokens:
            payload["max_tokens"] = role_config.max_tokens

        if tools:
            payload["tools"] = [self._convert_tool(tool) for tool in tools]
            if tool_choice:
                payload["tool_choice"] = tool_choice

        if response_format and response_format.type == "json_schema":
            # Provider-specific JSON schema format
            payload["response_format"] = {
                "type": "json",
                "schema": response_format.json_schema
            }

        return payload

    @staticmethod
    def _convert_message(message: ChatMessage) -> Dict[str, Any]:
        """Convert normalized ChatMessage to provider format.

        Handle different message types:
        - system: System prompts
        - user: User messages
        - assistant: Assistant responses (may include tool calls)
        - tool: Tool results
        """
        # System message
        if message.role == "system":
            return {"role": "system", "content": message.content}

        # User message
        if message.role == "user":
            return {"role": "user", "content": message.content}

        # Assistant message with tool calls
        if message.role == "assistant" and isinstance(message.content, dict):
            if "tool_calls" in message.content:
                return {
                    "role": "assistant",
                    "content": message.content.get("content", ""),
                    "tool_calls": message.content["tool_calls"],
                }

        # Tool result message
        if message.role == "tool":
            if not isinstance(message.content, dict):
                raise ValueError("Tool messages must have dict content")

            return {
                "role": "tool",
                "tool_call_id": message.content["tool_call_id"],
                "content": message.content.get("content", "")
            }

        # Default: simple message
        return {"role": message.role, "content": message.content}

    @staticmethod
    def _convert_tool(tool: ToolDefinition) -> Dict[str, Any]:
        """Convert normalized ToolDefinition to provider format."""
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema,
            }
        }

    @staticmethod
    def _convert_sdk_response(response: Any) -> ChatResponse:
        """Convert SDK response to normalized ChatResponse."""
        # Extract message content
        message = response.choices[0].message
        content = message.content or ""

        # Extract tool calls if present
        tool_calls = []
        if message.tool_calls:
            for tc in message.tool_calls:
                tool_calls.append(
                    ToolCall(
                        name=tc.function.name,
                        arguments=json.loads(tc.function.arguments or "{}"),
                        call_id=tc.id,
                    )
                )

        return ChatResponse(
            message=ChatMessage(role="assistant", content=content),
            tool_calls=tuple(tool_calls),
            raw=response,
        )

    @staticmethod
    def _convert_rest_response(response: Dict[str, Any]) -> ChatResponse:
        """Convert REST response to normalized ChatResponse."""
        choice = response["choices"][0]
        message = choice["message"]

        content = message.get("content", "")
        tool_calls = []

        for tc in message.get("tool_calls", []):
            function = tc.get("function", {})
            tool_calls.append(
                ToolCall(
                    name=function.get("name"),
                    arguments=json.loads(function.get("arguments", "{}")),
                    call_id=tc.get("id"),
                )
            )

        return ChatResponse(
            message=ChatMessage(role="assistant", content=content),
            tool_calls=tuple(tool_calls),
            raw=response,
        )

    def _make_rest_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Make a REST API request (fallback when SDK unavailable)."""
        import requests

        response = requests.post(
            "https://api.myprovider.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        return response.json()
```

### Step 4: Handle Provider-Specific Quirks

Different providers have different conventions. Here are common patterns:

#### System Prompts

```python
# OpenAI: Separate system messages
messages = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Hello"}
]

# Anthropic: system parameter
payload = {
    "model": "claude-3-5-sonnet-20241022",
    "system": "You are helpful.",
    "messages": [{"role": "user", "content": "Hello"}]
}

# Your adapter should normalize this:
def _build_payload(self, role_config, messages, ...):
    # Extract system prompts
    system_prompts = [m for m in messages if m.role == "system"]
    other_messages = [m for m in messages if m.role != "system"]

    payload = {
        "model": role_config.model,
        "system": "\n".join(m.content for m in system_prompts),
        "messages": [self._convert_message(m) for m in other_messages]
    }
    return payload
```

#### Tool Call Formats

```python
# OpenAI format
{
    "tool_calls": [
        {
            "id": "call_123",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": '{"location": "Paris"}'
            }
        }
    ]
}

# Anthropic format
{
    "content": [
        {
            "type": "tool_use",
            "id": "toolu_123",
            "name": "get_weather",
            "input": {"location": "Paris"}
        }
    ]
}

# Your adapter normalizes to:
ToolCall(
    name="get_weather",
    arguments={"location": "Paris"},
    call_id="call_123" or "toolu_123"
)
```

#### Content Blocks

```python
# Some providers use content blocks:
{
    "content": [
        {"type": "text", "text": "The weather is"},
        {"type": "tool_use", "name": "get_weather", ...}
    ]
}

# Your adapter should extract text:
def _extract_text_content(content):
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        texts = [block["text"] for block in content if block.get("type") == "text"]
        return "\n".join(texts)

    return ""
```

### Step 5: Register Your Adapter

Add your adapter to the module exports:

**`src/multi_agent_harness/adapters/__init__.py`:**
```python
from .openai import OpenAIAdapter
from .anthropic import AnthropicAdapter
from .xai import XAIAdapter
from .gemini import GeminiAdapter
from .my_provider import MyProviderAdapter  # Add this

__all__ = [
    "OpenAIAdapter",
    "AnthropicAdapter",
    "XAIAdapter",
    "GeminiAdapter",
    "MyProviderAdapter",  # Add this
]
```

---

## Testing Your Adapter

### Unit Tests (No Network Calls)

Test message conversion without making actual API calls:

```python
# tests/adapters/test_my_provider.py
import pytest
from multi_agent_harness.adapters.my_provider import MyProviderAdapter
from multi_agent_harness.adapters.base import ChatMessage, ToolDefinition


class TestMyProviderAdapter:
    def test_message_conversion_user(self):
        """Test user message conversion."""
        adapter = MyProviderAdapter(api_key="test-key")
        message = ChatMessage(role="user", content="Hello")

        converted = adapter._convert_message(message)

        assert converted == {"role": "user", "content": "Hello"}

    def test_message_conversion_system(self):
        """Test system message conversion."""
        adapter = MyProviderAdapter(api_key="test-key")
        message = ChatMessage(role="system", content="You are helpful.")

        converted = adapter._convert_message(message)

        assert converted == {"role": "system", "content": "You are helpful."}

    def test_tool_conversion(self):
        """Test tool definition conversion."""
        adapter = MyProviderAdapter(api_key="test-key")
        tool = ToolDefinition(
            name="get_weather",
            description="Get weather for a location",
            input_schema={
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                },
                "required": ["location"]
            }
        )

        converted = adapter._convert_tool(tool)

        assert converted["type"] == "function"
        assert converted["function"]["name"] == "get_weather"
        assert "location" in converted["function"]["parameters"]["properties"]

    def test_sdk_response_conversion(self):
        """Test SDK response parsing."""
        # Create a mock response object
        class MockMessage:
            content = "The weather is sunny."
            tool_calls = None

        class MockChoice:
            message = MockMessage()

        class MockResponse:
            choices = [MockChoice()]

        adapter = MyProviderAdapter(api_key="test-key")
        response = adapter._convert_sdk_response(MockResponse())

        assert response.message.role == "assistant"
        assert response.message.content == "The weather is sunny."
        assert len(response.tool_calls) == 0
```

### Integration Tests (Mocked Responses)

Test the full flow with recorded API responses:

```python
from unittest.mock import Mock, patch
import json


class TestMyProviderIntegration:
    @patch("my_provider_sdk.MyProviderClient")
    def test_send_chat_success(self, mock_client_class):
        """Test successful chat completion."""
        # Setup mock
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "Hello!"
        mock_response.choices[0].message.tool_calls = None

        mock_client = Mock()
        mock_client.chat.create.return_value = mock_response
        mock_client_class.return_value = mock_client

        # Test
        from multi_agent_harness import ParticipantConfig
        adapter = MyProviderAdapter(api_key="test-key")
        config = ParticipantConfig(model="my-model")
        messages = [ChatMessage(role="user", content="Hi")]

        response = adapter.send_chat(config, messages)

        assert response.message.content == "Hello!"
        mock_client.chat.create.assert_called_once()

    @patch("my_provider_sdk.MyProviderClient")
    def test_send_chat_with_tools(self, mock_client_class):
        """Test chat completion with tool calls."""
        # Setup mock with tool call
        mock_tool_call = Mock()
        mock_tool_call.id = "call_123"
        mock_tool_call.function = Mock()
        mock_tool_call.function.name = "get_weather"
        mock_tool_call.function.arguments = json.dumps({"location": "Paris"})

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = ""
        mock_response.choices[0].message.tool_calls = [mock_tool_call]

        mock_client = Mock()
        mock_client.chat.create.return_value = mock_response
        mock_client_class.return_value = mock_client

        # Test
        adapter = MyProviderAdapter(api_key="test-key")
        config = ParticipantConfig(model="my-model")
        messages = [ChatMessage(role="user", content="What's the weather?")]
        tools = [
            ToolDefinition(
                name="get_weather",
                description="Get weather",
                input_schema={"type": "object", "properties": {"location": {"type": "string"}}}
            )
        ]

        response = adapter.send_chat(config, messages, tools=tools)

        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "get_weather"
        assert response.tool_calls[0].arguments == {"location": "Paris"}
```

### Live Tests (Optional)

Mark these as slow/optional since they hit real APIs:

```python
import pytest
import os


@pytest.mark.slow
@pytest.mark.skipif(
    not os.getenv("MY_PROVIDER_API_KEY"),
    reason="MY_PROVIDER_API_KEY not set"
)
class TestMyProviderLive:
    def test_real_api_call(self):
        """Test against real API (slow, requires API key)."""
        adapter = MyProviderAdapter()  # Uses env var
        config = ParticipantConfig(model="my-model", temperature=0.7)
        messages = [
            ChatMessage(role="system", content="You are helpful."),
            ChatMessage(role="user", content="Say 'test passed'")
        ]

        response = adapter.send_chat(config, messages)

        assert "test" in response.message.content.lower()
```

---

## Contributing Guidelines

### Before Contributing

1. **Check existing issues/PRs** to avoid duplicates
2. **Open an issue** to discuss major changes
3. **Follow the code style** (black, ruff)
4. **Add tests** for new functionality
5. **Update documentation** as needed

### Development Setup

```bash
# Clone the repo
git clone https://github.com/yourusername/multi-agent-harness.git
cd multi-agent-harness

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest

# Type check
mypy src/

# Format code
black src/ tests/
ruff check src/ tests/ --fix
```

### Pull Request Process

1. **Create a feature branch**
   ```bash
   git checkout -b feature/my-provider-adapter
   ```

2. **Implement your changes**
   - Write code
   - Add tests (aim for 80%+ coverage)
   - Update docs if needed

3. **Test thoroughly**
   ```bash
   # Run all tests
   pytest

   # Run with coverage
   pytest --cov=multi_agent_harness --cov-report=html

   # Type check
   mypy src/
   ```

4. **Format and lint**
   ```bash
   black src/ tests/
   ruff check src/ tests/ --fix
   ```

5. **Commit with clear messages**
   ```bash
   git add .
   git commit -m "Add MyProvider adapter with tool support"
   ```

6. **Push and create PR**
   ```bash
   git push origin feature/my-provider-adapter
   ```

   Then open a PR on GitHub with:
   - Clear description of changes
   - Link to related issues
   - Test results/coverage

### Code Style

- **Black** for formatting (line length: 100)
- **Ruff** for linting
- **Type hints** for all public APIs
- **Docstrings** for all public classes/methods (Google style)

```python
def send_chat(
    self,
    role_config: RoleModelConfig,
    messages: Sequence[ChatMessage],
    tools: Optional[Iterable[ToolDefinition]] = None,
) -> ChatResponse:
    """Send a chat completion request.

    Args:
        role_config: Model configuration (temperature, etc.)
        messages: Conversation history
        tools: Optional tool definitions

    Returns:
        ChatResponse with normalized message and tool calls

    Raises:
        ValueError: If messages are invalid
        RuntimeError: If API call fails
    """
```

---

## Adding New Orchestrators

You can create custom orchestration patterns by composing existing primitives:

```python
from multi_agent_harness import (
    Participant,
    ConversationRunner,
    TranscriptAnalyzer,
    ConversationTranscript,
)


class DebateWithJudge:
    """Two participants debate, third participant judges."""

    def __init__(
        self,
        pro: Participant,
        con: Participant,
        judge: Participant,
        rounds: int = 3
    ):
        self.pro = pro
        self.con = con
        self.judge = judge
        self.rounds = rounds

    def run(self, topic: str) -> dict:
        """Run debate and return transcript + verdict."""
        # 1. Run debate
        conversation = ConversationRunner(participants=[self.pro, self.con])
        transcript = conversation.run(
            starting_message=f"Argue in favor of: {topic}",
            starting_participant=self.pro,
            max_turns=self.rounds * 2
        )

        # 2. Judge evaluates
        analyzer = TranscriptAnalyzer(participant=self.judge)
        verdict = analyzer.analyze(
            transcript=transcript,
            analysis_prompt="Who made the stronger argument? Why?"
        )

        return {
            "transcript": transcript,
            "verdict": verdict.message.content
        }


# Usage
debate = DebateWithJudge(pro=alice, con=bob, judge=charlie, rounds=3)
result = debate.run("Remote work is better than office work")
```

---

## Advanced Customization

### Custom Tool Executors

Wrap MCP servers, API clients, or function registries:

```python
from multi_agent_harness.adapters.base import ToolExecutor
from typing import Any


class MCPToolExecutor:
    """Execute tools via MCP server."""

    def __init__(self, mcp_client):
        self.mcp_client = mcp_client

    def __call__(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Execute a tool via MCP."""
        return self.mcp_client.call_tool(tool_name, arguments)


# Usage
mcp_executor = MCPToolExecutor(my_mcp_client)
runner = TurnRunner(participant=assistant, tools=tools, tool_executor=mcp_executor)
```

### Custom Response Parsing

For structured outputs:

```python
import json
from multi_agent_harness import TranscriptAnalyzer, ResponseFormat


def parse_json_response(response):
    """Parse and validate JSON response."""
    try:
        data = json.loads(response.message.content)
        # Validate structure
        assert "score" in data and "reasoning" in data
        return data
    except (json.JSONDecodeError, AssertionError) as e:
        raise ValueError(f"Invalid response format: {e}")


analyzer = TranscriptAnalyzer(participant=judge)
verdict = analyzer.analyze(
    transcript=transcript,
    response_format=ResponseFormat(type="json_object")
)

result = parse_json_response(verdict)
```

### Custom Participant Factories

For common configurations:

```python
def create_coding_assistant(provider: str = "openai") -> Participant:
    """Factory for coding assistant with standard prompts."""
    adapters = {
        "openai": OpenAIAdapter(),
        "anthropic": AnthropicAdapter(),
        "gemini": GeminiAdapter(),
    }

    return Participant(
        name="CodingAssistant",
        adapter=adapters[provider],
        model="gpt-4o" if provider == "openai" else "claude-sonnet-4-5-20250929",
        system_prompts=[
            "You are an expert software engineer.",
            "Provide working code examples.",
            "Explain your reasoning clearly."
        ]
    )


# Usage
assistant = create_coding_assistant(provider="anthropic")
```

---

## Resources

- **GitHub:** [https://github.com/yourusername/multi-agent-harness](https://github.com/yourusername/multi-agent-harness)
- **Issues:** [https://github.com/yourusername/multi-agent-harness/issues](https://github.com/yourusername/multi-agent-harness/issues)
- **Architecture:** [`ARCHITECTURE.md`](ARCHITECTURE.md)
- **Usage Guide:** [`USAGE.md`](USAGE.md)

---

## Questions?

If you have questions about extending the library:

1. Check existing adapters in `src/multi_agent_harness/adapters/`
2. Review the examples in `src/multi_agent_harness/examples/`
3. Open an issue on GitHub with the "question" label
4. Join our discussions (link TBD)

Happy extending!
