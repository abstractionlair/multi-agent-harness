"""Example: Assistant with tool usage.

This example demonstrates how to use TurnRunner to create an assistant that
can execute tools/functions. It shows:
- Defining tools with JSON schemas
- Implementing a tool executor
- Running a single turn with automatic tool loop
- Handling tool call results

This pattern is useful for building assistants, agents, or any participant that
needs access to external functions or APIs.
"""

from typing import Any

from multi_agent_harness import Participant, TurnRunner
from multi_agent_harness.adapters.openai import OpenAIAdapter
from multi_agent_harness.tools import ToolDefinition


def example_calculator_assistant():
    """Example: Calculator assistant that can add and multiply numbers."""
    print("=" * 60)
    print("Example: Calculator Assistant with Tools")
    print("=" * 60)
    print()

    # Define tools the assistant can use
    tools = [
        ToolDefinition(
            name="add",
            description="Add two numbers together",
            input_schema={
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "First number"},
                    "b": {"type": "number", "description": "Second number"},
                },
                "required": ["a", "b"],
            },
        ),
        ToolDefinition(
            name="multiply",
            description="Multiply two numbers",
            input_schema={
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "First number"},
                    "b": {"type": "number", "description": "Second number"},
                },
                "required": ["a", "b"],
            },
        ),
    ]

    # Implement tool executor
    def execute_tool(name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Execute a tool and return the result.

        Args:
            name: Tool name to execute
            args: Tool arguments as a dictionary

        Returns:
            Result dictionary that will be sent back to the model

        Raises:
            ValueError: If the tool name is unknown
        """
        if name == "add":
            result = args["a"] + args["b"]
            print(f"  [Tool] add({args['a']}, {args['b']}) = {result}")
            return {"result": result}

        if name == "multiply":
            result = args["a"] * args["b"]
            print(f"  [Tool] multiply({args['a']}, {args['b']}) = {result}")
            return {"result": result}

        raise ValueError(f"Unknown tool: {name}")

    # Create participant
    assistant = Participant(
        name="Calculator",
        adapter=OpenAIAdapter(),  # Will use OPENAI_API_KEY env var
        model="gpt-4o-mini",
        system_prompts=[
            "You are a helpful calculator assistant.",
            "Use the available tools to perform calculations.",
            "Always show your work step by step.",
        ],
        temperature=0.0,  # Deterministic responses
    )

    # Create turn runner with tools
    runner = TurnRunner(
        participant=assistant,
        tools=tools,
        tool_executor=execute_tool,
    )

    # Example queries
    queries = [
        "What is 15 + 27?",
        "What is (10 + 5) * 3?",
        "Calculate 8 * 7, then add 12 to the result.",
    ]

    for query in queries:
        print(f"User: {query}")

        # This would execute the turn (commented to avoid API calls without key)
        # response = runner.run_turn(
        #     history=[],  # No prior conversation
        #     user_message=query,
        #     max_tool_steps=5,  # Allow up to 5 tool calls
        # )
        # print(f"Assistant: {response.message.content}")
        # print()

        print(f"  (Would execute with tools: {[t.name for t in tools]})")
        print()

    print("=" * 60)
    print()


def example_api_assistant():
    """Example: Assistant that can make API calls (simulated)."""
    print("=" * 60)
    print("Example: API Assistant")
    print("=" * 60)
    print()

    # Define API-like tools
    tools = [
        ToolDefinition(
            name="get_weather",
            description="Get current weather for a location",
            input_schema={
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name or location",
                    },
                    "units": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature units",
                    },
                },
                "required": ["location"],
            },
        ),
        ToolDefinition(
            name="search_docs",
            description="Search documentation for a query",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "max_results": {"type": "integer", "description": "Max results to return"},
                },
                "required": ["query"],
            },
        ),
    ]

    def execute_tool(name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Simulate API calls."""
        if name == "get_weather":
            # Simulate weather API
            location = args["location"]
            units = args.get("units", "celsius")
            print(f"  [Tool] Fetching weather for {location} ({units})")
            return {
                "location": location,
                "temperature": 22 if units == "celsius" else 72,
                "condition": "sunny",
                "units": units,
            }

        if name == "search_docs":
            # Simulate documentation search
            query = args["query"]
            max_results = args.get("max_results", 5)
            print(f"  [Tool] Searching docs for '{query}' (max {max_results})")
            return {
                "results": [
                    {"title": f"Doc about {query}", "url": "https://example.com/doc1"},
                    {"title": f"{query} tutorial", "url": "https://example.com/doc2"},
                ],
                "total": 2,
            }

        raise ValueError(f"Unknown tool: {name}")

    # Create assistant with API access
    assistant = Participant(
        name="APIAssistant",
        adapter=OpenAIAdapter(),
        model="gpt-4o-mini",
        system_prompts=[
            "You are a helpful assistant with access to weather and documentation APIs.",
            "Use the tools to provide accurate, real-time information.",
        ],
        temperature=0.0,
    )

    runner = TurnRunner(participant=assistant, tools=tools, tool_executor=execute_tool)

    print("Setup complete:")
    print(f"  - Assistant: {assistant.name}")
    print(f"  - Available tools: {[t.name for t in tools]}")
    print(f"  - Example query: 'What's the weather in Paris?'")
    print()
    print("=" * 60)
    print()


def main():
    """Run all assistant examples."""
    print("\n" + "=" * 60)
    print("Assistant Examples - Tool Usage Patterns")
    print("=" * 60)
    print()

    example_calculator_assistant()
    example_api_assistant()

    print("=" * 60)
    print("Examples complete!")
    print()
    print("To run with real API calls:")
    print("1. Set OPENAI_API_KEY environment variable")
    print("2. Uncomment the runner.run_turn() calls")
    print("3. Run: python -m multi_agent_harness.examples.assistant")
    print("=" * 60)


if __name__ == "__main__":
    main()
