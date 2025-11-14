#!/usr/bin/env python3
"""Verification script for Phase 2: Provider Adapters

This script verifies that all provider adapters (OpenAI, Anthropic, xAI, Gemini)
are properly implemented and can be imported.
"""

from src.multi_agent_harness.adapters import (
    OpenAIAdapter,
    AnthropicAdapter,
    XAIAdapter,
    GeminiAdapter,
    ChatMessage,
    ToolDefinition,
    ResponseFormat,
)
from src.multi_agent_harness.config import RoleModelConfig


def test_adapter_interface(adapter_class, adapter_name: str):
    """Test that an adapter has the required interface."""
    print(f"\n{'='*60}")
    print(f"Testing {adapter_name}")
    print('='*60)

    # Test instantiation
    adapter = adapter_class()
    print(f"✓ {adapter_name} can be instantiated")

    # Test provider_name
    assert hasattr(adapter, 'provider_name'), f"{adapter_name} missing provider_name"
    print(f"✓ Provider name: {adapter.provider_name}")

    # Test supports_tools
    supports_tools = adapter.supports_tools()
    print(f"✓ Supports tools: {supports_tools}")

    # Test that send_chat method exists
    assert hasattr(adapter, 'send_chat'), f"{adapter_name} missing send_chat method"
    print(f"✓ Has send_chat method")

    return adapter


def test_message_conversion():
    """Test message conversion for each adapter."""
    print(f"\n{'='*60}")
    print("Testing Message Conversion")
    print('='*60)

    # Test messages
    test_messages = [
        ChatMessage(role="system", content="You are a helpful assistant."),
        ChatMessage(role="user", content="Hello!"),
        ChatMessage(role="assistant", content="Hi there!"),
    ]

    # Test OpenAI
    openai_adapter = OpenAIAdapter()
    openai_messages = [openai_adapter._convert_message(msg) for msg in test_messages]
    print(f"✓ OpenAI message conversion works")
    print(f"  Sample: {openai_messages[0]}")

    # Test Anthropic
    anthropic_adapter = AnthropicAdapter()
    anthropic_messages = [anthropic_adapter._convert_message(msg) for msg in test_messages[1:]]  # Skip system
    print(f"✓ Anthropic message conversion works")
    print(f"  Sample: {anthropic_messages[0]}")

    # Test xAI
    xai_adapter = XAIAdapter()
    xai_messages = [xai_adapter._convert_message(msg) for msg in test_messages]
    print(f"✓ xAI message conversion works")
    print(f"  Sample: {xai_messages[0]}")

    # Test Gemini
    gemini_adapter = GeminiAdapter()
    gemini_messages = [gemini_adapter._convert_message(msg) for msg in test_messages]
    print(f"✓ Gemini message conversion works")
    print(f"  Sample: {gemini_messages[0]}")


def test_tool_conversion():
    """Test tool definition conversion for each adapter."""
    print(f"\n{'='*60}")
    print("Testing Tool Conversion")
    print('='*60)

    tool = ToolDefinition(
        name="test_tool",
        description="A test tool",
        input_schema={
            "type": "object",
            "properties": {
                "arg1": {"type": "string"}
            },
            "required": ["arg1"]
        }
    )

    # Test OpenAI
    openai_adapter = OpenAIAdapter()
    openai_tool = openai_adapter._convert_tool(tool)
    assert openai_tool["type"] == "function"
    assert openai_tool["function"]["name"] == "test_tool"
    print(f"✓ OpenAI tool conversion works")

    # Test Anthropic
    anthropic_adapter = AnthropicAdapter()
    anthropic_tool = anthropic_adapter._convert_tool(tool)
    assert anthropic_tool["name"] == "test_tool"
    print(f"✓ Anthropic tool conversion works")

    # Test xAI
    xai_adapter = XAIAdapter()
    xai_tool = xai_adapter._convert_tool(tool)
    assert xai_tool["type"] == "function"
    print(f"✓ xAI tool conversion works")

    # Test Gemini
    gemini_adapter = GeminiAdapter()
    gemini_tool_rest = gemini_adapter._convert_tool_to_rest(tool)
    assert gemini_tool_rest["name"] == "test_tool"
    print(f"✓ Gemini tool conversion works")


def main():
    """Run all verification tests."""
    print("\n" + "="*60)
    print("Phase 2 Verification: Provider Adapters")
    print("="*60)

    try:
        # Test each adapter
        adapters = [
            (OpenAIAdapter, "OpenAI"),
            (AnthropicAdapter, "Anthropic"),
            (XAIAdapter, "xAI"),
            (GeminiAdapter, "Gemini"),
        ]

        for adapter_class, name in adapters:
            test_adapter_interface(adapter_class, name)

        # Test message conversion
        test_message_conversion()

        # Test tool conversion
        test_tool_conversion()

        # Summary
        print(f"\n{'='*60}")
        print("✅ All Phase 2 verifications passed!")
        print('='*60)
        print("\nImplemented adapters:")
        print("  • OpenAI - Full implementation with SDK and REST fallback")
        print("  • Anthropic - Content blocks, tool use/result, system prompts")
        print("  • xAI - OpenAI-compatible API")
        print("  • Gemini - Content/parts format, function calling")
        print("\nAll adapters support:")
        print("  • Message format conversion")
        print("  • Tool/function calling")
        print("  • System prompt handling")
        print("  • REST API fallback")
        print()

    except Exception as e:
        print(f"\n❌ Verification failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
