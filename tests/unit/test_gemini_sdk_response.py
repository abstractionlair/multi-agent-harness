"""Regression tests for Gemini SDK response conversion.

These tests cover the SDK response parsing path that previously had zero
coverage, which allowed a bug to slip through: Part is a pydantic model
where .text always exists (defaults to None), so hasattr(part, 'text')
is always True and the function_call branch was dead code.
"""

from __future__ import annotations

from multi_agent_harness.adapters.gemini import GeminiAdapter


class _FakeFunctionCall:
    """Minimal stand-in for google.genai.types.FunctionCall."""

    def __init__(self, name: str, args: dict) -> None:
        self.name = name
        self.args = args
        self.id = None


class _FakePart:
    """Minimal stand-in for google.genai.types.Part.

    Mirrors the real pydantic model: both .text and .function_call always
    exist as attributes, defaulting to None.
    """

    def __init__(self, *, text: str | None = None, function_call: _FakeFunctionCall | None = None) -> None:
        self.text = text
        self.function_call = function_call


class _FakeContent:
    """Minimal stand-in for google.genai.types.Content."""

    def __init__(self, parts: list[_FakePart]) -> None:
        self.parts = parts


class _FakeCandidate:
    """Minimal stand-in for a candidate in the SDK response."""

    def __init__(self, content: _FakeContent) -> None:
        self.content = content


class _FakeResponse:
    """Minimal stand-in for the full SDK GenerateContentResponse."""

    def __init__(self, candidates: list[_FakeCandidate]) -> None:
        self.candidates = candidates


class TestGeminiSdkResponseTextOnly:
    """Test text-only SDK response parsing."""

    def test_single_text_part(self) -> None:
        response = _FakeResponse([
            _FakeCandidate(_FakeContent([_FakePart(text="Hello, world!")])),
        ])
        result = GeminiAdapter._convert_sdk_response(response)
        assert result.message.content == "Hello, world!"
        assert len(result.tool_calls) == 0

    def test_multiple_text_parts_joined(self) -> None:
        response = _FakeResponse([
            _FakeCandidate(_FakeContent([
                _FakePart(text="Hello "),
                _FakePart(text="world!"),
            ])),
        ])
        result = GeminiAdapter._convert_sdk_response(response)
        assert result.message.content == "Hello world!"

    def test_multiple_candidates_text_concatenated(self) -> None:
        response = _FakeResponse([
            _FakeCandidate(_FakeContent([_FakePart(text="First ")])),
            _FakeCandidate(_FakeContent([_FakePart(text="Second")])),
        ])
        result = GeminiAdapter._convert_sdk_response(response)
        assert result.message.content == "First Second"


class TestGeminiSdkResponseFunctionCall:
    """Test function-call SDK response parsing."""

    def test_single_function_call(self) -> None:
        fc = _FakeFunctionCall(name="get_weather", args={"location": "Boston"})
        response = _FakeResponse([
            _FakeCandidate(_FakeContent([_FakePart(function_call=fc)])),
        ])
        result = GeminiAdapter._convert_sdk_response(response)
        assert result.message.content == ""
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "get_weather"
        assert result.tool_calls[0].arguments == {"location": "Boston"}

    def test_function_call_with_empty_args(self) -> None:
        fc = _FakeFunctionCall(name="ping", args={})
        response = _FakeResponse([
            _FakeCandidate(_FakeContent([_FakePart(function_call=fc)])),
        ])
        result = GeminiAdapter._convert_sdk_response(response)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].arguments == {}

    def test_multiple_function_calls(self) -> None:
        fc1 = _FakeFunctionCall(name="tool_a", args={"x": 1})
        fc2 = _FakeFunctionCall(name="tool_b", args={"y": 2})
        response = _FakeResponse([
            _FakeCandidate(_FakeContent([_FakePart(function_call=fc1), _FakePart(function_call=fc2)])),
        ])
        result = GeminiAdapter._convert_sdk_response(response)
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].name == "tool_a"
        assert result.tool_calls[1].name == "tool_b"


class TestGeminiSdkResponseMixed:
    """Test mixed text and function-call parts."""

    def test_text_and_function_call_in_same_response(self) -> None:
        """When a response has both text and function_call parts, both are captured."""
        fc = _FakeFunctionCall(name="search", args={"q": "test"})
        response = _FakeResponse([
            _FakeCandidate(_FakeContent([
                _FakePart(text="Let me search for that."),
                _FakePart(function_call=fc),
            ])),
        ])
        result = GeminiAdapter._convert_sdk_response(response)
        assert result.message.content == "Let me search for that."
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "search"

    def test_function_call_with_none_text(self) -> None:
        """A function_call Part has text=None; must not append None to text_parts."""
        fc = _FakeFunctionCall(name="calc", args={"a": 5})
        response = _FakeResponse([
            _FakeCandidate(_FakeContent([_FakePart(function_call=fc)])),
        ])
        # This would raise TypeError if the bug were present:
        # ''.join([None]) -> TypeError
        result = GeminiAdapter._convert_sdk_response(response)
        assert result.message.content == ""
        assert len(result.tool_calls) == 1


class TestGeminiSdkResponseEdgeCases:
    """Edge cases for SDK response parsing."""

    def test_empty_parts(self) -> None:
        response = _FakeResponse([
            _FakeCandidate(_FakeContent([])),
        ])
        result = GeminiAdapter._convert_sdk_response(response)
        assert result.message.content == ""
        assert len(result.tool_calls) == 0

    def test_part_with_both_none(self) -> None:
        """Part with neither text nor function_call set."""
        response = _FakeResponse([
            _FakeCandidate(_FakeContent([_FakePart()])),
        ])
        result = GeminiAdapter._convert_sdk_response(response)
        assert result.message.content == ""
        assert len(result.tool_calls) == 0
