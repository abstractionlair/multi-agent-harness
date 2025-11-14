# Tests

This directory contains the test suite for the multi-agent-harness library.

## Test Structure

```
tests/
├── unit/                       # Unit tests (fast, no network)
│   ├── test_openai_adapter.py # OpenAI adapter payload conversion
│   ├── test_turn_runner.py    # TurnRunner logic
│   ├── test_conversation_runner.py  # ConversationRunner state
│   └── test_tool_loop.py      # Tool loop behavior
├── integration/               # Integration tests (recorded responses)
│   └── test_recorded_responses.py
└── conftest.py                # Pytest configuration
```

## Running Tests

### Run all tests
```bash
pytest
```

### Run only unit tests (fast)
```bash
pytest tests/unit
```

### Run only integration tests
```bash
pytest tests/integration
```

### Run with coverage
```bash
pytest --cov=src/multi_agent_harness --cov-report=html
```

### Run specific test file
```bash
pytest tests/unit/test_turn_runner.py -v
```

### Run specific test
```bash
pytest tests/unit/test_turn_runner.py::TestTurnRunnerExecution::test_simple_turn_without_tools -v
```

## Test Categories

### Unit Tests
- **No network calls**: All tests use mocks/stubs
- **Fast execution**: Should complete in < 1 second each
- **Isolated**: Test single components in isolation
- **Coverage**: Focus on edge cases and error handling

### Integration Tests
- **Recorded responses**: Use pre-recorded API responses
- **End-to-end**: Test complete workflows
- **Realistic scenarios**: Match real-world usage patterns

## Development Workflow

1. **Write tests first** (TDD approach recommended)
2. **Run tests locally**: `pytest`
3. **Check coverage**: `pytest --cov`
4. **Pre-commit hooks**: Automatically run on commit
5. **CI checks**: Automatically run on push/PR

## Coverage Goals

- **Target**: 80%+ overall coverage
- **Critical paths**: 100% coverage for core logic
- **Edge cases**: Explicit tests for error conditions

## Adding New Tests

1. Choose appropriate directory (unit vs integration)
2. Follow naming convention: `test_*.py`
3. Use descriptive test class/function names
4. Add docstrings explaining test purpose
5. Use fixtures from conftest.py when applicable
6. Mark tests appropriately: `@pytest.mark.unit` or `@pytest.mark.integration`

## Continuous Integration

Tests run automatically on:
- Every push to main/master
- Every pull request
- Multiple Python versions (3.10, 3.11, 3.12)

See `.github/workflows/ci.yml` for CI configuration.
