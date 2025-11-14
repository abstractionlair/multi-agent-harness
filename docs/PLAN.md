# Plan: Multi‑Agent LLM Harness Library

Status: Draft (initial extraction)  
Last Updated: 2025-11-13

## Objectives

- Provide a reusable Python library for:
  - Multi‑role LLM conversations (assistant, user‑proxy, judge, interrogator, etc.).
  - Provider‑agnostic tool/function calling with JSON Schema definitions.
  - Structured transcripts and model‑as‑judge evaluation.
- Keep the core **domain‑agnostic**:
  - No GTD ontology or graph code.
  - No MCP specifics; treat tools as opaque callables.
- Allow host projects to:
  - Plug in domain prompts and tool backends (e.g., MCP, HTTP services).
  - Define their own judgment prompts and verdict schemas.

## Core Concepts

- **ChatMessage** — normalized chat message (`role`, `content`), plus support for tool call metadata.
- **ToolDefinition** — name, description, JSON Schema `parameters`.
- **ToolCall** — model‑requested tool invocation (name, arguments, optional id).
- **ToolInvocationRecord** — executed tool call with arguments + result or error.
- **ConversationTurn** / **ConversationTranscript** — ordered sequence of turns with attached tool invocations.
- **ProviderAdapter** — adapter per provider (OpenAI, Anthropic, xAI) with a common `send_chat` API.
- **RoleEngine** — base class plus concrete engines for assistant, user‑proxy, judge, interrogator.

## Package Layout (Initial)

```text
multi_agent_harness/
  __init__.py
  config.py          # RoleModelConfig
  adapters/
    __init__.py
    base.py          # core types + ProviderAdapter
    openai.py        # OpenAI implementation
  engines/
    __init__.py
    base.py          # RoleEngine + transcript models
    assistant.py
    user_proxy.py
    judge.py
    interrogator.py
  tools.py           # Minimal ToolRouter abstraction (optional)
```

The library deliberately does **not** contain:

- MCP clients or graph‑memory specifics.
- Domain prompts or test data.
- Project‑specific config schemas beyond basic role→model configs.

## Extraction Steps (This Repo → Library)

1. Copy generic types and adapters from the GTD repo:
   - `tests/harness_api/adapters/base.py` → `adapters/base.py`
   - `tests/harness_api/adapters/openai_adapter.py` → `adapters/openai.py`
   - Remove imports of GTD‑specific config; introduce a minimal `RoleModelConfig` in `config.py`.

2. Copy role engines with light generalization:
   - `tests/harness_api/engines/base.py` → `engines/base.py`
   - `tests/harness_api/engines/assistant.py` → `engines/assistant.py`
   - `tests/harness_api/engines/user_proxy.py` → `engines/user_proxy.py`
   - `tests/harness_api/engines/judge.py` → `engines/judge.py`
   - `tests/harness_api/engines/interrogator.py` → `engines/interrogator.py`
   - Replace references to project‑local `HarnessConfig` with direct `RoleModelConfig` where possible.

3. Add minimal `ToolRouter` abstraction:
   - Very small interface: register tools + executors, and execute by name.
   - Host projects are responsible for wiring this to their own backends (MCP, HTTP, DB, etc.).

4. Add a simple runner helper:

```python
from multi_agent_harness.adapters.openai import OpenAIAdapter
from multi_agent_harness.config import RoleModelConfig
from multi_agent_harness.engines.assistant import AssistantEngine

assistant_cfg = RoleModelConfig(provider="openai", model="gpt-4o-mini")
adapter = OpenAIAdapter()
assistant = AssistantEngine(role_config=assistant_cfg, adapter=adapter, tool_router=my_router)

response = assistant.run_turn(history=[], user_text="Hello", max_steps=3)
```

5. Add basic tests:
   - Unit tests for `OpenAIAdapter` payload shaping (no live network).
   - A small “toy tool” example:
     - Register a simple `add(a, b)` tool.
     - Give the assistant a prompt that explains when to call `add`.
     - Assert that a test prompt produces at least one tool call.

## Integration Back into GTD Repo

When ready, the GTD project can:

- Add a dependency on this library (local path or published package).
- Replace imports from `tests/harness_api/adapters` and `tests/harness_api/engines` with `multi_agent_harness.*`.
- Keep GTD‑specific wiring (prompts, MCP bridge, ontology setup) in its own code.

This document is the high‑level guide; detailed public API docs will evolve once the first version is stabilized.

