# Multi‑Agent LLM Harness (Draft)

Status: Experimental (extracted from GTD test harness)  
Goal: A reusable Python library for wiring multiple LLM roles into structured conversations with tools, plus model‑as‑judge evaluation.

## What This Is

A small, provider‑agnostic harness that lets you:

- Define multiple roles (e.g., assistant, user‑proxy, judge, interrogator).
- Plug in provider adapters (OpenAI, Anthropic, xAI, stub).
- Expose tools/functions to roles via JSON‑schema definitions.
- Run scripted scenarios and collect:
  - A normalized transcript (messages + tool invocations).
  - Optional judge verdicts (model‑as‑judge).

This library is being factored out of the GTD conversational layer test harness so it can be reused in other projects.

## High‑Level Design

- **Core types**: `ChatMessage`, `ToolDefinition`, `ToolCall`, `ConversationTranscript`, `ToolInvocationRecord`.
- **Provider adapters**: `OpenAIAdapter` plus stubs for others (Anthropic, xAI).
- **Role engines**:
  - `AssistantEngine` — runs the SUT with tool loops.
  - `UserProxyEngine` — simulates user turns (scripted or LLM‑driven).
  - `JudgeEngine` — runs a model that scores transcripts.
  - `InterrogatorEngine` — asks the assistant follow‑up questions after the main interaction.
- **Tool execution**: via a pluggable router interface; the library does not know about MCP or any specific backend.

## Status

This is an initial extraction:

- APIs are not stable yet.
- Only OpenAI adapter is wired seriously; others may be placeholders.
- Docs and examples are minimal.

Host projects (like the GTD repo) remain the source of truth for domain‑specific prompts, tools, and judging criteria.

## Next Steps

See `docs/PLAN.md` for:

- Extraction roadmap from the original GTD harness.
- Planned public API surface.
- Testing and example scenarios.

