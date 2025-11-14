# Multi‑Agent Harness — Agent Guide

This repo is for a **generic multi‑agent LLM testing harness**: wiring multiple model roles (assistant, user‑proxy, judge, interrogator, etc.) together with tools, and producing structured transcripts/verdicts.

For Codex:
- Start with `README.md`, then `docs/PLAN.md`.
- Keep this library **provider‑agnostic** (no project‑specific GTD logic).
- Focus on:
  - Clean abstractions for roles, providers, tools, and transcripts.
  - Minimal dependencies and clear configuration.
  - Being easy to embed into other projects (including the GTD repo).

Do not add GTD‑specific ontology, graph, or MCP details here; those live in host projects.

