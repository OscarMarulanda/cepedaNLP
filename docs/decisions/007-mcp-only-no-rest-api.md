# ADR 007: MCP-Only Architecture, No REST API

**Date:** 2026-02-28
**Status:** Accepted
**Supersedes:** Previous plan with FastAPI REST + MCP hybrid

## Context
Need to expose the RAG system to a frontend. Two approaches considered:
1. Traditional: FastAPI REST API + MCP server alongside
2. MCP-only: Claude on the frontend orchestrates MCP tools directly

## Decision
MCP-only. No REST API. Claude (Haiku) on the Streamlit server-side acts as the orchestrator, calling MCP tools based on user input.

## Rationale
- **Less code** — no routes, no Pydantic request/response models, no API layer to maintain.
- **Smarter routing** — Claude decides whether to retrieve chunks, list speeches, or search entities based on natural language. No need to build separate UI pages/filters.
- **Extensible** — adding new data sources later = plugging in another MCP server. No new UI, no new routes.
- **MCP proficiency** — demonstrates the protocol for the MVP demo. The same MCP server works with Claude Desktop, Cursor, or any MCP client.
- **Streamlit is the backend** — Python runs server-side, so API key is safe and MCP calls happen locally. The browser only sees rendered HTML.

## Tradeoffs Accepted
- Every user message costs a Claude API call (~$0.005 on Haiku), even simple reads like "list speeches." Acceptable at demo scale.
- Latency is 1-3s per message (Claude roundtrip) vs <100ms for direct DB reads. Acceptable for a chat interface.
- Less control — Claude might occasionally call the wrong tool. Mitigated by clear tool descriptions and constrained system prompt.

## Consequences
- `src/rag/generator.py` and `src/rag/query.py` are not used by the frontend — Claude generation happens on the Streamlit side. These modules remain available for CLI usage (`python -m src.rag.query`).
- MCP tools are pure data fetchers — no LLM calls inside tools.
- Security shifts from REST middleware (rate limiting, CORS) to session-based controls in Streamlit + input validation in MCP tools.
