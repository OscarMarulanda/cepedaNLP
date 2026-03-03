# ADR 005: No Authentication, Rate Limiting Only

**Date:** 2026-02-28
**Status:** Partially superseded by ADR 009 (MCP SSE endpoint now has API key auth)

## Context
The API needs to be accessible for demo purposes (MVP, public showcase). Authentication creates friction for users trying to interact with the chatbot.

## Decision
No authentication. Protect with rate limiting (slowapi), input validation, prompt injection filtering, and CORS instead.

## Security Stack
1. Rate limiting — 30 messages per session (MCP-only architecture, no REST endpoints)
2. Input validation — Pydantic max_length=500, type constraints
3. Prompt injection filter — 14+ regex patterns, rejects with 400
4. SQL injection — parameterized queries (psycopg2 `%s`)
5. Output sanitization — strip XSS vectors from LLM output
6. Transport encryption — `DB_SSLMODE=require` for remote DB connections (see ADR 008)

## Rationale
- Demo/MVP app — friction-free access is more important than gatekeeping.
- Rate limiting caps worst-case Claude API abuse at ~$67/day per IP.
- All queries are read-only against the corpus — no destructive operations possible.

## Consequences
- Vulnerable to determined abuse (IP rotation bypasses rate limits).
- Acceptable risk for a demo. Production would add API keys or OAuth.
- DB connections over the internet are encrypted via `sslmode=verify-full` (see `docs/DB_CONNECTION_SECURITY.md`).

## Update (2026-03-03)
The **Streamlit frontend** still follows this ADR (no auth, session-based rate limiting). However, the **public MCP SSE endpoint** on Render now requires API key authentication (`Authorization: Bearer <key>`) and has per-IP rate limiting. See ADR 009.
