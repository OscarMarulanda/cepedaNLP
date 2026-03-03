# ADR 009: API Key Authentication + Rate Limiting for MCP SSE Endpoint

**Date:** 2026-03-03
**Status:** Accepted

## Context
The MCP server on Render (`https://cepeda-nlp-mcp.onrender.com/sse`) is publicly accessible. Without protection, anyone can:
- Spam `submit_opinion` to fill the Supabase free tier (500 MB)
- Hammer the endpoint with requests
- Abuse the HuggingFace Inference API via `retrieve_chunks`

The Streamlit frontend is unaffected (it calls MCP tools in-process, not via HTTP).

## Decision
Add three Starlette middleware layers to the SSE endpoint:

1. **API key authentication** — `Authorization: Bearer <key>` header, constant-time comparison, controlled by `MCP_API_KEY` env var
2. **Per-IP rate limiting** — sliding window (default 30 req/min), proxy-aware IP extraction, in-memory storage
3. **SSE connection limiting** — per-IP concurrent connection cap (default 5), prevents resource exhaustion from long-lived SSE connections

All skip `/health` for Render health checks. Auth is disabled when `MCP_API_KEY` is unset (dev convenience).

## Implementation
- `src/mcp/middleware.py` — `APIKeyMiddleware` + `RateLimitMiddleware` + `SSEConnectionMiddleware` (Starlette `BaseHTTPMiddleware`)
- `run_mcp.py` — wires middleware via `mcp.http_app(middleware=[Middleware(...)])`
- `render.yaml` — `MCP_API_KEY` (secret) + `MCP_RATE_LIMIT` (default "30") + `DB_SSLMODE=verify-full`

## Alternatives Considered

| Option | Why rejected |
|--------|-------------|
| Query parameter (`?key=...`) | Leaks in logs and browser history |
| Redis rate limiting | Overkill — single Render instance, in-memory is fine |
| No auth (keep ADR 005) | `submit_opinion` is a write op — abuse risk too high for public endpoint |
| OAuth / JWT | Too complex for a demo MCP server |

## Rationale
- Bearer header is the standard for API key auth — well-supported by MCP clients
- Constant-time key comparison resists side-channel analysis
- Proxy-aware IP extraction ensures rate limiting works correctly behind reverse proxies
- SSE connection limiting prevents resource exhaustion on single-instance free tier
- In-memory rate limiting is appropriate for a single-instance deployment
- No new dependencies (Starlette is already a FastMCP dependency)
- Optional auth preserves zero-friction local development

## Consequences
- External MCP clients must include `Authorization: Bearer <key>` header
- Rate limiting state is lost on Render restart (acceptable — sliding window resets cleanly)
- SSE connection state is lost on restart (acceptable — clients reconnect)
- Partially supersedes ADR 005 for the MCP endpoint (Streamlit frontend still has no auth)
