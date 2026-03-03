"""Programmatic entry point for the CepedaNLP MCP server (SSE transport).

Avoids `fastmcp run` CLI edge cases and gives control over host/port
binding. Adds a /health route for Render's health checks.

Security middleware (applied in order):
1. APIKeyMiddleware — Bearer token auth (only if MCP_API_KEY is set)
2. RateLimitMiddleware — per-IP sliding window (default 30 req/min)
3. SSEConnectionMiddleware — per-IP concurrent SSE connection limit (default 5)

Usage:
    PORT=8000 python run_mcp.py                          # no auth (dev)
    MCP_API_KEY=secret PORT=8000 python run_mcp.py       # auth enabled
"""

import logging
import os

from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from src.mcp.middleware import (
    APIKeyMiddleware,
    RateLimitMiddleware,
    SSEConnectionMiddleware,
)
from src.mcp.server import mcp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


async def health(request: Request) -> JSONResponse:
    """Health check endpoint for Render."""
    return JSONResponse({"status": "ok"})


# Build middleware stack
middleware: list[Middleware] = []

if os.getenv("MCP_API_KEY"):
    middleware.append(Middleware(APIKeyMiddleware))
    logger.info("API key authentication enabled")

middleware.append(Middleware(RateLimitMiddleware))
middleware.append(Middleware(SSEConnectionMiddleware))

app = mcp.http_app(transport="sse", middleware=middleware)
app.routes.append(Route("/health", health))

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    logger.info("Starting CepedaNLP MCP server on port %d (SSE)", port)
    uvicorn.run(app, host="0.0.0.0", port=port)
