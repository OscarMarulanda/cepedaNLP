"""Programmatic entry point for the CepedaNLP MCP server (SSE transport).

Avoids `fastmcp run` CLI edge cases and gives control over host/port
binding. Adds a /health route for Render's health checks.

Usage:
    PORT=8000 python run_mcp.py
"""

import logging
import os

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from src.mcp.server import mcp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


async def health(request: Request) -> JSONResponse:
    """Health check endpoint for Render."""
    return JSONResponse({"status": "ok"})


app = mcp.http_app(transport="sse")
app.routes.append(Route("/health", health))

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    logger.info("Starting CepedaNLP MCP server on port %d (SSE)", port)
    uvicorn.run(app, host="0.0.0.0", port=port)
