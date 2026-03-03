"""Security middleware for the MCP SSE server.

Three middleware classes:
- APIKeyMiddleware: Bearer token authentication (optional, controlled by MCP_API_KEY)
- RateLimitMiddleware: Per-IP sliding window rate limiter (in-memory)
- SSEConnectionMiddleware: Per-IP concurrent SSE connection limiter

All skip /health so Render health checks work without auth.
"""

import hmac
import logging
import os
import time
from collections import defaultdict

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

logger = logging.getLogger(__name__)

SKIP_PATHS = {"/health"}
SSE_PATHS = {"/sse", "/messages"}

# Cleanup stale entries every 5 minutes
_CLEANUP_INTERVAL = 300


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Validate Authorization: Bearer <key> header on all requests except /health.

    Only active when MCP_API_KEY environment variable is set. If unset, all
    requests pass through (convenient for local development).
    """

    def __init__(self, app, api_key: str | None = None) -> None:  # noqa: D107
        super().__init__(app)
        self.api_key = api_key or os.getenv("MCP_API_KEY")

    async def dispatch(self, request: Request, call_next) -> Response:
        if request.url.path in SKIP_PATHS:
            return await call_next(request)

        if not self.api_key:
            return await call_next(request)

        auth_header = request.headers.get("authorization", "")
        if not auth_header.startswith("Bearer "):
            return JSONResponse(
                {"error": "Missing or invalid Authorization header"},
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
            )

        token = auth_header[7:]  # len("Bearer ") == 7
        if not hmac.compare_digest(token, self.api_key):
            return JSONResponse(
                {"error": "Invalid API key"},
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
            )

        return await call_next(request)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Per-IP sliding window rate limiter using in-memory storage.

    Extracts real client IP from X-Forwarded-For (Render's proxy sets this).
    Configurable via MCP_RATE_LIMIT env var (default: 30 requests/minute).
    Returns 429 with Retry-After header when limit is exceeded.
    """

    def __init__(self, app, max_requests: int | None = None) -> None:  # noqa: D107
        super().__init__(app)
        self.max_requests = max_requests or int(
            os.getenv("MCP_RATE_LIMIT", "30")
        )
        self.window = 60  # 1-minute sliding window
        # {ip: [timestamp, timestamp, ...]}
        self._requests: dict[str, list[float]] = defaultdict(list)
        self._last_cleanup = time.monotonic()

    def _get_client_ip(self, request: Request) -> str:
        """Extract real client IP from X-Forwarded-For.

        Uses the rightmost value — the one appended by the closest trusted
        proxy (Render). The leftmost value is client-supplied and spoofable.
        """
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[-1].strip()
        client = request.client
        return client.host if client else "unknown"

    def _cleanup_stale(self, now: float) -> None:
        """Remove entries for IPs with no recent requests."""
        if now - self._last_cleanup < _CLEANUP_INTERVAL:
            return
        self._last_cleanup = now
        cutoff = now - self.window
        stale_ips = [
            ip for ip, timestamps in self._requests.items()
            if not timestamps or timestamps[-1] < cutoff
        ]
        for ip in stale_ips:
            del self._requests[ip]
        if stale_ips:
            logger.debug("Rate limiter cleanup: removed %d stale IPs", len(stale_ips))

    async def dispatch(self, request: Request, call_next) -> Response:
        if request.url.path in SKIP_PATHS:
            return await call_next(request)

        now = time.monotonic()
        self._cleanup_stale(now)

        ip = self._get_client_ip(request)
        cutoff = now - self.window

        # Remove timestamps outside the window
        timestamps = self._requests[ip]
        self._requests[ip] = [t for t in timestamps if t > cutoff]
        timestamps = self._requests[ip]

        if len(timestamps) >= self.max_requests:
            # Time until the oldest request in window expires
            retry_after = int(timestamps[0] - cutoff) + 1
            return JSONResponse(
                {"error": "Rate limit exceeded", "retry_after": retry_after},
                status_code=429,
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(self.max_requests),
                    "X-RateLimit-Remaining": "0",
                },
            )

        timestamps.append(now)
        remaining = self.max_requests - len(timestamps)

        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self.max_requests)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        return response


class SSEConnectionMiddleware(BaseHTTPMiddleware):
    """Per-IP concurrent SSE connection limiter.

    SSE connections are long-lived. Without a limit, an attacker can exhaust
    server resources by opening hundreds of persistent connections.
    Configurable via MCP_MAX_SSE_CONNS env var (default: 5 per IP).
    """

    def __init__(self, app, max_conns: int | None = None) -> None:  # noqa: D107
        super().__init__(app)
        self.max_conns = max_conns or int(
            os.getenv("MCP_MAX_SSE_CONNS", "5")
        )
        # {ip: active_connection_count}
        self._active: dict[str, int] = defaultdict(int)

    def _get_client_ip(self, request: Request) -> str:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[-1].strip()
        client = request.client
        return client.host if client else "unknown"

    async def dispatch(self, request: Request, call_next) -> Response:
        if request.url.path not in SSE_PATHS:
            return await call_next(request)

        ip = self._get_client_ip(request)

        if self._active[ip] >= self.max_conns:
            logger.warning(
                "SSE connection limit reached for %s (%d/%d)",
                ip, self._active[ip], self.max_conns,
            )
            return JSONResponse(
                {"error": "Too many concurrent connections"},
                status_code=429,
            )

        self._active[ip] += 1
        try:
            response = await call_next(request)
            return response
        finally:
            self._active[ip] -= 1
            if self._active[ip] <= 0:
                del self._active[ip]
