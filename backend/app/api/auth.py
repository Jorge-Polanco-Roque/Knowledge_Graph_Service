"""API key authentication and OpenAI key management for NeuroGraph.

Two-layer auth:
1. Service API Key (NEUROGRAPH_API_KEY): authenticates the client application.
2. Per-request OpenAI Key (X-OpenAI-Key header): optional override so each
   end-user can pay for their own LLM usage.

When NEUROGRAPH_API_KEY is not set, auth is disabled (local dev mode).
"""

from __future__ import annotations

import hmac
import logging
import os

from fastapi import HTTPException, Request, WebSocket
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from .context import current_openai_key

logger = logging.getLogger(__name__)

# Endpoints that bypass auth (health checks, docs)
_PUBLIC_PATHS = frozenset({"/health", "/docs", "/openapi.json", "/redoc"})


# ---------------------------------------------------------------------------
# Service API key helpers
# ---------------------------------------------------------------------------

def _get_valid_api_keys() -> set[str]:
    """Load valid API keys from environment.

    Supports multiple keys via comma separation for key rotation:
    NEUROGRAPH_API_KEY=nk_key1,nk_key2
    """
    raw = os.environ.get("NEUROGRAPH_API_KEY", "")
    if not raw:
        return set()
    return {k.strip() for k in raw.split(",") if k.strip()}


def _extract_token(request: Request) -> str | None:
    """Extract bearer token from Authorization header or X-API-Key."""
    auth = request.headers.get("authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:].strip()
    return request.headers.get("x-api-key")


def _extract_ws_token(websocket: WebSocket) -> str | None:
    """Extract token from WebSocket query param or headers."""
    token = websocket.query_params.get("token")
    if token:
        return token
    auth = websocket.headers.get("authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:].strip()
    return websocket.headers.get("x-api-key")


# ---------------------------------------------------------------------------
# Middleware: API key auth
# ---------------------------------------------------------------------------

class APIKeyAuthMiddleware(BaseHTTPMiddleware):
    """Validates service API key on every request except public paths."""

    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        # Skip auth for public paths
        if request.url.path in _PUBLIC_PATHS:
            return await call_next(request)

        # Skip auth for WebSocket upgrades (handled in the WS endpoint)
        if request.headers.get("upgrade", "").lower() == "websocket":
            return await call_next(request)

        valid_keys = _get_valid_api_keys()

        # If no keys configured, auth is disabled (local dev mode)
        if not valid_keys:
            return await call_next(request)

        token = _extract_token(request)
        if not token:
            return JSONResponse(
                status_code=401,
                content={
                    "detail": "Missing API key. Provide via "
                    "'Authorization: Bearer <key>' or 'X-API-Key: <key>' header.",
                },
            )

        # Constant-time comparison to prevent timing attacks
        if not any(hmac.compare_digest(token, k) for k in valid_keys):
            logger.warning(
                "Invalid API key attempt from %s",
                request.client.host if request.client else "unknown",
            )
            return JSONResponse(status_code=403, content={"detail": "Invalid API key."})

        return await call_next(request)


def validate_ws_token(websocket: WebSocket) -> bool:
    """Validate WebSocket connection token. Returns True if valid or auth disabled."""
    valid_keys = _get_valid_api_keys()
    if not valid_keys:
        return True  # Auth disabled (local dev)

    token = _extract_ws_token(websocket)
    if not token:
        return False

    return any(hmac.compare_digest(token, k) for k in valid_keys)


# ---------------------------------------------------------------------------
# Middleware: Per-request OpenAI key
# ---------------------------------------------------------------------------

class OpenAIKeyMiddleware(BaseHTTPMiddleware):
    """Injects the effective OpenAI API key into request-scoped contextvar.

    Priority:
    1. X-OpenAI-Key header (per-request, highest priority)
    2. OPENAI_API_KEY env var (container-level default)
    """

    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        key = request.headers.get("x-openai-key", "").strip()
        if not key:
            key = os.environ.get("OPENAI_API_KEY", "").strip()
        token = current_openai_key.set(key)
        try:
            response = await call_next(request)
        finally:
            current_openai_key.reset(token)
        return response


# ---------------------------------------------------------------------------
# Middleware: Security headers
# ---------------------------------------------------------------------------

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add standard security headers to all responses."""

    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Cache-Control"] = "no-store"
        if "server" in response.headers:
            del response.headers["server"]
        return response


# ---------------------------------------------------------------------------
# Middleware: Request size limit
# ---------------------------------------------------------------------------

class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """Reject request bodies larger than MAX_BODY_SIZE."""

    MAX_BODY_SIZE = 1_048_576  # 1 MB

    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.MAX_BODY_SIZE:
            return JSONResponse(
                status_code=413,
                content={"detail": f"Request body too large. Max: {self.MAX_BODY_SIZE} bytes."},
            )
        return await call_next(request)
