"""FastAPI application entry point for NeuroGraph."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.auth import (
    APIKeyAuthMiddleware,
    OpenAIKeyMiddleware,
    RequestSizeLimitMiddleware,
    SecurityHeadersMiddleware,
)
from .api.routes import router as api_router
from .api.websocket import websocket_endpoint
from .config import settings

log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan context manager."""
    logger.info("NeuroGraph starting up...")
    logger.info("Data directory: %s", settings.data_dir)
    yield
    logger.info("NeuroGraph shutting down...")


app = FastAPI(
    title="NeuroGraph",
    description="Brain-plasticity knowledge graph API",
    version="1.0.0",
    lifespan=lifespan,
)

# --- Middleware stack (order matters: outermost first) ---

# 1. Security headers on all responses
app.add_middleware(SecurityHeadersMiddleware)

# 2. Request body size limit (before parsing)
app.add_middleware(RequestSizeLimitMiddleware)

# 3. CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 4. API key authentication
app.add_middleware(APIKeyAuthMiddleware)

# 5. Per-request OpenAI key injection
app.add_middleware(OpenAIKeyMiddleware)

# Health check at root (no auth, used by Docker/LB)
@app.get("/health", tags=["system"])
async def health() -> dict[str, str]:
    return {"status": "healthy", "version": "1.0.0"}


# REST routes
app.include_router(api_router)

# WebSocket route
app.add_api_websocket_route("/ws/{user_id}", websocket_endpoint)
