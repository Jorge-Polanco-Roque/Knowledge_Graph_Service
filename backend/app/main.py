"""FastAPI application entry point for NeuroGraph."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.routes import router as api_router
from .api.websocket import websocket_endpoint
from .config import settings

logging.basicConfig(
    level=logging.INFO,
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
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# REST routes
app.include_router(api_router)

# WebSocket route
app.add_api_websocket_route("/ws/{user_id}", websocket_endpoint)
