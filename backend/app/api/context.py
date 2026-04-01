"""Per-request context variables for NeuroGraph.

Allows per-request OpenAI API key override without global state.
"""

from __future__ import annotations

import contextvars

# The effective OpenAI API key for the current request.
# Set by middleware; read by agent nodes, embedding cache, etc.
current_openai_key: contextvars.ContextVar[str] = contextvars.ContextVar(
    "openai_key", default=""
)
