from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from .graph import GraphDiff


def _now() -> datetime:
    return datetime.now(timezone.utc)


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ChatMessage(BaseModel):
    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=_now)


class ExtractedConcept(BaseModel):
    """A concept extracted from a user message by the LLM."""

    label: str
    category: str = "concept"
    relationships: list[ConceptRelationship] = Field(default_factory=list)


class ConceptRelationship(BaseModel):
    """A relationship between two extracted concepts."""

    target_label: str
    relationship: str = "related_to"


# Rebuild ExtractedConcept so the forward ref to ConceptRelationship resolves
ExtractedConcept.model_rebuild()


# --- WebSocket message types ---


class WSMessageType(str, Enum):
    # Client -> Server
    USER_MESSAGE = "user_message"
    # Server -> Client
    AGENT_RESPONSE = "agent_response"
    RESPONSE_CHUNK = "response_chunk"
    RESPONSE_END = "response_end"
    GRAPH_DIFF = "graph_diff"
    SPREADING_ACTIVATION = "spreading_activation"
    ERROR = "error"


class WSMessage(BaseModel):
    """Envelope for all WebSocket messages."""

    type: WSMessageType
    data: dict[str, Any] = Field(default_factory=dict)


class AgentResponseData(BaseModel):
    content: str
    extracted_concepts: list[ExtractedConcept] = Field(default_factory=list)
    graph_diff: GraphDiff = Field(default_factory=GraphDiff)
