from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


def _now() -> datetime:
    return datetime.now(timezone.utc)


class NodeCategory(str, Enum):
    PERSON = "person"
    PLACE = "place"
    CONCEPT = "concept"
    ACTION = "action"
    OBJECT = "object"
    EVENT = "event"


class MemoryType(str, Enum):
    STM = "stm"
    LTM = "ltm"


class RelationshipType(str, Enum):
    RELATED_TO = "related_to"
    IS_A = "is_a"
    PART_OF = "part_of"
    CAUSES = "causes"
    HAS_PROPERTY = "has_property"
    USED_FOR = "used_for"
    LOCATED_IN = "located_in"
    CREATED_BY = "created_by"
    DEPENDS_ON = "depends_on"
    OPPOSITE_OF = "opposite_of"


class NodeData(BaseModel):
    """A concept node in the knowledge graph."""

    id: str
    label: str
    category: NodeCategory = NodeCategory.CONCEPT
    activation: float = Field(default=0.0, ge=0.0, le=1.0)
    importance: float = Field(default=0.1, ge=0.0, le=1.0)
    memory_type: MemoryType = MemoryType.STM
    first_seen: datetime = Field(default_factory=_now)
    last_activated: datetime = Field(default_factory=_now)
    activation_count: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class EdgeData(BaseModel):
    """A weighted connection between two concept nodes."""

    source: str
    target: str
    weight: float = Field(default=0.1, ge=0.0, le=1.0)
    relationship: RelationshipType = RelationshipType.RELATED_TO
    co_activation_count: int = 0
    last_co_activated: datetime = Field(default_factory=_now)


class GraphSnapshot(BaseModel):
    """Full serializable state of a knowledge graph."""

    nodes: list[NodeData] = Field(default_factory=list)
    edges: list[EdgeData] = Field(default_factory=list)


class DiffAction(str, Enum):
    ADDED = "added"
    UPDATED = "updated"
    REMOVED = "removed"


class NodeDiff(BaseModel):
    action: DiffAction
    node: NodeData


class EdgeDiff(BaseModel):
    action: DiffAction
    edge: EdgeData


class GraphDiff(BaseModel):
    """Delta between two graph states — sent over WebSocket."""

    node_diffs: list[NodeDiff] = Field(default_factory=list)
    edge_diffs: list[EdgeDiff] = Field(default_factory=list)
    activated_nodes: list[str] = Field(default_factory=list)
    spreading_waves: list[list[str]] = Field(default_factory=list)
