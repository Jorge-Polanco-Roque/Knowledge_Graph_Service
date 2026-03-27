"""LangGraph agent state definition."""

from __future__ import annotations

from typing import TypedDict

from ..models.graph import GraphDiff
from ..models.messages import ChatMessage, ExtractedConcept


class AgentState(TypedDict, total=False):
    """Typed state flowing through the LangGraph agent pipeline."""

    messages: list[ChatMessage]
    user_id: str
    extracted_concepts: list[ExtractedConcept]
    activated_ids: list[str]
    graph_context: str
    document_chunks: list[dict]
    response: str
    graph_diff: GraphDiff
    spreading_waves: list[list[str]]
