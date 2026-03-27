"""LangGraph node functions for the NeuroGraph agent pipeline."""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_openai import ChatOpenAI

from ..config import settings
from ..graph_kb.memory import MemoryManager
from ..models.graph import GraphDiff
from ..models.messages import ConceptRelationship, ExtractedConcept
from .state import AgentState

logger = logging.getLogger(__name__)

# Cache MemoryManager instances per user to avoid reloading from disk every turn
_memory_managers: dict[str, MemoryManager] = {}


def _get_memory_manager(user_id: str) -> MemoryManager:
    """Get or create a MemoryManager for the given user."""
    if user_id not in _memory_managers:
        _memory_managers[user_id] = MemoryManager(user_id)
    return _memory_managers[user_id]


def reset_memory_manager(user_id: str) -> None:
    """Remove a cached MemoryManager (used when resetting a user graph)."""
    _memory_managers.pop(user_id, None)


def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        temperature=0.3,
    )


# ------------------------------------------------------------------
# Node: extract_concepts
# ------------------------------------------------------------------

EXTRACTION_PROMPT = """\
You are a concept extraction engine for a knowledge graph.
Analyze the user's message and extract all meaningful concepts, entities, and relationships.

Return ONLY valid JSON in this exact format (no markdown, no explanation):
{
  "concepts": [
    {
      "label": "concept name",
      "category": "person|place|concept|action|object|event",
      "relationships": [
        {
          "target_label": "other concept",
          "relationship": "related_to|is_a|part_of|causes|has_property|used_for|located_in|created_by|depends_on|opposite_of"
        }
      ]
    }
  ]
}

Extract between 1 and 8 concepts. Focus on the most meaningful ones.
Every relationship target must also appear as a concept in the list.
"""


def extract_concepts(state: AgentState) -> dict[str, Any]:
    """Use the LLM to extract concepts and relationships from the latest user message."""
    messages = state["messages"]

    # Find the latest user message
    user_message = ""
    for msg in reversed(messages):
        if msg.role == "user":
            user_message = msg.content
            break

    if not user_message:
        return {"extracted_concepts": []}

    llm = _get_llm()
    response = llm.invoke([
        {"role": "system", "content": EXTRACTION_PROMPT},
        {"role": "user", "content": user_message},
    ])

    # Parse the JSON response
    try:
        raw = response.content.strip()
        # Handle potential markdown code blocks
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]
            raw = raw.rsplit("```", 1)[0]
        parsed = json.loads(raw)
        raw_concepts = parsed.get("concepts", [])
    except (json.JSONDecodeError, KeyError, IndexError) as exc:
        logger.warning("Failed to parse concept extraction response: %s", exc)
        return {"extracted_concepts": []}

    concepts: list[ExtractedConcept] = []
    for c in raw_concepts:
        relationships = [
            ConceptRelationship(
                target_label=r.get("target_label", ""),
                relationship=r.get("relationship", "related_to"),
            )
            for r in c.get("relationships", [])
            if r.get("target_label")
        ]
        concepts.append(
            ExtractedConcept(
                label=c.get("label", "unknown"),
                category=c.get("category", "concept"),
                relationships=relationships,
            )
        )

    return {"extracted_concepts": concepts}


# ------------------------------------------------------------------
# Node: activate_memory
# ------------------------------------------------------------------

def activate_memory(state: AgentState) -> dict[str, Any]:
    """Activate extracted concepts in the user's memory graph."""
    user_id = state["user_id"]
    extracted = state.get("extracted_concepts", [])

    if not extracted:
        return {
            "activated_ids": [],
            "graph_diff": GraphDiff(),
        }

    mm = _get_memory_manager(user_id)

    # Convert ExtractedConcept models to dicts expected by MemoryManager
    concept_dicts = [
        {
            "label": c.label,
            "category": c.category,
            "relationships": [
                {"target_label": r.target_label, "relationship": r.relationship}
                for r in c.relationships
            ],
        }
        for c in extracted
    ]

    activated_ids, diff = mm.activate_concepts(concept_dicts)

    return {
        "activated_ids": activated_ids,
        "graph_diff": diff,
    }


# ------------------------------------------------------------------
# Node: build_context
# ------------------------------------------------------------------

def build_context(state: AgentState) -> dict[str, Any]:
    """Build textual context from the active subgraph for the LLM prompt."""
    user_id = state["user_id"]
    activated_ids = state.get("activated_ids", [])

    if not activated_ids:
        return {"graph_context": ""}

    mm = _get_memory_manager(user_id)
    context = mm.get_context_for_prompt(activated_ids)

    return {"graph_context": context}


# ------------------------------------------------------------------
# Node: generate_response
# ------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are NeuroGraph, an AI assistant with a dynamic knowledge graph brain.
Your memory works like a biological neural network: concepts have activation levels,
connections strengthen through co-activation (Hebbian learning), and memories
consolidate from short-term to long-term storage.

Use the knowledge graph context below to inform your responses. Reference connections
and concepts you know about when relevant. If you notice patterns or relationships
in your graph, mention them naturally.

{graph_context}

Be conversational, insightful, and reference your evolving understanding when appropriate.
"""


def generate_response(state: AgentState) -> dict[str, Any]:
    """Generate the final response using the LLM with graph context."""
    graph_context = state.get("graph_context", "")
    messages = state["messages"]

    system = SYSTEM_PROMPT.format(
        graph_context=graph_context if graph_context else "No relevant context yet."
    )

    # Build message history for the LLM
    llm_messages: list[dict[str, str]] = [{"role": "system", "content": system}]
    for msg in messages:
        llm_messages.append({"role": msg.role.value, "content": msg.content})

    llm = _get_llm()
    response = llm.invoke(llm_messages)

    return {"response": response.content}


# ------------------------------------------------------------------
# Node: update_graph
# ------------------------------------------------------------------

def update_graph(state: AgentState) -> dict[str, Any]:
    """Persist the graph and collect spreading activation waves."""
    user_id = state["user_id"]
    activated_ids = state.get("activated_ids", [])

    mm = _get_memory_manager(user_id)

    # Get spreading waves for animation
    spreading_waves: list[list[str]] = []
    if activated_ids:
        _, _, waves = mm.spread_and_collect_context(activated_ids)
        spreading_waves = waves

    # Persist to disk
    mm.save()

    # Update diff with spreading waves
    diff = state.get("graph_diff", GraphDiff())
    diff.spreading_waves = spreading_waves

    return {
        "graph_diff": diff,
        "spreading_waves": spreading_waves,
    }
