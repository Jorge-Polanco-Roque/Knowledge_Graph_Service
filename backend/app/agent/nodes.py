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
from ..graph_kb.document_store import get_document_store
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
# Node: retrieve_documents
# ------------------------------------------------------------------

def retrieve_documents(state: AgentState) -> dict[str, Any]:
    """Retrieve relevant document chunks via dual strategy: semantic + graph-guided."""
    user_id = state["user_id"]
    messages = state["messages"]
    activated_ids = state.get("activated_ids", [])

    query = ""
    for msg in reversed(messages):
        if msg.role == "user":
            query = msg.content
            break

    if not query:
        return {"document_chunks": []}

    try:
        doc_store = get_document_store(user_id)

        # Strategy 1: Pure semantic search
        semantic_chunks = doc_store.search(query, top_k=3)

        # Strategy 2: Graph-guided — search within source docs of activated nodes
        source_chunks: list[dict] = []
        if activated_ids:
            mm = _get_memory_manager(user_id)
            source_files: set[str] = set()
            for nid in activated_ids:
                node = mm.graph.get_node(nid)
                if node and node.metadata.get("source_files"):
                    source_files.update(node.metadata["source_files"])
            if source_files:
                source_chunks = doc_store.search_by_sources(
                    query, list(source_files)[:10], top_k=3
                )

        # Merge and deduplicate by chunk ID
        seen_ids: set[str] = set()
        merged: list[dict] = []
        for chunk in semantic_chunks + source_chunks:
            if chunk["id"] not in seen_ids:
                seen_ids.add(chunk["id"])
                merged.append(chunk)

        return {"document_chunks": merged[:5]}
    except Exception as exc:
        logger.warning("Document retrieval failed: %s", exc)
        return {"document_chunks": []}


# ------------------------------------------------------------------
# Node: build_context
# ------------------------------------------------------------------

def build_context(state: AgentState) -> dict[str, Any]:
    """Build textual context from the active subgraph + document chunks for the LLM prompt."""
    user_id = state["user_id"]
    activated_ids = state.get("activated_ids", [])

    # Graph context from spreading activation
    context = ""
    if activated_ids:
        mm = _get_memory_manager(user_id)
        context = mm.get_context_for_prompt(activated_ids)

    # Append RAG document excerpts
    chunks = state.get("document_chunks", [])
    if chunks:
        doc_lines = ["\n\n## Relevant Document Excerpts\n"]
        for chunk in chunks:
            source = chunk["metadata"].get("source_file", "unknown")
            folder = chunk["metadata"].get("folder", "")
            doc_lines.append(f"### [{source}] (from {folder})")
            doc_lines.append(chunk["content"])
            doc_lines.append("")
        context += "\n".join(doc_lines)

    return {"graph_context": context}


# ------------------------------------------------------------------
# Node: generate_response
# ------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are NeuroGraph, an AI assistant with a dynamic knowledge graph brain.
Your memory works like a biological neural network: concepts have activation levels,
connections strengthen through co-activation (Hebbian learning), and memories
consolidate from short-term to long-term storage.

## Core Principles

1. **Think critically before answering.** Analyze whether the user's question contains false premises, contradictions, or logical errors. If it does, point them out clearly and explain why — do NOT fabricate an answer to fit a flawed question. For example, if someone asks about "the sleeves of a vest", note that vests by definition have no sleeves.

2. **Distinguish what you know from what you don't.** Use the knowledge graph context below when relevant, but never invent facts or relationships that don't exist. If the graph doesn't have information on a topic, say so honestly and rely on your general knowledge instead — clearly marking the boundary.

3. **Reason, don't just retrieve.** Go beyond surface-level pattern matching. Draw inferences, identify implications, and connect ideas across domains. When multiple concepts in your graph are relevant, synthesize them into a coherent insight rather than listing them.

4. **Challenge assumptions when needed.** If the user's premise is wrong or their question is based on a misunderstanding, correct it respectfully before answering. Being helpful means being truthful, not agreeable.

## Knowledge Graph Context

{graph_context}

## Response Guidelines

- **Always ground your response in the Knowledge Graph Context above.** Cite specific concepts, relationships, and data from your graph. If the graph has relevant information, USE IT — don't give a generic answer when you have specific knowledge.
- When document excerpts are provided, **cite them directly**. Use quotes and reference the source file name (e.g., 'According to [article_3.docx]...'). Prefer evidence from documents over general knowledge.
- Synthesize information across multiple sources when relevant.
- If the graph context is empty or irrelevant to the question, say so and answer from general knowledge.
- Be conversational, precise, and insightful. Prioritize accuracy and logical coherence.
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
