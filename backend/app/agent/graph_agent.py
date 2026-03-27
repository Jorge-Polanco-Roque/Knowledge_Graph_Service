"""LangGraph agent definition for NeuroGraph."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .nodes import (
    activate_memory,
    build_context,
    extract_concepts,
    generate_response,
    update_graph,
)
from .state import AgentState


def create_agent() -> StateGraph:
    """Build and compile the NeuroGraph LangGraph agent.

    Returns a compiled StateGraph with the following linear pipeline:
        START -> extract_concepts -> activate_memory -> build_context
              -> generate_response -> update_graph -> END
    """
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("extract_concepts", extract_concepts)
    graph.add_node("activate_memory", activate_memory)
    graph.add_node("build_context", build_context)
    graph.add_node("generate_response", generate_response)
    graph.add_node("update_graph", update_graph)

    # Linear flow
    graph.add_edge(START, "extract_concepts")
    graph.add_edge("extract_concepts", "activate_memory")
    graph.add_edge("activate_memory", "build_context")
    graph.add_edge("build_context", "generate_response")
    graph.add_edge("generate_response", "update_graph")
    graph.add_edge("update_graph", END)

    return graph.compile()
