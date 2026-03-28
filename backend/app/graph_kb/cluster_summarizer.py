"""Generate LLM summaries for graph clusters."""

from __future__ import annotations

import logging
from typing import Any

from langchain_openai import ChatOpenAI

from ..config import settings
from .document_store import get_document_store
from .knowledge_graph import KnowledgeGraph

logger = logging.getLogger(__name__)

CLUSTER_SUMMARY_PROMPT = """\
You are summarizing a thematic cluster from a knowledge graph.

Cluster label: {label}
Top concepts (by importance):
{concepts}

Key relationships:
{relationships}

{document_context}

Write a 2-3 sentence summary of what this cluster of knowledge represents.
Focus on the main theme, key insights, and how the concepts connect.
Be specific — mention actual concept names. Write in the same language as the concepts.
"""


def generate_cluster_summaries(
    graph: KnowledgeGraph,
    clusters: list[dict[str, Any]],
    user_id: str,
) -> list[dict[str, Any]]:
    """Generate LLM summaries for each cluster. Modifies clusters in place and returns them."""
    llm = ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        temperature=0.2,
    )

    try:
        doc_store = get_document_store(user_id)
    except Exception:
        doc_store = None

    for cluster in clusters:
        if cluster.get("label") == "Other" or cluster.get("size", 0) < 3:
            cluster["summary"] = ""
            continue

        # Get top 10 nodes by importance
        cluster_nodes = [graph.get_node(nid) for nid in cluster.get("node_ids", [])]
        cluster_nodes = [n for n in cluster_nodes if n is not None]
        top_nodes = sorted(cluster_nodes, key=lambda n: n.importance, reverse=True)[:10]

        if not top_nodes:
            cluster["summary"] = ""
            continue

        # Get relationships between cluster nodes
        cluster_id_set = set(cluster.get("node_ids", []))
        edge_strs: list[str] = []
        seen_edges: set[str] = set()
        for nid in list(cluster_id_set)[:20]:
            for edge in graph.edges_of_node(nid):
                edge_key = f"{edge.source}->{edge.target}"
                if edge_key in seen_edges:
                    continue
                seen_edges.add(edge_key)
                if edge.source in cluster_id_set and edge.target in cluster_id_set:
                    src = graph.get_node(edge.source)
                    tgt = graph.get_node(edge.target)
                    if src and tgt:
                        edge_strs.append(f"{src.label} --[{edge.relationship}]--> {tgt.label}")

        # Sample document chunks if available
        doc_context = ""
        if doc_store:
            source_files: set[str] = set()
            for n in top_nodes[:5]:
                source_files.update(n.metadata.get("source_files", []))
            if source_files:
                try:
                    chunks = doc_store.search_by_sources(
                        cluster["label"], list(source_files)[:3], top_k=2
                    )
                    if chunks:
                        doc_context = "Sample document excerpts:\n" + "\n".join(
                            c["content"][:200] for c in chunks
                        )
                except Exception:
                    pass

        prompt = CLUSTER_SUMMARY_PROMPT.format(
            label=cluster["label"],
            concepts="\n".join(
                f"- {n.label} (importance: {n.importance:.0%})" for n in top_nodes
            ),
            relationships="\n".join(edge_strs[:10]) if edge_strs else "No direct relationships found.",
            document_context=doc_context,
        )

        try:
            response = llm.invoke([{"role": "user", "content": prompt}])
            cluster["summary"] = response.content.strip()
        except Exception as exc:
            logger.warning("Failed to generate summary for cluster '%s': %s", cluster["label"], exc)
            cluster["summary"] = ""

    return clusters
