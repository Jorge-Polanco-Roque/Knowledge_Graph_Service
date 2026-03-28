"""Embedding-based semantic deduplication of graph nodes."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from langchain_openai import OpenAIEmbeddings

from ..config import settings
from ..models.graph import EdgeData
from .knowledge_graph import KnowledgeGraph

logger = logging.getLogger(__name__)


class NodeDeduplicator:
    """Find and merge semantically duplicate nodes using cosine similarity."""

    def __init__(self, graph: KnowledgeGraph) -> None:
        self._graph = graph
        self._embeddings = OpenAIEmbeddings(
            model=settings.rag_embedding_model,
            api_key=settings.openai_api_key,
        )

    def find_duplicates(self, threshold: float = 0.85) -> list[list[str]]:
        """Return groups of node IDs that are semantically duplicate."""
        nodes = self._graph.all_nodes()
        candidates = [n for n in nodes if not n.metadata.get("is_folder") and len(n.label) > 2]

        if len(candidates) < 2:
            return []

        # Batch embed all labels
        labels = [n.label for n in candidates]
        embeddings = self._embeddings.embed_documents(labels)

        # Cosine similarity matrix
        matrix = np.array(embeddings)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-10, norms)
        similarity = (matrix @ matrix.T) / (norms @ norms.T)

        # Union-Find to group connected duplicates
        parent = list(range(len(candidates)))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            parent[find(a)] = find(b)

        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                if similarity[i][j] >= threshold:
                    union(i, j)

        # Collect groups
        groups: dict[int, list[int]] = {}
        for i in range(len(candidates)):
            root = find(i)
            groups.setdefault(root, []).append(i)

        # Return only groups with 2+ members
        return [
            [candidates[i].id for i in group]
            for group in groups.values()
            if len(group) > 1
        ]

    def merge_group(self, node_ids: list[str]) -> str:
        """Merge a group of duplicate nodes into one. Returns the surviving node ID."""
        nodes = [self._graph.get_node(nid) for nid in node_ids]
        nodes = [n for n in nodes if n is not None]
        if len(nodes) < 2:
            return node_ids[0]

        # Primary = highest importance
        primary = max(nodes, key=lambda n: n.importance)
        secondaries = [n for n in nodes if n.id != primary.id]

        for sec in secondaries:
            # Merge metadata
            primary_node = self._graph.get_node(primary.id)
            if not primary_node:
                continue

            merged_sources = list(set(
                primary_node.metadata.get("source_files", []) +
                sec.metadata.get("source_files", [])
            ))
            self._graph.update_node(
                primary.id,
                activation_count=primary_node.activation_count + sec.activation_count,
                importance=max(primary_node.importance, sec.importance),
                metadata={**primary_node.metadata, "source_files": merged_sources},
            )

            # Redirect edges
            for edge in self._graph.edges_of_node(sec.id):
                other = edge.target if edge.source == sec.id else edge.source
                if other == primary.id:
                    continue  # Skip self-loops

                # Check if primary already has this edge (either direction)
                existing = (
                    self._graph.get_edge(primary.id, other)
                    or self._graph.get_edge(other, primary.id)
                )
                if existing:
                    new_weight = min(1.0, max(existing.weight, edge.weight))
                    self._graph.update_edge(existing.source, existing.target, weight=new_weight)
                else:
                    new_source = primary.id if edge.source == sec.id else edge.source
                    new_target = primary.id if edge.target == sec.id else edge.target
                    self._graph.add_edge(EdgeData(
                        source=new_source,
                        target=new_target,
                        weight=edge.weight,
                        relationship=edge.relationship,
                        co_activation_count=edge.co_activation_count,
                    ))

            # Delete secondary node (also removes its edges in NetworkX)
            self._graph.remove_node(sec.id)

        logger.info(
            "Merged %d duplicates into '%s'",
            len(secondaries), primary.label,
        )
        return primary.id

    def deduplicate(self, threshold: float = 0.85) -> dict[str, Any]:
        """Full deduplication pipeline. Returns stats."""
        groups = self.find_duplicates(threshold)
        merged_count = 0
        merge_details: list[dict] = []

        for group in groups:
            labels = []
            for nid in group:
                node = self._graph.get_node(nid)
                if node:
                    labels.append(node.label)
            survivor = self.merge_group(group)
            merged_count += len(group) - 1
            merge_details.append({
                "survivor": survivor,
                "merged_labels": labels,
            })

        return {
            "groups_found": len(groups),
            "nodes_merged": merged_count,
            "details": merge_details,
        }
