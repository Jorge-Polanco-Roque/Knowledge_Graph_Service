"""NetworkX-backed knowledge graph with CRUD operations and JSON persistence."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import networkx as nx

from ..models.graph import (
    DiffAction,
    EdgeData,
    EdgeDiff,
    GraphDiff,
    GraphSnapshot,
    MemoryType,
    NodeData,
    NodeDiff,
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


class KnowledgeGraph:
    """Wrapper around a NetworkX DiGraph storing concept nodes and weighted edges."""

    def __init__(self) -> None:
        self._graph = nx.DiGraph()

    # ------------------------------------------------------------------
    # Node operations
    # ------------------------------------------------------------------

    def add_node(self, node: NodeData) -> None:
        self._graph.add_node(node.id, **node.model_dump(mode="json"))

    def get_node(self, node_id: str) -> NodeData | None:
        data = self._graph.nodes.get(node_id)
        if data is None or len(data) == 0:
            return None
        return NodeData.model_validate(data)

    def update_node(self, node_id: str, **updates: Any) -> NodeData | None:
        if node_id not in self._graph:
            return None
        self._graph.nodes[node_id].update(updates)
        return self.get_node(node_id)

    def remove_node(self, node_id: str) -> bool:
        if node_id not in self._graph:
            return False
        self._graph.remove_node(node_id)
        return True

    def has_node(self, node_id: str) -> bool:
        return node_id in self._graph

    def all_nodes(self) -> list[NodeData]:
        return [NodeData.model_validate(d) for _, d in self._graph.nodes(data=True) if d]

    def nodes_by_memory_type(self, memory_type: MemoryType) -> list[NodeData]:
        return [n for n in self.all_nodes() if n.memory_type == memory_type]

    # ------------------------------------------------------------------
    # Edge operations
    # ------------------------------------------------------------------

    def add_edge(self, edge: EdgeData) -> None:
        self._graph.add_edge(
            edge.source, edge.target, **edge.model_dump(mode="json")
        )

    def get_edge(self, source: str, target: str) -> EdgeData | None:
        data = self._graph.edges.get((source, target))
        if data is None or len(data) == 0:
            return None
        return EdgeData.model_validate(data)

    def update_edge(self, source: str, target: str, **updates: Any) -> EdgeData | None:
        if not self._graph.has_edge(source, target):
            return None
        self._graph.edges[source, target].update(updates)
        return self.get_edge(source, target)

    def remove_edge(self, source: str, target: str) -> bool:
        if not self._graph.has_edge(source, target):
            return False
        self._graph.remove_edge(source, target)
        return True

    def all_edges(self) -> list[EdgeData]:
        return [
            EdgeData.model_validate(d)
            for _, _, d in self._graph.edges(data=True)
            if d
        ]

    def edges_of_node(self, node_id: str) -> list[EdgeData]:
        """Return all edges (in + out) connected to a node."""
        edges: list[EdgeData] = []
        if node_id not in self._graph:
            return edges
        for _, target, d in self._graph.out_edges(node_id, data=True):
            if d:
                edges.append(EdgeData.model_validate(d))
        for source, _, d in self._graph.in_edges(node_id, data=True):
            if d:
                edges.append(EdgeData.model_validate(d))
        return edges

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def neighbors(self, node_id: str) -> list[NodeData]:
        """Return direct neighbors (both directions) of a node."""
        if node_id not in self._graph:
            return []
        neighbor_ids = set(self._graph.successors(node_id)) | set(
            self._graph.predecessors(node_id)
        )
        return [self.get_node(nid) for nid in neighbor_ids if self.get_node(nid)]

    def detect_clusters(self) -> dict:
        """Detect thematic clusters using Louvain community detection.

        Returns:
            {
                "clusters": [
                    {
                        "id": int,
                        "label": str,  # most important node in cluster
                        "node_ids": [str],
                        "size": int,
                    }
                ],
                "node_cluster_map": { node_id: cluster_id }
            }
        """
        if self.node_count < 2:
            return {"clusters": [], "node_cluster_map": {}}

        # Convert to undirected for community detection
        undirected = self._graph.to_undirected()

        try:
            from networkx.algorithms.community import louvain_communities
            # Higher resolution = more, smaller clusters (scale with edge density)
            communities = louvain_communities(undirected, weight="weight", resolution=1.5, seed=42)
        except Exception:
            # Fallback: connected components
            communities = list(nx.connected_components(undirected))

        # Sort communities by size (largest first) and merge small ones
        sorted_communities = sorted(communities, key=len, reverse=True)

        # Merge communities with < 3 nodes into an "Other" bucket
        min_cluster_size = 3
        main_communities = []
        other_nodes: set[str] = set()
        for community in sorted_communities:
            if len(community) >= min_cluster_size:
                main_communities.append(community)
            else:
                other_nodes |= community

        node_cluster_map: dict[str, int] = {}
        clusters = []

        for idx, community in enumerate(main_communities):
            node_ids = list(community)
            best_node = None
            best_importance = -1
            for nid in node_ids:
                node = self.get_node(nid)
                if node and node.importance > best_importance:
                    if not nid.startswith("_folder_") and not nid.startswith("_file_"):
                        best_importance = node.importance
                        best_node = node
            label = best_node.label if best_node else f"Cluster {idx}"

            clusters.append({
                "id": idx,
                "label": label,
                "node_ids": node_ids,
                "size": len(node_ids),
            })
            for nid in node_ids:
                node_cluster_map[nid] = idx

        # Add "Other" cluster for small communities
        if other_nodes:
            other_id = len(clusters)
            clusters.append({
                "id": other_id,
                "label": "Other",
                "node_ids": list(other_nodes),
                "size": len(other_nodes),
            })
            for nid in other_nodes:
                node_cluster_map[nid] = other_id

        return {"clusters": clusters, "node_cluster_map": node_cluster_map}

    def subgraph(self, node_ids: list[str]) -> GraphSnapshot:
        """Extract a subgraph containing the given nodes and their interconnections."""
        sub = self._graph.subgraph(node_ids)
        nodes = [NodeData.model_validate(d) for _, d in sub.nodes(data=True) if d]
        edges = [EdgeData.model_validate(d) for _, _, d in sub.edges(data=True) if d]
        return GraphSnapshot(nodes=nodes, edges=edges)

    def find_node_by_label(self, label: str) -> NodeData | None:
        """Case-insensitive lookup by label."""
        label_lower = label.lower()
        for _, data in self._graph.nodes(data=True):
            if data and data.get("label", "").lower() == label_lower:
                return NodeData.model_validate(data)
        return None

    def prune_ghost_nodes(self, min_importance: float = 0.3) -> int:
        """Remove LTM nodes with 0 edges and importance below threshold.

        Returns number of nodes removed.
        """
        to_remove = []
        for node in self.all_nodes():
            if (node.memory_type.value == "ltm"
                and not node.metadata.get("is_folder")
                and len(self.edges_of_node(node.id)) == 0
                and node.importance < min_importance):
                to_remove.append(node.id)
        for nid in to_remove:
            self.remove_node(nid)
        return len(to_remove)

    def detect_knowledge_gaps(self) -> dict:
        """Find nodes without document evidence (knowledge gaps)."""
        all_nodes = self.all_nodes()
        gaps = []
        grounded = 0

        for node in all_nodes:
            if node.metadata.get("is_folder"):
                continue
            sources = node.metadata.get("source_files", [])
            edges = len(self.edges_of_node(node.id))

            if not sources and edges > 0:
                gaps.append({
                    "id": node.id,
                    "label": node.label,
                    "importance": round(node.importance, 2),
                    "edges": edges,
                })
            elif sources:
                grounded += 1

        # Cluster-level gaps
        clusters = self.detect_clusters()
        cluster_gaps = []
        for cluster in clusters.get("clusters", []):
            cnodes = [self.get_node(nid) for nid in cluster.get("node_ids", [])]
            cnodes = [n for n in cnodes if n and not n.metadata.get("is_folder")]
            if not cnodes:
                continue
            with_sources = sum(1 for n in cnodes if n.metadata.get("source_files"))
            coverage = with_sources / len(cnodes)
            if coverage < 0.5:
                cluster_gaps.append({
                    "cluster": cluster.get("label", "?"),
                    "size": cluster.get("size", 0),
                    "coverage": f"{coverage:.0%}",
                })

        return {
            "total_nodes": len(all_nodes),
            "grounded_nodes": grounded,
            "gap_nodes": sorted(gaps, key=lambda g: g["importance"], reverse=True)[:20],
            "gap_count": len(gaps),
            "cluster_gaps": cluster_gaps,
        }

    @property
    def node_count(self) -> int:
        return self._graph.number_of_nodes()

    @property
    def edge_count(self) -> int:
        return self._graph.number_of_edges()

    # ------------------------------------------------------------------
    # Snapshot & diff
    # ------------------------------------------------------------------

    def snapshot(self) -> GraphSnapshot:
        return GraphSnapshot(nodes=self.all_nodes(), edges=self.all_edges())

    def compute_diff(self, old: GraphSnapshot, new: GraphSnapshot) -> GraphDiff:
        """Compute the difference between two snapshots."""
        old_node_ids = {n.id for n in old.nodes}
        new_node_ids = {n.id for n in new.nodes}
        old_edge_keys = {(e.source, e.target) for e in old.edges}
        new_edge_keys = {(e.source, e.target) for e in new.edges}

        old_nodes_map = {n.id: n for n in old.nodes}
        new_nodes_map = {n.id: n for n in new.nodes}
        old_edges_map = {(e.source, e.target): e for e in old.edges}
        new_edges_map = {(e.source, e.target): e for e in new.edges}

        node_diffs: list[NodeDiff] = []
        edge_diffs: list[EdgeDiff] = []

        # Added nodes
        for nid in new_node_ids - old_node_ids:
            node_diffs.append(NodeDiff(action=DiffAction.ADDED, node=new_nodes_map[nid]))

        # Removed nodes
        for nid in old_node_ids - new_node_ids:
            node_diffs.append(NodeDiff(action=DiffAction.REMOVED, node=old_nodes_map[nid]))

        # Updated nodes
        for nid in new_node_ids & old_node_ids:
            if new_nodes_map[nid] != old_nodes_map[nid]:
                node_diffs.append(NodeDiff(action=DiffAction.UPDATED, node=new_nodes_map[nid]))

        # Added edges
        for key in new_edge_keys - old_edge_keys:
            edge_diffs.append(EdgeDiff(action=DiffAction.ADDED, edge=new_edges_map[key]))

        # Removed edges
        for key in old_edge_keys - new_edge_keys:
            edge_diffs.append(EdgeDiff(action=DiffAction.REMOVED, edge=old_edges_map[key]))

        # Updated edges
        for key in new_edge_keys & old_edge_keys:
            if new_edges_map[key] != old_edges_map[key]:
                edge_diffs.append(EdgeDiff(action=DiffAction.UPDATED, edge=new_edges_map[key]))

        return GraphDiff(node_diffs=node_diffs, edge_diffs=edge_diffs)

    # ------------------------------------------------------------------
    # Persistence (JSON)
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        snapshot = self.snapshot()
        path.write_text(snapshot.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> KnowledgeGraph:
        kg = cls()
        if not path.exists():
            return kg
        raw = path.read_text(encoding="utf-8")
        snapshot = GraphSnapshot.model_validate_json(raw)
        for node in snapshot.nodes:
            kg.add_node(node)
        for edge in snapshot.edges:
            kg.add_edge(edge)
        return kg
