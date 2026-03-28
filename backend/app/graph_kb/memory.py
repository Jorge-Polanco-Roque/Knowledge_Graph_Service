"""Dual memory system — Short-Term Memory (STM) and Long-Term Memory (LTM).

STM acts like the hippocampus: high plasticity, rapid decay, limited capacity.
LTM acts like the neocortex: stable, persistent, consolidated from STM.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

from ..config import settings
from ..models.graph import (
    DiffAction,
    EdgeData,
    EdgeDiff,
    GraphDiff,
    MemoryType,
    NodeData,
    NodeDiff,
)
from .knowledge_graph import KnowledgeGraph
from .plasticity import PlasticityEngine
from .spreading import SpreadingActivation


def _now() -> datetime:
    return datetime.now(timezone.utc)


class MemoryManager:
    """Manages dual memory (STM + LTM) for a single user."""

    _locks: dict[str, threading.Lock] = {}

    def __init__(self, user_id: str, data_dir: Path | None = None) -> None:
        self.user_id = user_id
        self._data_dir = data_dir or settings.data_dir
        self._graph_path = self._data_dir / "users" / user_id / "graph.json"

        # Per-user lock for thread-safe graph mutations
        if user_id not in self._locks:
            self._locks[user_id] = threading.Lock()
        self._lock = self._locks[user_id]

        # Single unified graph — nodes are tagged as STM or LTM
        self.graph = KnowledgeGraph.load(self._graph_path)
        self.plasticity = PlasticityEngine(self.graph)
        self.spreading = SpreadingActivation(self.graph)

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def activate_concepts(
        self, concepts: list[dict],
    ) -> tuple[list[str], GraphDiff]:
        """Process extracted concepts: add/activate nodes, apply plasticity, return diff.

        Thread-safe: acquires per-user lock to prevent concurrent graph corruption.

        Args:
            concepts: list of dicts with keys: label, category, relationships (optional)

        Returns:
            activated_ids: list of activated node IDs
            diff: graph changes for the frontend
        """
        with self._lock:
            snapshot_before = self.graph.snapshot()
            activated_ids: list[str] = []

            for concept in concepts:
                label = concept["label"]
                if self._is_sensitive_concept(label):
                    logger.warning("Blocked sensitive concept: %s", label)
                    continue

                category = concept.get("category", "concept")
                node_id = self._label_to_id(label)

                existing = self.graph.get_node(node_id)
                if existing is None:
                    node = NodeData(
                        id=node_id, label=label, category=category,
                        activation=1.0, importance=0.1,
                        memory_type=MemoryType.STM, activation_count=1,
                    )
                    self.graph.add_node(node)
                else:
                    self.graph.update_node(
                        node_id,
                        activation=min(1.0, existing.activation + 0.5),
                        activation_count=existing.activation_count + 1,
                        last_activated=_now().isoformat(),
                    )
                activated_ids.append(node_id)

                for rel in concept.get("relationships", []):
                    target_label = rel["target_label"]
                    target_id = self._label_to_id(target_label)
                    rel_type = rel.get("relationship", "related_to")
                    valid_rels = {"related_to", "is_a", "part_of", "causes", "has_property",
                                  "used_for", "located_in", "created_by", "depends_on", "opposite_of"}
                    if rel_type not in valid_rels:
                        rel_type = "related_to"

                    if not self.graph.has_node(target_id):
                        target_node = NodeData(
                            id=target_id, label=target_label, category="concept",
                            activation=0.5, memory_type=MemoryType.STM, activation_count=1,
                        )
                        self.graph.add_node(target_node)
                        activated_ids.append(target_id)

                    if not self.graph.get_edge(node_id, target_id):
                        edge = EdgeData(
                            source=node_id, target=target_id,
                            weight=settings.hebbian_learning_rate,
                            relationship=rel_type, co_activation_count=1,
                        )
                        self.graph.add_edge(edge)

            # Fuzzy-match extracted concepts against existing graph nodes
            for concept in concepts:
                related = self._find_related_nodes(concept["label"], max_results=5)
                for rid in related:
                    if rid not in activated_ids:
                        activated_ids.append(rid)
                        existing = self.graph.get_node(rid)
                        if existing:
                            self.graph.update_node(
                                rid, activation=min(1.0, existing.activation + 0.3),
                                last_activated=_now().isoformat(),
                            )

            self.plasticity.apply_plasticity(activated_ids)
            self._enforce_stm_capacity()
            self._consolidate()

            snapshot_after = self.graph.snapshot()
            diff = self.graph.compute_diff(snapshot_before, snapshot_after)
            diff.activated_nodes = activated_ids

            return activated_ids, diff

    def spread_and_collect_context(
        self, seed_ids: list[str], max_depth: int = 2, query: str | None = None,
    ) -> tuple[list[NodeData], list[EdgeData], list[list[str]]]:
        """Spread activation from seeds, optionally weighted by query relevance."""
        query_similarities = None
        if query:
            try:
                from .embedding_cache import get_embedding_cache
                cache = get_embedding_cache(self.user_id)
                cache.ensure_fresh(self.graph)
                query_similarities = cache.get_similarities(query)
            except Exception as e:
                logger.warning("Embedding cache failed, using uniform spreading: %s", e)

        activations, waves = self.spreading.spread(
            seed_ids, max_depth=max_depth, query_similarities=query_similarities,
        )

        # Collect all activated node IDs (seeds + spread targets)
        relevant_ids = list(activations.keys())
        sub = self.graph.subgraph(relevant_ids)

        return sub.nodes, sub.edges, waves

    def get_context_for_prompt(self, seed_ids: list[str], query: str | None = None) -> str:
        """Build a textual context from the active subgraph for the LLM prompt.

        Caps context to max_context_concepts and max_context_relationships
        to keep token usage bounded and response times fast.
        """
        nodes, edges, _ = self.spread_and_collect_context(seed_ids, max_depth=2, query=query)

        if not nodes:
            return ""

        # Cap: top concepts by activation score
        nodes = sorted(nodes, key=lambda n: n.activation, reverse=True)[:settings.max_context_concepts]
        node_ids = {n.id for n in nodes}

        # Only include edges where both endpoints are in the capped node set
        edges = [e for e in edges if e.source in node_ids and e.target in node_ids]
        edges = sorted(edges, key=lambda e: e.weight, reverse=True)[:settings.max_context_relationships]

        lines = ["## Relevant Knowledge Graph Context\n"]
        lines.append("### Concepts:")
        for n in nodes:
            mem = "STM" if n.memory_type == MemoryType.STM else "LTM"
            lines.append(
                f"- **{n.label}** (category: {n.category}, "
                f"importance: {n.importance:.2f}, memory: {mem})"
            )

        if edges:
            lines.append("\n### Relationships:")
            for e in edges:
                src = self.graph.get_node(e.source)
                tgt = self.graph.get_node(e.target)
                src_label = src.label if src else e.source
                tgt_label = tgt.label if tgt else e.target
                lines.append(
                    f"- {src_label} --[{e.relationship}, "
                    f"strength: {e.weight:.2f}]--> {tgt_label}"
                )

        return "\n".join(lines)

    def save(self) -> None:
        """Persist the graph to disk."""
        self.graph.save(self._graph_path)

    def get_full_snapshot(self):
        return self.graph.snapshot()

    def get_stats(self) -> dict:
        stm_count = len(self.graph.nodes_by_memory_type(MemoryType.STM))
        ltm_count = len(self.graph.nodes_by_memory_type(MemoryType.LTM))
        all_nodes = self.graph.all_nodes()
        top_concepts = sorted(all_nodes, key=lambda n: n.importance, reverse=True)[:5]
        return {
            "total_nodes": self.graph.node_count,
            "total_edges": self.graph.edge_count,
            "stm_nodes": stm_count,
            "ltm_nodes": ltm_count,
            "stm_capacity": f"{stm_count}/{settings.stm_capacity}",
            "top_concepts": [
                {"label": n.label, "importance": round(n.importance, 2)}
                for n in top_concepts
            ],
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    _SENSITIVE_PATTERNS = [
        "password", "secret", "credential", "api_key", "api key",
        "private_key", "private key", "ssn", "social_security",
        "credit_card", "credit card", "bank_account", "bank account",
    ]

    def _is_sensitive_concept(self, label: str) -> bool:
        """Check if a concept label contains sensitive information."""
        lower = label.lower()
        return any(p in lower for p in self._SENSITIVE_PATTERNS)

    def _label_to_id(self, label: str) -> str:
        """Deterministic ID from label."""
        return label.lower().strip().replace(" ", "_")

    def _find_related_nodes(self, label: str, max_results: int = 10) -> list[str]:
        """Find existing graph nodes whose labels contain the query (fuzzy match).

        This enables the agent to leverage existing knowledge even when the
        extracted concept label doesn't exactly match a node ID.
        """
        query = label.lower().strip()
        matches: list[tuple[str, float]] = []
        for node in self.graph.all_nodes():
            node_label = node.label.lower()
            # Exact match gets highest score
            if node_label == query:
                matches.append((node.id, 3.0 + node.importance))
            # Node label contains query or query contains node label
            elif query in node_label or node_label in query:
                matches.append((node.id, 2.0 + node.importance))
            # Partial word overlap
            else:
                query_words = set(query.split())
                node_words = set(node_label.split())
                overlap = query_words & node_words
                if overlap and len(overlap) / max(len(query_words), 1) >= 0.5:
                    matches.append((node.id, 1.0 + node.importance))
        matches.sort(key=lambda x: x[1], reverse=True)
        return [m[0] for m in matches[:max_results]]

    def _enforce_stm_capacity(self) -> None:
        """If STM exceeds capacity, evict lowest-activation nodes or consolidate them."""
        stm_nodes = self.graph.nodes_by_memory_type(MemoryType.STM)
        if len(stm_nodes) <= settings.stm_capacity:
            return

        # Sort by activation (lowest first)
        stm_nodes.sort(key=lambda n: n.activation)

        excess = len(stm_nodes) - settings.stm_capacity
        for node in stm_nodes[:excess]:
            if node.activation_count >= 3:
                # Frequently accessed — consolidate to LTM instead of evicting
                self.graph.update_node(node.id, memory_type=MemoryType.LTM.value)
            else:
                # Low activation, low use — remove from graph
                self.graph.remove_node(node.id)

    def _consolidate(self) -> None:
        """Promote STM nodes to LTM when they exceed the consolidation threshold."""
        for node in self.graph.nodes_by_memory_type(MemoryType.STM):
            if node.importance >= settings.consolidation_threshold:
                self.graph.update_node(node.id, memory_type=MemoryType.LTM.value)
            elif node.activation_count >= 5:
                # Repeated activation — consolidate even with lower importance
                self.graph.update_node(node.id, memory_type=MemoryType.LTM.value)
