"""Plasticity engine — implements biological learning rules on the knowledge graph."""

from __future__ import annotations

from datetime import datetime, timezone

from ..config import settings
from ..models.graph import EdgeData, MemoryType, NodeData
from .knowledge_graph import KnowledgeGraph


def _now() -> datetime:
    return datetime.now(timezone.utc)


class PlasticityEngine:
    """Applies Hebbian learning, LTP, LTD, and pruning to a KnowledgeGraph."""

    def __init__(self, graph: KnowledgeGraph) -> None:
        self.graph = graph

    # ------------------------------------------------------------------
    # Hebbian Learning: "neurons that fire together wire together"
    # ------------------------------------------------------------------

    def hebbian_update(self, co_activated_ids: list[str]) -> list[EdgeData]:
        """Strengthen edges between all pairs of co-activated concepts.

        If an edge doesn't exist, create it. Returns affected edges.
        """
        updated: list[EdgeData] = []
        for i, src in enumerate(co_activated_ids):
            for tgt in co_activated_ids[i + 1 :]:
                if not self.graph.has_node(src) or not self.graph.has_node(tgt):
                    continue
                edge = self.graph.get_edge(src, tgt)
                if edge is None:
                    # Also check reverse direction
                    edge = self.graph.get_edge(tgt, src)
                if edge is None:
                    # Create new connection (neurogenesis of synapses)
                    new_edge = EdgeData(
                        source=src,
                        target=tgt,
                        weight=settings.hebbian_learning_rate,
                        co_activation_count=1,
                        last_co_activated=_now(),
                    )
                    self.graph.add_edge(new_edge)
                    updated.append(new_edge)
                else:
                    # Strengthen existing connection
                    new_weight = min(1.0, edge.weight + settings.hebbian_learning_rate)
                    result = self.graph.update_edge(
                        edge.source,
                        edge.target,
                        weight=new_weight,
                        co_activation_count=edge.co_activation_count + 1,
                        last_co_activated=_now().isoformat(),
                    )
                    if result:
                        updated.append(result)
        return updated

    # ------------------------------------------------------------------
    # LTP: Long-term Potentiation — repeated activation increases importance
    # ------------------------------------------------------------------

    def ltp_update(self, activated_ids: list[str]) -> list[NodeData]:
        """Increase importance and activation of activated nodes."""
        updated: list[NodeData] = []
        for node_id in activated_ids:
            node = self.graph.get_node(node_id)
            if node is None:
                continue
            new_importance = min(1.0, node.importance + settings.ltp_rate)
            new_activation = min(1.0, node.activation + 0.3)
            result = self.graph.update_node(
                node_id,
                importance=new_importance,
                activation=new_activation,
                activation_count=node.activation_count + 1,
                last_activated=_now().isoformat(),
            )
            if result:
                updated.append(result)
        return updated

    # ------------------------------------------------------------------
    # LTD: Long-term Depression — decay for inactive connections
    # ------------------------------------------------------------------

    def ltd_decay(self, active_ids: set[str] | None = None) -> list[EdgeData]:
        """Apply decay to all edges NOT involving currently active nodes.

        Returns edges that were weakened.
        """
        active_ids = active_ids or set()
        weakened: list[EdgeData] = []
        for edge in self.graph.all_edges():
            if edge.source in active_ids or edge.target in active_ids:
                continue
            new_weight = max(0.0, edge.weight - settings.ltd_decay_rate)
            result = self.graph.update_edge(edge.source, edge.target, weight=new_weight)
            if result:
                weakened.append(result)
        return weakened

    def decay_node_activations(self, active_ids: set[str] | None = None) -> None:
        """Decay activation level of inactive nodes (STM rapid decay)."""
        active_ids = active_ids or set()
        for node in self.graph.all_nodes():
            if node.id in active_ids:
                continue
            if node.memory_type == MemoryType.STM:
                new_activation = max(0.0, node.activation - settings.stm_decay_rate)
            else:
                new_activation = max(0.0, node.activation - settings.ltd_decay_rate)
            self.graph.update_node(node.id, activation=new_activation)

    # ------------------------------------------------------------------
    # Pruning: remove weak connections
    # ------------------------------------------------------------------

    def prune(self) -> list[EdgeData]:
        """Remove edges whose weight has fallen below the pruning threshold."""
        pruned: list[EdgeData] = []
        for edge in self.graph.all_edges():
            if edge.weight < settings.pruning_threshold:
                self.graph.remove_edge(edge.source, edge.target)
                pruned.append(edge)
        return pruned

    # ------------------------------------------------------------------
    # Combined plasticity step
    # ------------------------------------------------------------------

    def apply_plasticity(
        self, activated_ids: list[str]
    ) -> dict[str, list]:
        """Run a full plasticity cycle and return all changes.

        Steps:
        1. LTP on activated nodes
        2. Hebbian on co-activated pairs
        3. LTD decay on inactive edges
        4. Decay node activations
        5. Prune weak edges
        """
        active_set = set(activated_ids)

        ltp_nodes = self.ltp_update(activated_ids)
        hebbian_edges = self.hebbian_update(activated_ids)
        ltd_edges = self.ltd_decay(active_set)
        self.decay_node_activations(active_set)
        pruned_edges = self.prune()

        return {
            "ltp_nodes": ltp_nodes,
            "hebbian_edges": hebbian_edges,
            "ltd_edges": ltd_edges,
            "pruned_edges": pruned_edges,
        }
