"""Spreading activation algorithm — simulates neural signal propagation."""

from __future__ import annotations

from ..config import settings
from .knowledge_graph import KnowledgeGraph


class SpreadingActivation:
    """Propagate activation from seed nodes through the graph, weighted by edge strength."""

    def __init__(self, graph: KnowledgeGraph) -> None:
        self.graph = graph

    def spread(
        self,
        seed_ids: list[str],
        max_depth: int = 2,
        min_activation: float = 0.05,
        query_similarities: dict[str, float] | None = None,
    ) -> tuple[dict[str, float], list[list[str]]]:
        """Spread activation from seed nodes, optionally weighted by query relevance.

        Args:
            seed_ids: starting nodes for activation
            max_depth: how many hops to propagate
            min_activation: threshold below which propagation stops
            query_similarities: {node_id: cosine_similarity} for contextual weighting.
                When provided, neighbors more relevant to the query get stronger activation.

        Returns:
            activations: mapping of node_id -> activation level after spreading
            waves: list of node ID lists per depth level (for animation)
        """
        activations: dict[str, float] = {}
        waves: list[list[str]] = []

        # Initialize seeds with full activation
        current_layer = set()
        for sid in seed_ids:
            if self.graph.has_node(sid):
                activations[sid] = 1.0
                current_layer.add(sid)

        if current_layer:
            waves.append(list(current_layer))

        visited = set(current_layer)

        for _depth in range(max_depth):
            next_layer: set[str] = set()
            for node_id in current_layer:
                parent_activation = activations.get(node_id, 0.0)
                edges = self.graph.edges_of_node(node_id)
                # Limit hub nodes: only traverse top edges by weight
                if len(edges) > settings.max_edges_per_node:
                    edges = sorted(edges, key=lambda e: e.weight, reverse=True)[:settings.max_edges_per_node]
                for edge in edges:
                    # Determine the neighbor
                    neighbor_id = (
                        edge.target if edge.source == node_id else edge.source
                    )
                    if neighbor_id in visited:
                        continue

                    # Context factor: modulate by query relevance if available
                    if query_similarities and neighbor_id in query_similarities:
                        sim = max(0.0, query_similarities[neighbor_id])
                        context_factor = 0.3 + 0.7 * sim
                    else:
                        context_factor = 1.0  # No query info → uniform spreading

                    # Activation = parent × edge_weight × spreading_factor × context_factor
                    spread_value = parent_activation * edge.weight * settings.spreading_factor * context_factor

                    if spread_value < min_activation:
                        continue

                    # Keep the max if reached from multiple parents
                    existing = activations.get(neighbor_id, 0.0)
                    activations[neighbor_id] = max(existing, spread_value)
                    next_layer.add(neighbor_id)

                    # Update node activation in the graph
                    node = self.graph.get_node(neighbor_id)
                    if node:
                        new_act = min(1.0, node.activation + spread_value)
                        self.graph.update_node(neighbor_id, activation=new_act)

            if not next_layer:
                break

            waves.append(list(next_layer))
            visited |= next_layer
            current_layer = next_layer

        return activations, waves
