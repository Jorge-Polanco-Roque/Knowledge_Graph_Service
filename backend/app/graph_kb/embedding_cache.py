"""Persistent cache for node label embeddings — enables contextual spreading activation."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

import numpy as np
from langchain_openai import OpenAIEmbeddings

from ..config import settings
from .knowledge_graph import KnowledgeGraph

logger = logging.getLogger(__name__)

_caches: dict[str, "NodeEmbeddingCache"] = {}


def get_embedding_cache(user_id: str) -> NodeEmbeddingCache:
    """Get or create an embedding cache for the given user."""
    if user_id not in _caches:
        _caches[user_id] = NodeEmbeddingCache(user_id, settings.data_dir)
    return _caches[user_id]


class NodeEmbeddingCache:
    """Persistent cache for node label embeddings.

    Stores embeddings as a numpy .npz file. Rebuilds automatically
    when graph nodes change (add/remove/merge).
    """

    def __init__(self, user_id: str, data_dir: Path) -> None:
        self._user_id = user_id
        self._path = data_dir / "users" / user_id / "node_embeddings.npz"
        self._embeddings_model = OpenAIEmbeddings(
            model=settings.rag_embedding_model,
            api_key=settings.openai_api_key,
        )
        self._ids: np.ndarray | None = None
        self._embeddings: np.ndarray | None = None
        self._version: str = ""
        self._load()

    def _load(self) -> None:
        """Load cache from disk if it exists."""
        if self._path.exists():
            try:
                data = np.load(self._path, allow_pickle=True)
                self._ids = data["ids"]
                self._embeddings = data["embeddings"]
                self._version = str(data["version"])
            except Exception as e:
                logger.warning("Failed to load embedding cache: %s", e)
                self._ids = None
                self._embeddings = None
                self._version = ""

    def _save(self) -> None:
        """Persist cache to disk."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            self._path,
            ids=self._ids,
            embeddings=self._embeddings,
            version=np.array(self._version),
        )

    def _compute_version(self, node_ids: list[str]) -> str:
        """Hash of sorted node IDs for cache invalidation."""
        return hashlib.md5(",".join(sorted(node_ids)).encode()).hexdigest()

    def ensure_fresh(self, graph: KnowledgeGraph) -> None:
        """Rebuild cache if stale (LTM nodes changed since last build).

        Only LTM nodes count for version — STM changes (every query) don't
        invalidate the cache. This prevents expensive rebuilds on every query.
        """
        all_nodes = graph.all_nodes()
        # Version based on LTM nodes only — STM is ephemeral
        ltm_ids = [n.id for n in all_nodes
                    if n.memory_type.value == "ltm" and not n.metadata.get("is_folder")]
        current_version = self._compute_version(ltm_ids)

        if current_version == self._version and self._embeddings is not None:
            return  # Cache is fresh

        # Rebuild with ALL non-folder nodes (including STM for query matching)
        labels: list[str] = []
        ids: list[str] = []
        for n in all_nodes:
            if not n.metadata.get("is_folder") and len(n.label) > 1:
                labels.append(n.label)
                ids.append(n.id)

        if not labels:
            self._ids = np.array([])
            self._embeddings = np.zeros((0, 1536))
            self._version = current_version
            self._save()
            return

        raw_embeddings = self._embeddings_model.embed_documents(labels)
        self._ids = np.array(ids)
        self._embeddings = np.array(raw_embeddings)
        self._version = current_version
        self._save()
        logger.info(
            "Rebuilt embedding cache: %d nodes (user: %s, version: %s)",
            len(ids), self._user_id, current_version[:8],
        )

    def get_similarities(self, query: str) -> dict[str, float]:
        """Compute cosine similarity between query and all cached node labels.

        Returns: {node_id: similarity} for all cached nodes.
        """
        if self._embeddings is None or len(self._embeddings) == 0:
            return {}

        query_embedding = np.array(self._embeddings_model.embed_query(query))

        # Cosine similarity: dot(q, E) / (||q|| × ||E||)
        norms = np.linalg.norm(self._embeddings, axis=1)
        query_norm = np.linalg.norm(query_embedding)
        if query_norm == 0:
            return {}

        similarities = (self._embeddings @ query_embedding) / (norms * query_norm + 1e-10)

        return {str(self._ids[i]): float(similarities[i]) for i in range(len(self._ids))}
