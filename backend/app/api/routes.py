"""REST API routes for NeuroGraph."""

from __future__ import annotations

import logging
import re
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

_VALID_USER_ID = re.compile(r'^[a-zA-Z0-9_-]{1,64}$')


def _validate_user_id(user_id: str) -> str:
    """Validate user_id to prevent path traversal and other attacks."""
    if not _VALID_USER_ID.match(user_id):
        raise HTTPException(
            status_code=400,
            detail="Invalid user_id. Only alphanumeric, hyphens, and underscores allowed (max 64 chars).",
        )
    return user_id

from ..agent.graph_agent import create_agent
from ..agent.nodes import _get_memory_manager, reset_memory_manager
from ..config import settings
from ..ingestion.ingest import ingest_all_files, ingest_file
from ..models.messages import ChatMessage, MessageRole

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["graph"])

# Compile agent for REST usage
_agent = create_agent()


# ------------------------------------------------------------------
# Request / response schemas
# ------------------------------------------------------------------

class ChatRequest(BaseModel):
    user_id: str
    message: str
    history: list[dict[str, str]] = []  # Optional: [{"role": "user", "content": "..."}, ...]


class ChatResponse(BaseModel):
    response: str
    extracted_concepts: list[dict[str, Any]] = []
    graph_diff: dict[str, Any] = {}


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------

@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    """Send a message and receive an agent response with graph updates."""
    _validate_user_id(req.user_id)
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    # Build message history (optional) + current message
    messages: list[ChatMessage] = []
    for h in req.history:
        try:
            role = MessageRole(h.get("role", "user"))
            messages.append(ChatMessage(role=role, content=h.get("content", "")))
        except ValueError:
            pass  # Skip invalid roles
    messages.append(ChatMessage(role=MessageRole.USER, content=req.message.strip()))

    initial_state = {
        "messages": messages,
        "user_id": req.user_id,
    }

    try:
        result = await _agent.ainvoke(initial_state)
    except Exception as exc:
        logger.error("Agent error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Agent error: {str(exc)}")

    extracted = result.get("extracted_concepts", [])
    diff = result.get("graph_diff")

    return ChatResponse(
        response=result.get("response", ""),
        extracted_concepts=[c.model_dump(mode="json") for c in extracted],
        graph_diff=diff.model_dump(mode="json") if diff else {},
    )


@router.get("/graph/{user_id}")
async def get_graph(user_id: str) -> dict[str, Any]:
    """Return the full graph snapshot for a user, including cluster info."""
    _validate_user_id(user_id)
    try:
        mm = _get_memory_manager(user_id)
        snapshot = mm.get_full_snapshot()
        clusters = mm.graph.detect_clusters()
        result = snapshot.model_dump(mode="json")
        result["clusters"] = clusters["clusters"]
        result["node_cluster_map"] = clusters["node_cluster_map"]
        return result
    except Exception as exc:
        logger.error("Error fetching graph for %s: %s", user_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/graph/{user_id}/stats")
async def get_graph_stats(user_id: str) -> dict[str, Any]:
    """Return statistics about a user's knowledge graph."""
    _validate_user_id(user_id)
    try:
        mm = _get_memory_manager(user_id)
        return mm.get_stats()
    except Exception as exc:
        logger.error("Error fetching stats for %s: %s", user_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/graph/{user_id}/summarize-clusters")
async def summarize_clusters(user_id: str) -> dict[str, Any]:
    """Generate LLM summaries for each cluster in the graph."""
    try:
        from ..graph_kb.cluster_summarizer import generate_cluster_summaries

        mm = _get_memory_manager(user_id)
        cluster_data = mm.graph.detect_clusters()
        clusters = generate_cluster_summaries(mm.graph, cluster_data["clusters"], user_id)
        return {"clusters": clusters}
    except Exception as exc:
        logger.error("Cluster summary error for %s: %s", user_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/graph/{user_id}/prune-ghosts")
async def prune_ghost_nodes(user_id: str, min_importance: float = 0.3) -> dict[str, Any]:
    """Remove LTM nodes with 0 edges and low importance."""
    try:
        mm = _get_memory_manager(user_id)
        removed = mm.graph.prune_ghost_nodes(min_importance)
        mm.save()
        reset_memory_manager(user_id)
        return {"nodes_removed": removed, "remaining_nodes": mm.graph.node_count}
    except Exception as exc:
        logger.error("Ghost pruning error for %s: %s", user_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/graph/{user_id}/deduplicate")
async def deduplicate_graph(user_id: str, threshold: float = 0.85) -> dict[str, Any]:
    """Merge semantically duplicate nodes using embedding-based cosine similarity."""
    try:
        from ..graph_kb.deduplication import NodeDeduplicator

        mm = _get_memory_manager(user_id)
        dedup = NodeDeduplicator(mm.graph)
        stats = dedup.deduplicate(threshold)
        mm.save()
        reset_memory_manager(user_id)
        return stats
    except Exception as exc:
        logger.error("Deduplication error for %s: %s", user_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/graph/{user_id}/timeline")
async def get_timeline(user_id: str) -> list[dict[str, Any]]:
    """List all graph timeline snapshots."""
    _validate_user_id(user_id)
    from ..graph_kb.timeline import list_snapshots
    return list_snapshots(user_id)


@router.get("/graph/{user_id}/gaps")
async def knowledge_gaps(user_id: str) -> dict[str, Any]:
    """Detect knowledge gaps: nodes without document evidence."""
    _validate_user_id(user_id)
    try:
        mm = _get_memory_manager(user_id)
        return mm.graph.detect_knowledge_gaps()
    except Exception as exc:
        logger.error("Knowledge gaps error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/feedback/{user_id}")
async def submit_feedback(user_id: str, feedback: dict[str, Any]) -> dict[str, str]:
    """Store user feedback. If comment + negative rating, store as learning correction."""
    _validate_user_id(user_id)
    import json as _json
    from ..graph_kb.document_store import get_document_store

    # 1. Store in JSONL (analytics)
    feedback_dir = settings.data_dir / "users" / user_id
    feedback_dir.mkdir(parents=True, exist_ok=True)
    filepath = feedback_dir / "feedback.jsonl"
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(_json.dumps(feedback, ensure_ascii=False) + "\n")

    # 2. If comment provided with negative rating, store as correction chunk in ChromaDB
    comment = feedback.get("comment", "").strip()
    if comment and feedback.get("rating", 0) < 0:
        try:
            doc_store = get_document_store(user_id)
            doc_store.ingest_summary(
                summary=f"User correction: {comment}",
                metadata={
                    "source_file": "user_feedback",
                    "chunk_type": "user_correction",
                    "original_query": feedback.get("query", ""),
                    "timestamp": feedback.get("timestamp", ""),
                },
            )
            logger.info("Stored user correction for %s: %s", user_id, comment[:80])
        except Exception as e:
            logger.warning("Failed to store correction: %s", e)

    return {"status": "ok"}


@router.get("/feedback/{user_id}/summary")
async def feedback_summary(user_id: str) -> dict[str, Any]:
    """Summarize user feedback."""
    _validate_user_id(user_id)
    import json as _json
    filepath = settings.data_dir / "users" / user_id / "feedback.jsonl"
    if not filepath.exists():
        return {"total": 0, "positive": 0, "negative": 0}
    entries = []
    for line in filepath.read_text("utf-8").strip().split("\n"):
        if line:
            entries.append(_json.loads(line))
    positive = sum(1 for e in entries if e.get("rating", 0) > 0)
    negative = sum(1 for e in entries if e.get("rating", 0) < 0)
    return {
        "total": len(entries),
        "positive": positive,
        "negative": negative,
        "worst_queries": [
            e.get("query", "")[:100] for e in entries if e.get("rating", 0) < 0
        ][:10],
    }


@router.delete("/graph/{user_id}")
async def reset_graph(user_id: str) -> dict[str, str]:
    """Full reset: clear knowledge graph, ChromaDB chunks, and embedding cache."""
    _validate_user_id(user_id)
    try:
        from ..graph_kb.knowledge_graph import KnowledgeGraph

        # 1. Reset NetworkX graph
        mm = _get_memory_manager(user_id)
        mm.graph = KnowledgeGraph()
        mm.save()
        reset_memory_manager(user_id)

        # 2. Clear ChromaDB chunks
        try:
            from ..graph_kb.document_store import get_document_store, _document_stores
            doc_store = get_document_store(user_id)
            # Delete all documents in the collection
            if doc_store._collection.count() > 0:
                doc_store._collection.delete(where={"source_file": {"$ne": ""}})
            _document_stores.pop(user_id, None)
        except Exception as e:
            logger.warning("Failed to clear ChromaDB for %s: %s", user_id, e)

        # 3. Delete embedding cache
        try:
            cache_path = settings.data_dir / "users" / user_id / "node_embeddings.npz"
            if cache_path.exists():
                cache_path.unlink()
            from ..graph_kb.embedding_cache import _caches
            _caches.pop(user_id, None)
        except Exception as e:
            logger.warning("Failed to clear embedding cache for %s: %s", user_id, e)

        return {"status": "ok", "message": f"Full reset for '{user_id}': graph, ChromaDB, and embedding cache cleared."}
    except Exception as exc:
        logger.error("Error resetting graph for %s: %s", user_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


# ------------------------------------------------------------------
# Ingestion routes
# ------------------------------------------------------------------

@router.get("/ingest/files")
async def list_input_files() -> dict[str, Any]:
    """List files in the inputs directory organized by folder."""
    import json as _json

    inputs_dir = settings.data_dir.parent / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)

    # Load taxonomy for folder metadata
    taxonomy_path = inputs_dir / ".taxonomy.json"
    taxonomy = {}
    if taxonomy_path.exists():
        taxonomy = _json.loads(taxonomy_path.read_text("utf-8"))
    folder_meta = taxonomy.get("folders", {})

    # Root-level files
    root_files = [
        {"name": f.name, "size": f.stat().st_size, "extension": f.suffix}
        for f in sorted(inputs_dir.iterdir())
        if f.is_file() and not f.name.startswith(".") and not f.name.startswith("~$")
    ]

    # Subfolder files
    folders = {}
    for subfolder in sorted(inputs_dir.iterdir()):
        if not subfolder.is_dir() or subfolder.name.startswith("."):
            continue
        meta = folder_meta.get(subfolder.name, {})
        files_in = [
            {"name": f.name, "size": f.stat().st_size, "extension": f.suffix}
            for f in sorted(subfolder.rglob("*"))
            if f.is_file() and not f.name.startswith(".") and not f.name.startswith("~$")
        ]
        folders[subfolder.name] = {
            "description": meta.get("description", subfolder.name),
            "icon": meta.get("icon", "\U0001F4C1"),
            "files": files_in,
            "count": len(files_in),
        }

    total = sum(f["count"] for f in folders.values()) + len(root_files)
    return {
        "root_files": root_files,
        "folders": folders,
        "total_files": total,
        "path": str(inputs_dir),
    }


@router.post("/ingest/{user_id}")
async def ingest_all(user_id: str) -> dict[str, Any]:
    """Ingest all files from the inputs/ directory into the user's knowledge graph."""
    _validate_user_id(user_id)
    inputs_dir = settings.data_dir.parent / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)

    try:
        result = ingest_all_files(inputs_dir, user_id)
        reset_memory_manager(user_id)
        return result
    except Exception as exc:
        logger.error("Ingestion error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/ingest/{user_id}/{filename}")
async def ingest_single(user_id: str, filename: str) -> dict[str, Any]:
    """Ingest a single file from the inputs/ directory."""
    inputs_dir = settings.data_dir.parent / "inputs"
    file_path = inputs_dir / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")

    try:
        result = ingest_file(file_path, user_id)
        result.pop("graph_diff", None)
        reset_memory_manager(user_id)
        return result
    except Exception as exc:
        logger.error("Ingestion error for %s: %s", filename, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))
