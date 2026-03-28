"""REST API routes for NeuroGraph."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

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
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    user_msg = ChatMessage(role=MessageRole.USER, content=req.message.strip())

    initial_state = {
        "messages": [user_msg],
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


@router.delete("/graph/{user_id}")
async def reset_graph(user_id: str) -> dict[str, str]:
    """Reset a user's knowledge graph (delete all data)."""
    try:
        mm = _get_memory_manager(user_id)
        # Clear the graph file by saving an empty graph
        from ..graph_kb.knowledge_graph import KnowledgeGraph

        mm.graph = KnowledgeGraph()
        mm.save()
        # Remove cached manager so a fresh one is created next time
        reset_memory_manager(user_id)
        return {"status": "ok", "message": f"Graph for user '{user_id}' has been reset."}
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
