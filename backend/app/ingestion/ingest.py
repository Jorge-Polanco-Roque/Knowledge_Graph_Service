"""Ingestion pipeline — reads files from inputs/ subfolders and feeds them into the knowledge graph.

Folder names act as knowledge categories, enriching the extraction prompt so the LLM
produces more accurate and contextually relevant concepts. Each folder also becomes
a category node in the graph, connecting all concepts extracted from its files.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_openai import ChatOpenAI

from ..config import settings
from ..graph_kb.memory import MemoryManager
from ..models.graph import EdgeData, GraphDiff, NodeData
from ..graph_kb.document_store import get_document_store
from .file_reader import read_file

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Taxonomy
# ------------------------------------------------------------------

_taxonomy_cache: dict | None = None


def _load_taxonomy(inputs_dir: Path) -> dict:
    global _taxonomy_cache
    if _taxonomy_cache is not None:
        return _taxonomy_cache
    taxonomy_path = inputs_dir / ".taxonomy.json"
    if taxonomy_path.exists():
        _taxonomy_cache = json.loads(taxonomy_path.read_text("utf-8"))
    else:
        _taxonomy_cache = {"folders": {}, "root_files_hint": "Extract all meaningful concepts."}
    return _taxonomy_cache


def _folder_context(folder_name: str, taxonomy: dict) -> dict:
    """Return extraction hint and default category for a folder."""
    folders = taxonomy.get("folders", {})
    if folder_name in folders:
        return folders[folder_name]
    return {
        "description": folder_name,
        "extraction_hint": taxonomy.get("root_files_hint", "Extract all meaningful concepts."),
        "default_category": "concept",
        "icon": "📁",
    }


# ------------------------------------------------------------------
# Prompts
# ------------------------------------------------------------------

EXTRACTION_PROMPT = """\
You are a knowledge extraction engine for a brain-like knowledge graph.

CONTEXT: This file is categorized under **{folder_name}** — {folder_description}
{extraction_hint}

File: {filename}

Content:
{content}

Return ONLY valid JSON (no markdown, no explanation):
{{
  "concepts": [
    {{
      "label": "concept name",
      "category": "person|place|concept|action|object|event",
      "relationships": [
        {{
          "target_label": "other concept",
          "relationship": "related_to|is_a|part_of|causes|has_property|used_for|located_in|created_by|depends_on|opposite_of"
        }}
      ]
    }}
  ]
}}

Rules:
- Extract 3-15 concepts depending on content richness.
- Every relationship target must also appear as a concept in the list.
- Prefer the default category "{default_category}" when ambiguous.
- Focus on the most meaningful and distinctive concepts.
"""

IMAGE_PROMPT = """\
You are a knowledge extraction engine for a brain-like knowledge graph.

CONTEXT: This image is categorized under **{folder_name}** — {folder_description}
{extraction_hint}

Image file: {filename}

Return ONLY valid JSON (no markdown, no explanation):
{{
  "concepts": [
    {{
      "label": "concept name",
      "category": "person|place|concept|action|object|event",
      "relationships": [
        {{
          "target_label": "other concept",
          "relationship": "related_to|is_a|part_of|causes|has_property|used_for|located_in|created_by|depends_on|opposite_of"
        }}
      ]
    }}
  ]
}}

Extract 3-15 concepts. Every relationship target must also appear as a concept.
"""

CONVERSATION_PROMPT = """\
You are a knowledge extraction engine for a brain-like knowledge graph.

This is a **conversation transcript** (chat log, meeting notes, or discussion).
{extraction_hint}

File: {filename}

Content:
{content}

Extract:
1. **Participants** as person nodes (if identifiable)
2. **Topics discussed** as concept nodes
3. **Decisions or conclusions** as event nodes
4. **Relationships** between people and topics (who discussed what)

Return ONLY valid JSON (no markdown, no explanation):
{{
  "concepts": [
    {{
      "label": "concept name",
      "category": "person|place|concept|action|object|event",
      "relationships": [
        {{
          "target_label": "other concept",
          "relationship": "related_to|is_a|part_of|causes|has_property|used_for|located_in|created_by|depends_on|opposite_of"
        }}
      ]
    }}
  ]
}}

Extract 3-20 concepts. Every relationship target must also appear as a concept.
"""


# ------------------------------------------------------------------
# LLM helpers
# ------------------------------------------------------------------

def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        temperature=0.2,
    )


def _extract_concepts_from_text(
    filename: str, content: str, folder_name: str, folder_ctx: dict
) -> list[dict]:
    """Use LLM to extract concepts with folder-specific context."""
    if len(content) > 15000:
        content = content[:7000] + "\n\n[...truncated...]\n\n" + content[-3000:]

    llm = _get_llm()

    if folder_name == "conversations":
        prompt = CONVERSATION_PROMPT.format(
            filename=filename,
            content=content,
            extraction_hint=folder_ctx.get("extraction_hint", ""),
        )
    else:
        prompt = EXTRACTION_PROMPT.format(
            filename=filename,
            content=content,
            folder_name=folder_name,
            folder_description=folder_ctx.get("description", folder_name),
            extraction_hint=folder_ctx.get("extraction_hint", ""),
            default_category=folder_ctx.get("default_category", "concept"),
        )

    response = llm.invoke([{"role": "user", "content": prompt}])
    return _parse_concepts_json(response.content)


def _extract_concepts_from_image(
    filename: str, image_base64: str, mime_type: str,
    folder_name: str, folder_ctx: dict
) -> list[dict]:
    """Use GPT-4o vision with folder context."""
    llm = _get_llm()
    prompt = IMAGE_PROMPT.format(
        filename=filename,
        folder_name=folder_name,
        folder_description=folder_ctx.get("description", folder_name),
        extraction_hint=folder_ctx.get("extraction_hint", ""),
    )

    response = llm.invoke([
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{image_base64}",
                        "detail": "auto",
                    },
                },
            ],
        }
    ])
    return _parse_concepts_json(response.content)


def _parse_concepts_json(raw_content: str) -> list[dict]:
    try:
        raw = raw_content.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]
            raw = raw.rsplit("```", 1)[0]
        parsed = json.loads(raw)
        return parsed.get("concepts", [])
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning("Failed to parse extraction response: %s", e)
        return []


# ------------------------------------------------------------------
# Ingestion
# ------------------------------------------------------------------

def ingest_file(
    file_path: Path,
    user_id: str,
    folder_name: str = "root",
    folder_ctx: dict | None = None,
    memory_manager: MemoryManager | None = None,
) -> dict[str, Any]:
    """Ingest a single file into the knowledge graph with folder context."""
    logger.info("Ingesting [%s] %s for user %s", folder_name, file_path.name, user_id)

    folder_ctx = folder_ctx or {
        "description": "general",
        "extraction_hint": "Extract all meaningful concepts.",
        "default_category": "concept",
    }

    # Read the file
    file_data = read_file(file_path)

    # RAG: Store document chunks (text files only, skip images and conversations)
    # Conversations are auto-saved agent responses that may contain hallucinations —
    # ingesting them as RAG sources creates a self-hallucination feedback loop.
    is_conversation = folder_name == "conversations" or file_data["filename"].startswith("chat_")
    if file_data["type"] != "image" and file_data.get("content") and not is_conversation:
        content = file_data["content"]
        if not content.startswith("[Error") and not content.startswith("[Empty"):
            try:
                doc_store = get_document_store(user_id)
                inputs_dir = settings.data_dir.parent / "inputs"
                try:
                    relative_folder = str(file_path.parent.relative_to(inputs_dir))
                except ValueError:
                    relative_folder = folder_name
                # Use folder/filename as source_file to avoid collisions
                # (multiple files named article_1.docx in different folders)
                source_id = f"{relative_folder}/{file_data['filename']}" if relative_folder != folder_name else file_data["filename"]
                chunk_metadata = {
                    "source_file": source_id,
                    "folder": relative_folder,
                    "folder_type": folder_ctx.get("default_category", "general") if folder_ctx else "general",
                }
                doc_store.ingest(content, metadata=chunk_metadata)

                # Auto-generate document summary and store as special chunk
                if len(content) > 200:
                    try:
                        llm = _get_llm()
                        summary_text = content[:3000]
                        summary_resp = llm.invoke([{
                            "role": "user",
                            "content": f"Summarize this document in 2-3 sentences. Be specific — mention key topics, entities, and conclusions.\n\n{summary_text}",
                        }])
                        doc_store.ingest_summary(summary_resp.content.strip(), chunk_metadata)
                        logger.info("Generated summary for %s", file_data["filename"])
                    except Exception as e:
                        logger.warning("Summary generation failed for %s: %s", file_data["filename"], e)
            except Exception as e:
                logger.warning("RAG ingestion failed for %s: %s", file_path.name, e)

    # Extract concepts
    if file_data["type"] == "image":
        concepts = _extract_concepts_from_image(
            file_data["filename"], file_data["image_base64"],
            file_data["mime_type"], folder_name, folder_ctx,
        )
    else:
        content = file_data["content"]
        if not content or content.startswith("[Error") or content.startswith("[Empty"):
            return {
                "filename": file_data["filename"],
                "folder": folder_name,
                "status": "skipped",
                "reason": content,
                "concepts_added": 0,
            }
        concepts = _extract_concepts_from_text(
            file_data["filename"], content, folder_name, folder_ctx,
        )

    if not concepts:
        return {
            "filename": file_data["filename"],
            "folder": folder_name,
            "status": "no_concepts",
            "concepts_added": 0,
        }

    mm = memory_manager or MemoryManager(user_id)

    # Create a folder category node (but don't link every concept to it —
    # only the first few, to avoid a massive central hub)
    folder_node_id = f"_folder_{folder_name}"
    if not mm.graph.has_node(folder_node_id):
        icon = folder_ctx.get("icon", "📁")
        mm.graph.add_node(NodeData(
            id=folder_node_id,
            label=f"{icon} {folder_name.title()}",
            category="event",
            activation=0.2,
            importance=0.3,
            memory_type="ltm",
            metadata={"is_folder": True, "folder_name": folder_name},
        ))

    # Convert and activate concepts
    concept_dicts = []
    for c in concepts:
        concept_dicts.append({
            "label": c.get("label", "unknown"),
            "category": c.get("category", folder_ctx.get("default_category", "concept")),
            "relationships": [
                {
                    "target_label": r.get("target_label", ""),
                    "relationship": r.get("relationship", "related_to"),
                }
                for r in c.get("relationships", [])
                if r.get("target_label")
            ],
        })

    activated_ids, diff = mm.activate_concepts(concept_dicts)

    # Cross-link: store source filename in each activated node's metadata
    source_filename = file_data["filename"]
    for node_id in activated_ids:
        node = mm.graph.get_node(node_id)
        if node and not node.metadata.get("is_folder"):
            sources = list(node.metadata.get("source_files", []))
            if source_filename not in sources:
                sources.append(source_filename)
                mm.graph.update_node(node_id, metadata={**node.metadata, "source_files": sources})

    # Only link the top 3 most important new concepts to the folder node
    # (avoids creating a massive central hub)
    new_concepts_for_folder = [nid for nid in activated_ids if nid != folder_node_id][:3]
    for node_id in new_concepts_for_folder:
        if not mm.graph.get_edge(node_id, folder_node_id):
            mm.graph.add_edge(EdgeData(
                source=node_id,
                target=folder_node_id,
                weight=0.15,
                relationship="part_of",
            ))

    mm.save()

    return {
        "filename": file_data["filename"],
        "folder": folder_name,
        "status": "ingested",
        "concepts_added": len(activated_ids),
        "concept_labels": [c.get("label") for c in concepts],
        "graph_diff": diff,
    }


def ingest_all_files(inputs_dir: Path, user_id: str) -> dict[str, Any]:
    """Ingest all files from inputs/ and its subfolders.

    Folder structure determines the extraction context:
        inputs/
        ├── documents/   → document-focused extraction
        ├── people/      → person-focused extraction
        ├── conversations/ → conversation parsing
        └── file.txt     → generic extraction (root level)
    """
    if not inputs_dir.exists():
        inputs_dir.mkdir(parents=True, exist_ok=True)
        return {"status": "empty", "files_processed": 0, "results": []}

    taxonomy = _load_taxonomy(inputs_dir)
    mm = MemoryManager(user_id)

    results = []
    total_concepts = 0
    folders_processed = []

    # Process root-level files
    root_files = [f for f in inputs_dir.iterdir() if f.is_file() and not f.name.startswith(".") and not f.name.startswith("~$")]
    if root_files:
        root_ctx = {
            "description": "general",
            "extraction_hint": taxonomy.get("root_files_hint", "Extract all meaningful concepts."),
            "default_category": "concept",
            "icon": "📁",
        }
        for file_path in sorted(root_files):
            try:
                result = ingest_file(file_path, user_id, "general", root_ctx, mm)
                results.append(result)
                total_concepts += result.get("concepts_added", 0)
            except Exception as e:
                logger.error("Error ingesting %s: %s", file_path.name, e, exc_info=True)
                results.append({
                    "filename": file_path.name, "folder": "root",
                    "status": "error", "error": str(e), "concepts_added": 0,
                })

    # Process subfolders
    for subfolder in sorted(inputs_dir.iterdir()):
        if not subfolder.is_dir() or subfolder.name.startswith("."):
            continue

        folder_name = subfolder.name
        folder_ctx = _folder_context(folder_name, taxonomy)
        folders_processed.append(folder_name)

        # Get all files recursively
        files = [f for f in subfolder.rglob("*") if f.is_file() and not f.name.startswith(".") and not f.name.startswith("~$")]

        for file_path in sorted(files):
            try:
                result = ingest_file(file_path, user_id, folder_name, folder_ctx, mm)
                results.append(result)
                total_concepts += result.get("concepts_added", 0)
            except Exception as e:
                logger.error("Error ingesting %s: %s", file_path.name, e, exc_info=True)
                results.append({
                    "filename": file_path.name, "folder": folder_name,
                    "status": "error", "error": str(e), "concepts_added": 0,
                })

    final_snapshot = mm.get_full_snapshot()

    # Save timeline snapshot after ingestion
    try:
        from ..graph_kb.timeline import save_snapshot
        save_snapshot(user_id, "ingestion", {
            "files": len(results),
            "concepts": total_concepts,
            "nodes": mm.graph.node_count,
            "edges": mm.graph.edge_count,
        })
    except Exception as e:
        logger.warning("Timeline snapshot failed: %s", e)

    return {
        "status": "completed",
        "files_processed": len(results),
        "total_concepts_added": total_concepts,
        "folders_processed": folders_processed,
        "results": [{k: v for k, v in r.items() if k != "graph_diff"} for r in results],
        "graph_snapshot": final_snapshot.model_dump(mode="json"),
        "stats": mm.get_stats(),
    }


# ------------------------------------------------------------------
# Conversation auto-save
# ------------------------------------------------------------------

def save_conversation(
    user_id: str, messages: list[dict], inputs_dir: Path | None = None
) -> Path:
    """Save a conversation to the conversations/ folder for future ingestion.

    This creates a persistent record that can be re-ingested to enrich the graph.
    """
    base_dir = inputs_dir or (settings.data_dir.parent / "inputs")
    conv_dir = base_dir / "conversations"
    conv_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"chat_{user_id}_{timestamp}.md"
    filepath = conv_dir / filename

    lines = [f"# Conversation — {user_id}", f"Date: {timestamp}\n"]
    for msg in messages:
        role = msg.get("role", "unknown").upper()
        content = msg.get("content", "")
        lines.append(f"**{role}:** {content}\n")

    filepath.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Saved conversation to %s", filepath)
    return filepath
