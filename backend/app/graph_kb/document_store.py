"""ChromaDB-backed document chunk storage with semantic search."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

import chromadb
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..config import settings

logger = logging.getLogger(__name__)

# Module-level cache for DocumentStore instances (one per user_id)
_document_stores: dict[str, "DocumentStore"] = {}


def get_document_store(user_id: str) -> DocumentStore:
    """Get or create a DocumentStore for the given user."""
    if user_id not in _document_stores:
        _document_stores[user_id] = DocumentStore(user_id, settings.data_dir)
    return _document_stores[user_id]


class DocumentStore:
    """ChromaDB-backed document chunk storage with semantic search."""

    def __init__(self, user_id: str, data_dir: Path) -> None:
        self._user_id = user_id
        chroma_path = data_dir / "users" / user_id / "chroma"
        chroma_path.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(path=str(chroma_path))
        self._collection = self._client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"},
        )
        self._embeddings = OpenAIEmbeddings(
            model=settings.rag_embedding_model,
            api_key=settings.openai_api_key,
        )
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.rag_chunk_size,
            chunk_overlap=settings.rag_chunk_overlap,
            separators=["\n\n", "\n", ". ", " "],
        )

    def ingest(self, content: str, metadata: dict[str, Any]) -> int:
        """Chunk, embed, and store a document. Returns number of chunks stored."""
        source_file = metadata.get("source_file", "unknown")

        # Delete existing chunks for this source (idempotent re-ingestion)
        self.delete_by_source(source_file)

        # Split into chunks
        chunks = self._splitter.split_text(content)
        if not chunks:
            return 0

        # Generate deterministic IDs based on source + chunk index
        ids = [
            hashlib.md5(f"{self._user_id}:{source_file}:{i}".encode()).hexdigest()
            for i in range(len(chunks))
        ]

        # Build per-chunk metadata
        metadatas = [
            {
                **metadata,
                "chunk_index": i,
                "total_chunks": len(chunks),
            }
            for i in range(len(chunks))
        ]

        # Embed all chunks in one batch call
        embeddings = self._embeddings.embed_documents(chunks)

        # Upsert into ChromaDB
        self._collection.upsert(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        logger.info(
            "Stored %d chunks for %s (user: %s)", len(chunks), source_file, self._user_id
        )
        return len(chunks)

    def ingest_summary(self, summary: str, metadata: dict[str, Any]) -> None:
        """Store a document summary as a special chunk (auto-generated during ingestion)."""
        source_file = metadata.get("source_file", "unknown")
        chunk_id = hashlib.md5(
            f"{self._user_id}:summary:{source_file}".encode()
        ).hexdigest()

        embedding = self._embeddings.embed_documents([summary])[0]

        self._collection.upsert(
            ids=[chunk_id],
            documents=[summary],
            embeddings=[embedding],
            metadatas=[{**metadata, "chunk_type": "summary"}],
        )

    def search(
        self, query: str, top_k: int | None = None, where: dict | None = None
    ) -> list[dict[str, Any]]:
        """Semantic search. Returns chunks with content, metadata, and distance."""
        if self._collection.count() == 0:
            return []

        top_k = top_k or settings.rag_top_k

        query_embedding = self._embeddings.embed_query(query)

        kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": min(top_k, self._collection.count()),
        }
        if where:
            kwargs["where"] = where

        results = self._collection.query(**kwargs)

        # Flatten ChromaDB's nested list format
        chunks = []
        for i in range(len(results["ids"][0])):
            chunks.append({
                "id": results["ids"][0][i],
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            })

        return chunks

    def search_by_sources(
        self, query: str, source_files: list[str], top_k: int = 3
    ) -> list[dict[str, Any]]:
        """Search within specific source files only (graph-guided retrieval)."""
        if not source_files or self._collection.count() == 0:
            return []
        return self.search(query, top_k=top_k, where={"source_file": {"$in": source_files}})

    def delete_by_source(self, source_file: str) -> None:
        """Remove all chunks from a specific source file (for re-ingestion)."""
        try:
            self._collection.delete(where={"source_file": source_file})
        except Exception as e:
            logger.debug("delete_by_source(%s): %s", source_file, e)
