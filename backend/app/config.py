import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

# Force .env to override system env vars
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)


class Settings(BaseSettings):
    # OpenAI
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"

    # Server
    host: str = "0.0.0.0"
    port: int = 8899
    cors_origins: list[str] = ["http://localhost:3477"]

    # Data
    data_dir: Path = Path(__file__).resolve().parent.parent / "data"

    # Plasticity parameters
    hebbian_learning_rate: float = 0.1
    ltp_rate: float = 0.05
    ltd_decay_rate: float = 0.01
    pruning_threshold: float = 0.05
    stm_capacity: int = 7
    stm_decay_rate: float = 0.3
    consolidation_threshold: float = 0.7
    spreading_factor: float = 0.5

    # Clusters
    max_clusters: int = 7

    # Query routing
    routing_model: str = "gpt-4o-mini"

    # Context mode: "chunks_only" (Mode D) or "graph_and_chunks" (Mode A)
    # chunks_only: only RAG document chunks, no graph context (more accurate, less noise)
    # graph_and_chunks: graph context + RAG chunks (richer connections, but may infer from labels)
    context_mode: str = "chunks_only"

    # Performance limits
    max_context_concepts: int = 20
    max_context_relationships: int = 30
    max_edges_per_node: int = 30

    # RAG / Document Store
    rag_chunk_size: int = 800
    rag_chunk_overlap: int = 150
    rag_top_k: int = 7
    rag_embedding_model: str = "text-embedding-3-small"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "env_file_override": True}


settings = Settings()
