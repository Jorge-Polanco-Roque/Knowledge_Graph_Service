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

    # RAG / Document Store
    rag_chunk_size: int = 600
    rag_chunk_overlap: int = 100
    rag_top_k: int = 5
    rag_embedding_model: str = "text-embedding-3-small"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "env_file_override": True}


settings = Settings()
