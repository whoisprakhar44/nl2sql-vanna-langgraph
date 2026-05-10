"""
Centralised application settings.

Reads from environment variables / .env file.
All retrieval-tuning knobs live here so they can be adjusted without
touching retrieval code.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Immutable, validated application configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # -- Ollama ---------------------------------------------------------------
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"

    # -- ChromaDB -------------------------------------------------------------
    chroma_persist_dir: str = "./data/chroma_store"

    # -- Database (for future execution steps) --------------------------------
    database_path: str = "./data/sample.db"

    # -- Retrieval limits -----------------------------------------------------
    # These cap how many chunks Vanna's semantic search returns.
    # Kept deliberately low to minimise noise for downstream agents.
    retrieval_ddl_limit: int = 5
    retrieval_doc_limit: int = 3
    retrieval_sql_limit: int = 2

    # -- Metadata directory ---------------------------------------------------
    metadata_dir: str = "./data/metadata"

    # -- Logging --------------------------------------------------------------
    log_level: str = "INFO"

    # -- Helpers --------------------------------------------------------------
    @property
    def chroma_path(self) -> Path:
        return Path(self.chroma_persist_dir)

    @property
    def metadata_path(self) -> Path:
        return Path(self.metadata_dir)

    @property
    def db_path(self) -> Path:
        return Path(self.database_path)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return a cached singleton of the application settings.

    Using lru_cache ensures the .env file is parsed only once.
    """
    return Settings()
