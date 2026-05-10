"""
VannaRetriever — thin wrapper around Vanna 0.7.x that exposes ONLY
semantic retrieval methods.

Design decisions
────────────────
1.  We subclass ChromaDB_VectorStore + Ollama from Vanna so that its
    internal `train()` and `get_related_*` methods work correctly with
    a local ChromaDB persistent store and Ollama embeddings.

2.  We deliberately override `generate_sql` to raise an error.  Vanna
    must NEVER be used for SQL generation in this architecture — that
    responsibility belongs to the LangGraph reasoning agents.

3.  The public API of this class is intentionally narrow:
        - retrieve_ddl(question, n)
        - retrieve_documentation(question, n)
        - retrieve_sql_examples(question, n)
        - train_ddl(ddl)
        - train_documentation(doc)
        - train_sql(question, sql)

4.  This class is the ONLY place in the codebase that imports from
    `vanna`.  If Vanna is later replaced with BM25 or a schema graph,
    only this file needs to change.
"""

from __future__ import annotations

import logging
from typing import Any

from vanna.chromadb import ChromaDB_VectorStore
from vanna.ollama import Ollama

from src.config.settings import Settings, get_settings

logger = logging.getLogger(__name__)


class VannaRetriever(ChromaDB_VectorStore, Ollama):
    """
    Retrieval-only Vanna instance backed by ChromaDB + Ollama embeddings.

    This class inherits Vanna's vector-store machinery for training and
    retrieval but intentionally blocks all SQL generation methods.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        settings = settings or get_settings()

        config = {
            # ChromaDB persistence
            "path": str(settings.chroma_path),
            # Ollama connection (used only for embedding, not SQL gen)
            "ollama_host": settings.ollama_base_url,
            "model": settings.ollama_model,
        }

        ChromaDB_VectorStore.__init__(self, config=config)
        Ollama.__init__(self, config=config)

        logger.info(
            "VannaRetriever initialised — chroma_path=%s, ollama=%s",
            settings.chroma_path,
            settings.ollama_base_url,
        )

    # ── Blocked methods (Vanna must NOT generate SQL) ─────────────────────

    def generate_sql(self, *args: Any, **kwargs: Any) -> str:
        """Blocked. SQL generation is owned by LangGraph, not Vanna."""
        raise NotImplementedError(
            "VannaRetriever is retrieval-only. "
            "SQL generation must go through the LangGraph reasoning agents."
        )

    def ask(self, *args: Any, **kwargs: Any) -> Any:
        """Blocked. The ask() convenience method would trigger SQL gen."""
        raise NotImplementedError(
            "VannaRetriever.ask() is disabled. "
            "Use retrieve_ddl / retrieve_documentation instead."
        )

    # ── Public retrieval API ──────────────────────────────────────────────

    def retrieve_ddl(self, question: str, n: int | None = None) -> list[str]:
        """
        Semantic search over trained DDL chunks.

        Returns up to `n` DDL strings most relevant to the question.
        """
        n = n or get_settings().retrieval_ddl_limit
        try:
            results = self.get_related_ddl(question=question, n_results=n)
        except Exception:
            logger.exception("DDL retrieval failed for question: %s", question)
            results = []
        logger.debug("Retrieved %d DDL chunks for: %s", len(results), question)
        return results

    def retrieve_documentation(self, question: str, n: int | None = None) -> list[str]:
        """
        Semantic search over trained documentation / business rules.

        Returns up to `n` documentation strings most relevant to the question.
        """
        n = n or get_settings().retrieval_doc_limit
        try:
            results = self.get_related_documentation(question=question, n_results=n)
        except Exception:
            logger.exception("Documentation retrieval failed for question: %s", question)
            results = []
        logger.debug("Retrieved %d doc chunks for: %s", len(results), question)
        return results

    def retrieve_sql_examples(self, question: str, n: int | None = None) -> list[dict[str, str]]:
        """
        Semantic search over trained question → SQL pairs.

        Returns up to `n` example dicts: [{"question": ..., "sql": ...}].
        Kept deliberately low (default 2) to avoid template-matching.
        """
        n = n or get_settings().retrieval_sql_limit
        try:
            results = self.get_similar_question_sql(question=question, n_results=n)
        except Exception:
            logger.exception("SQL example retrieval failed for question: %s", question)
            results = []
        logger.debug("Retrieved %d SQL examples for: %s", len(results), question)
        return results

    # ── Training helpers (thin pass-through) ──────────────────────────────

    def train_ddl(self, ddl: str) -> str:
        """Add a DDL statement to the ChromaDB DDL collection."""
        result = self.train(ddl=ddl)
        logger.info("Trained DDL chunk (%d chars)", len(ddl))
        return result

    def train_documentation(self, doc: str) -> str:
        """Add a documentation / business-rule string to ChromaDB."""
        result = self.train(documentation=doc)
        logger.info("Trained documentation chunk (%d chars)", len(doc))
        return result

    def train_sql(self, question: str, sql: str) -> str:
        """Add a question → SQL example pair to ChromaDB (use sparingly)."""
        result = self.train(question=question, sql=sql)
        logger.info("Trained SQL example: %s", question[:80])
        return result
