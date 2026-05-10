"""
SchemaIngestor — trains the VannaRetriever on DDL, documentation,
and (optionally) example SQL pairs.

This is a one-time or periodic process.  You run it once when you set
up a new database, and again only when the schema changes.

Usage:
──────
    from src.training.ingest import SchemaIngestor

    ingestor = SchemaIngestor()

    # Train from DDL files in data/ddl/
    ingestor.ingest_ddl_directory()

    # Train from YAML metadata (documentation + business rules)
    ingestor.ingest_metadata_directory()

    # Train a single DDL string
    ingestor.ingest_ddl("CREATE TABLE orders (...);")

    # Train a documentation string
    ingestor.ingest_documentation("Revenue = SUM(order_total) for shipped orders only.")

    # Train an example SQL pair (use sparingly)
    ingestor.ingest_sql_example(
        question="Total revenue last month",
        sql="SELECT SUM(total) FROM orders WHERE status='shipped' AND ..."
    )

    # Auto-ingest from SQLite information schema
    ingestor.ingest_from_sqlite("./data/sample.db")
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import yaml

from src.config.settings import Settings, get_settings
from src.retrieval.vanna_retriever import VannaRetriever

logger = logging.getLogger(__name__)


class SchemaIngestor:
    """
    Manages training data ingestion into the VannaRetriever's
    ChromaDB collections.
    """

    def __init__(
        self,
        retriever: VannaRetriever | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._vanna = retriever or VannaRetriever(self._settings)

    # ── Bulk ingestion ────────────────────────────────────────────────────

    def ingest_ddl_directory(self, ddl_dir: str | Path | None = None) -> int:
        """
        Read all .sql files from a directory and train each as a DDL chunk.

        Returns the number of files successfully ingested.
        """
        ddl_path = Path(ddl_dir) if ddl_dir else Path("./data/ddl")
        if not ddl_path.exists():
            logger.warning("DDL directory does not exist: %s", ddl_path)
            return 0

        count = 0
        for fpath in sorted(ddl_path.glob("*.sql")):
            ddl = fpath.read_text(encoding="utf-8").strip()
            if ddl:
                self._vanna.train_ddl(ddl)
                count += 1
                logger.debug("Ingested DDL from: %s", fpath.name)

        logger.info("Ingested %d DDL files from %s", count, ddl_path)
        return count

    def ingest_metadata_directory(self, meta_dir: str | Path | None = None) -> int:
        """
        Read all YAML metadata files and train their documentation
        and business rules into Vanna's documentation collection.

        This does NOT duplicate the YAML data — it feeds the textual
        descriptions and rules into ChromaDB so they can be found via
        semantic search alongside the keyword-based MetadataStore.

        Returns the number of files successfully ingested.
        """
        meta_path = Path(meta_dir) if meta_dir else self._settings.metadata_path
        if not meta_path.exists():
            logger.warning("Metadata directory does not exist: %s", meta_path)
            return 0

        count = 0
        for fpath in sorted(
            list(meta_path.glob("*.yaml")) + list(meta_path.glob("*.yml"))
        ):
            try:
                data = yaml.safe_load(fpath.read_text(encoding="utf-8"))
                if not data or "table" not in data:
                    continue

                # Train table description as documentation.
                table_name = data["table"]
                desc = data.get("description", "")
                if desc:
                    self._vanna.train_documentation(
                        f"Table '{table_name}': {desc}"
                    )

                # Train column descriptions.
                for col in data.get("columns", []):
                    col_desc = col.get("description", "")
                    if col_desc:
                        self._vanna.train_documentation(
                            f"Column '{table_name}.{col['name']}' "
                            f"({col.get('type', 'TEXT')}): {col_desc}"
                        )

                # Train business rules.
                for rule in data.get("business_rules", []):
                    self._vanna.train_documentation(
                        f"Business rule — {rule['term']}: {rule['definition']}"
                    )

                # Build and train a synthetic DDL from column definitions.
                cols = data.get("columns", [])
                if cols:
                    col_defs = ", ".join(
                        f"{c['name']} {c.get('type', 'TEXT')}" for c in cols
                    )
                    ddl = f"CREATE TABLE {table_name} ({col_defs});"
                    self._vanna.train_ddl(ddl)

                count += 1
                logger.debug("Ingested metadata from: %s", fpath.name)

            except Exception:
                logger.exception("Failed to ingest metadata: %s", fpath.name)

        logger.info("Ingested %d metadata files from %s", count, meta_path)
        return count

    # ── Single-item ingestion ─────────────────────────────────────────────

    def ingest_ddl(self, ddl: str) -> None:
        """Train a single DDL statement."""
        self._vanna.train_ddl(ddl)

    def ingest_documentation(self, doc: str) -> None:
        """Train a single documentation / business-rule string."""
        self._vanna.train_documentation(doc)

    def ingest_sql_example(self, question: str, sql: str) -> None:
        """
        Train a single question → SQL pair.

        Use sparingly — the architecture avoids heavy reliance on
        example-based retrieval to prevent template-matching.
        """
        self._vanna.train_sql(question=question, sql=sql)

    # ── SQLite auto-ingestion ─────────────────────────────────────────────

    def ingest_from_sqlite(self, db_path: str | Path | None = None) -> int:
        """
        Automatically extract DDL from a SQLite database and train it.

        This reads the sqlite_master table to get CREATE TABLE statements,
        then feeds each one into Vanna's DDL collection.

        Returns the number of tables ingested.
        """
        db = Path(db_path) if db_path else self._settings.db_path
        if not db.exists():
            logger.warning("SQLite database not found: %s", db)
            return 0

        conn = sqlite3.connect(str(db))
        try:
            cursor = conn.execute(
                "SELECT name, sql FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%' "
                "ORDER BY name"
            )
            rows = cursor.fetchall()

            count = 0
            for name, ddl in rows:
                if ddl:
                    self._vanna.train_ddl(ddl)
                    count += 1
                    logger.debug("Ingested SQLite DDL for table: %s", name)

            logger.info("Ingested %d tables from SQLite: %s", count, db)
            return count

        finally:
            conn.close()
