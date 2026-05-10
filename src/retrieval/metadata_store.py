"""
MetadataStore — keyword-based retrieval from YAML metadata files.

This complements VannaRetriever's semantic search with exact-match
keyword lookups for:

  - Table descriptions and column details
  - Foreign-key relationships
  - Business glossary terms
  - Enum / categorical values
  - Date/time column semantics

────────────────────────────────────────────────────────────────
YAML file format (one file per table in data/metadata/):
────────────────────────────────────────────────────────────────

  # data/metadata/orders.yaml
  table: orders
  description: "Customer purchase orders"
  columns:
    - name: id
      type: INTEGER
      description: "Primary key"
    - name: customer_id
      type: INTEGER
      description: "FK to customers.id"
    - name: order_date
      type: DATE
      description: "When the order was placed"
      time_granularity: day
      time_format: "YYYY-MM-DD"
    - name: status
      type: VARCHAR(20)
      description: "Order lifecycle status"
      enum_values: ["pending", "shipped", "delivered", "cancelled"]

  relationships:
    - from_column: customer_id
      to_table: customers
      to_column: id
      type: foreign_key

  business_rules:
    - term: "active order"
      definition: "An order with status IN ('pending', 'shipped')"

────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from src.config.settings import Settings, get_settings
from src.models.schema_context import (
    BusinessRule,
    ColumnInfo,
    FilterInfo,
    RelationshipInfo,
    TableInfo,
    TimeColumnInfo,
)

logger = logging.getLogger(__name__)


class MetadataStore:
    """
    Loads YAML metadata files and provides keyword-based lookups.

    This is the non-embedding side of hybrid retrieval.  It returns
    precise, structured metadata when the question mentions known
    table names, column names, or business terms.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        settings = settings or get_settings()
        self._metadata_dir: Path = settings.metadata_path
        self._tables: dict[str, dict[str, Any]] = {}
        self._glossary: list[BusinessRule] = []
        self._load_all()

    # ── Loading ───────────────────────────────────────────────────────────

    def _load_all(self) -> None:
        """Scan the metadata directory and parse every YAML file."""
        if not self._metadata_dir.exists():
            logger.warning("Metadata directory does not exist: %s", self._metadata_dir)
            return

        yaml_files = list(self._metadata_dir.glob("*.yaml")) + list(
            self._metadata_dir.glob("*.yml")
        )

        for fpath in sorted(yaml_files):
            try:
                data = yaml.safe_load(fpath.read_text(encoding="utf-8"))
                if not data or "table" not in data:
                    logger.warning("Skipping malformed metadata file: %s", fpath.name)
                    continue

                table_name = data["table"].lower()
                self._tables[table_name] = data

                # Collect business rules into a flat glossary.
                for rule in data.get("business_rules", []):
                    self._glossary.append(
                        BusinessRule(term=rule["term"], definition=rule["definition"])
                    )

                logger.debug("Loaded metadata for table: %s", table_name)

            except Exception:
                logger.exception("Failed to parse metadata file: %s", fpath.name)

        logger.info(
            "MetadataStore loaded %d tables, %d glossary terms",
            len(self._tables),
            len(self._glossary),
        )

    # ── Keyword search ────────────────────────────────────────────────────

    def find_tables(self, keywords: list[str]) -> list[TableInfo]:
        """
        Return TableInfo for any table whose name appears in the keywords.

        Also matches against column names and descriptions for broader recall.
        """
        matched: list[TableInfo] = []
        normalised = [kw.lower() for kw in keywords]

        for table_name, data in self._tables.items():
            # Direct table name match.
            if table_name in normalised:
                matched.append(self._to_table_info(data))
                continue

            # Partial match: keyword appears in table name or description.
            desc = data.get("description", "").lower()
            if any(kw in table_name or kw in desc for kw in normalised):
                matched.append(self._to_table_info(data))
                continue

            # Column-name match: if a keyword matches a column, include the table.
            col_names = [c.get("name", "").lower() for c in data.get("columns", [])]
            if any(kw in col_names for kw in normalised):
                matched.append(self._to_table_info(data))

        return matched

    def find_columns(self, table_name: str) -> list[ColumnInfo]:
        """Return all columns for a known table."""
        data = self._tables.get(table_name.lower())
        if not data:
            return []
        return [
            ColumnInfo(
                table=table_name,
                name=c["name"],
                type=c.get("type", ""),
                description=c.get("description", ""),
            )
            for c in data.get("columns", [])
        ]

    def find_relationships(self, table_names: list[str]) -> list[RelationshipInfo]:
        """
        Return all FK / logical relationships that involve any of the
        given table names (either as source or target).
        """
        normalised = {t.lower() for t in table_names}
        rels: list[RelationshipInfo] = []

        for table_name, data in self._tables.items():
            for rel in data.get("relationships", []):
                from_table = table_name
                to_table = rel.get("to_table", "").lower()

                if from_table in normalised or to_table in normalised:
                    rels.append(
                        RelationshipInfo(
                            from_table=from_table,
                            from_column=rel["from_column"],
                            to_table=to_table,
                            to_column=rel["to_column"],
                            type=rel.get("type", "foreign_key"),
                        )
                    )

        return rels

    def find_time_columns(self, table_names: list[str]) -> list[TimeColumnInfo]:
        """Return time/date column metadata for the given tables."""
        normalised = {t.lower() for t in table_names}
        time_cols: list[TimeColumnInfo] = []

        for table_name in normalised:
            data = self._tables.get(table_name)
            if not data:
                continue
            for col in data.get("columns", []):
                if col.get("time_granularity"):
                    time_cols.append(
                        TimeColumnInfo(
                            table=table_name,
                            column=col["name"],
                            granularity=col["time_granularity"],
                            format=col.get("time_format", ""),
                        )
                    )

        return time_cols

    def find_filters(self, table_names: list[str]) -> list[FilterInfo]:
        """Return enum / categorical value metadata for the given tables."""
        normalised = {t.lower() for t in table_names}
        filters: list[FilterInfo] = []

        for table_name in normalised:
            data = self._tables.get(table_name)
            if not data:
                continue
            for col in data.get("columns", []):
                if col.get("enum_values"):
                    filters.append(
                        FilterInfo(
                            table=table_name,
                            column=col["name"],
                            values=col["enum_values"],
                        )
                    )

        return filters

    def find_business_rules(self, keywords: list[str]) -> list[BusinessRule]:
        """
        Return glossary entries whose term matches any of the keywords.

        Uses simple substring matching — fast and predictable.
        """
        normalised = [kw.lower() for kw in keywords]
        return [
            rule
            for rule in self._glossary
            if any(kw in rule.term.lower() for kw in normalised)
        ]

    def get_all_table_names(self) -> list[str]:
        """Return all table names known to the metadata store."""
        return list(self._tables.keys())

    # ── Relationship expansion ────────────────────────────────────────────

    def expand_related_tables(self, table_names: list[str]) -> list[str]:
        """
        Given a set of table names, return additional table names that
        are connected via foreign keys.  This is the 'relationship
        expansion' step of hybrid retrieval.
        """
        normalised = {t.lower() for t in table_names}
        expanded: set[str] = set(normalised)

        for table_name, data in self._tables.items():
            for rel in data.get("relationships", []):
                to_table = rel.get("to_table", "").lower()

                if table_name in normalised:
                    expanded.add(to_table)
                if to_table in normalised:
                    expanded.add(table_name)

        new_tables = expanded - normalised
        if new_tables:
            logger.debug("Relationship expansion added tables: %s", new_tables)

        return list(expanded)

    # ── Internals ─────────────────────────────────────────────────────────

    @staticmethod
    def _to_table_info(data: dict[str, Any]) -> TableInfo:
        """Convert raw YAML dict to a TableInfo model."""
        # Reconstruct a lightweight DDL from column definitions.
        cols = data.get("columns", [])
        col_lines = [f"  {c['name']} {c.get('type', 'TEXT')}" for c in cols]
        ddl = f"CREATE TABLE {data['table']} (\n" + ",\n".join(col_lines) + "\n);"

        return TableInfo(
            name=data["table"],
            ddl=ddl,
            description=data.get("description", ""),
        )
