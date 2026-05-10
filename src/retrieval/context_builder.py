"""
SchemaContextBuilder — the public API of Step 2.

This is the ONLY class that downstream code (LangGraph agents) should
interact with.  It orchestrates:

  1. Semantic retrieval   → VannaRetriever  (ChromaDB embeddings)
  2. Keyword retrieval    → MetadataStore   (YAML exact-match)
  3. Relationship expansion  → MetadataStore FK traversal
  4. Context assembly     → SchemaContext Pydantic model

Usage (from a LangGraph node or tool):
──────────────────────────────────────

    from src.retrieval import SchemaContextBuilder

    builder = SchemaContextBuilder()
    ctx = builder.retrieve("Show me revenue by region vs last month")
    # ctx is a SchemaContext Pydantic model
    # ctx.model_dump() → JSON dict for LangGraph state
    # ctx.to_prompt_str() → compact string for LLM prompt injection
"""

from __future__ import annotations

import logging
import re
from typing import Any

from src.config.settings import Settings, get_settings
from src.models.schema_context import (
    BusinessRule,
    ColumnInfo,
    FilterInfo,
    RelationshipInfo,
    SchemaContext,
    SQLExample,
    TableInfo,
    TimeColumnInfo,
)
from src.retrieval.metadata_store import MetadataStore
from src.retrieval.vanna_retriever import VannaRetriever

logger = logging.getLogger(__name__)


class SchemaContextBuilder:
    """
    Orchestrates hybrid retrieval to produce a SchemaContext.

    This class is the single entry-point for Step 2.  It combines
    semantic (Vanna/ChromaDB) and keyword (YAML metadata) retrieval
    strategies, then merges and deduplicates results into a clean,
    typed SchemaContext object.
    """

    def __init__(
        self,
        vanna_retriever: VannaRetriever | None = None,
        metadata_store: MetadataStore | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._vanna = vanna_retriever or VannaRetriever(self._settings)
        self._meta = metadata_store or MetadataStore(self._settings)

    # ── Public API ────────────────────────────────────────────────────────

    def retrieve(self, question: str) -> SchemaContext:
        """
        Run the full hybrid retrieval pipeline for a natural-language question.

        Returns a SchemaContext containing the top relevant schema entities,
        relationships, business rules, and (limited) SQL examples.

        This is the ONLY method LangGraph agents need to call.
        """
        logger.info("Step 2 retrieval started for: %s", question[:120])

        # Extract keywords from the question for keyword-based lookup.
        keywords = self._extract_keywords(question)
        logger.debug("Extracted keywords: %s", keywords)

        # ── 1. Semantic retrieval via Vanna ───────────────────────────────
        ddl_chunks = self._vanna.retrieve_ddl(question)
        doc_chunks = self._vanna.retrieve_documentation(question)
        sql_examples_raw = self._vanna.retrieve_sql_examples(question)

        # ── 2. Keyword retrieval via MetadataStore ────────────────────────
        keyword_tables = self._meta.find_tables(keywords)
        keyword_rules = self._meta.find_business_rules(keywords)

        # ── 3. Merge tables from both sources ─────────────────────────────
        tables = self._merge_tables(ddl_chunks, keyword_tables)
        table_names = [t.name for t in tables]

        # ── 4. Relationship expansion ─────────────────────────────────────
        expanded_names = self._meta.expand_related_tables(table_names)

        # Add any newly discovered tables from expansion.
        for name in expanded_names:
            if name not in table_names:
                extra_tables = self._meta.find_tables([name])
                tables.extend(extra_tables)
                table_names.append(name)

        # ── 5. Gather structured metadata for discovered tables ───────────
        columns: list[ColumnInfo] = []
        for name in table_names:
            columns.extend(self._meta.find_columns(name))

        relationships = self._meta.find_relationships(table_names)
        time_columns = self._meta.find_time_columns(table_names)
        filters = self._meta.find_filters(table_names)

        # ── 6. Business rules (merge semantic docs + keyword glossary) ────
        business_rules = list(keyword_rules)
        for doc in doc_chunks:
            # Each doc chunk is a raw string from Vanna's documentation
            # collection.  Wrap it as a business rule.
            business_rules.append(
                BusinessRule(term="(retrieved documentation)", definition=doc)
            )
        business_rules = self._deduplicate_rules(business_rules)

        # ── 7. SQL examples (kept minimal) ────────────────────────────────
        examples = self._parse_sql_examples(sql_examples_raw)

        # ── 8. Assemble final context ─────────────────────────────────────
        ctx = SchemaContext(
            tables=tables,
            columns=columns,
            relationships=relationships,
            business_rules=business_rules,
            time_columns=time_columns,
            filters=filters,
            examples=examples,
        )

        logger.info(
            "Step 2 retrieval complete — %d tables, %d columns, "
            "%d relationships, %d rules, %d examples",
            len(ctx.tables),
            len(ctx.columns),
            len(ctx.relationships),
            len(ctx.business_rules),
            len(ctx.examples),
        )

        return ctx

    # ── Internal helpers ──────────────────────────────────────────────────

    @staticmethod
    def _extract_keywords(question: str) -> list[str]:
        """
        Naive keyword extraction from the natural-language question.

        Strips common stop-words and returns lowercase tokens.
        This is intentionally simple — the LLM agents do the real
        understanding later.
        """
        stop_words = {
            "show", "me", "the", "a", "an", "and", "or", "of", "in", "by",
            "for", "to", "from", "with", "is", "are", "was", "were", "what",
            "how", "many", "much", "do", "does", "did", "can", "could",
            "would", "should", "will", "i", "my", "our", "their", "vs",
            "versus", "compared", "last", "this", "all", "each", "every",
            "give", "get", "find", "list", "display", "tell", "please",
            "want", "need", "see", "look", "at", "up", "on", "it", "its",
        }
        # Tokenise: split on non-alphanumeric, keep underscores.
        tokens = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", question.lower())
        return [t for t in tokens if t not in stop_words and len(t) > 1]

    @staticmethod
    def _merge_tables(
        ddl_chunks: list[str],
        keyword_tables: list[TableInfo],
    ) -> list[TableInfo]:
        """
        Merge tables discovered via semantic DDL retrieval with those
        found via keyword metadata lookup.  Deduplicate by table name.
        """
        seen: set[str] = set()
        merged: list[TableInfo] = []

        # Keyword tables are higher quality (they have full metadata),
        # so they get priority.
        for t in keyword_tables:
            key = t.name.lower()
            if key not in seen:
                seen.add(key)
                merged.append(t)

        # DDL chunks from Vanna may contain table names we haven't seen.
        for ddl in ddl_chunks:
            # Try to extract the table name from the DDL.
            match = re.search(r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(\w+)", ddl, re.IGNORECASE)
            if match:
                name = match.group(1).lower()
                if name not in seen:
                    seen.add(name)
                    merged.append(
                        TableInfo(name=name, ddl=ddl, description="(from semantic retrieval)")
                    )

        return merged

    @staticmethod
    def _deduplicate_rules(rules: list[BusinessRule]) -> list[BusinessRule]:
        """Remove duplicate business rules by (term, definition) pair."""
        seen: set[tuple[str, str]] = set()
        unique: list[BusinessRule] = []
        for rule in rules:
            key = (rule.term.lower(), rule.definition.lower())
            if key not in seen:
                seen.add(key)
                unique.append(rule)
        return unique

    @staticmethod
    def _parse_sql_examples(raw: list[Any]) -> list[SQLExample]:
        """
        Parse Vanna's get_similar_question_sql output into typed models.

        Vanna returns a list of dicts or tuples depending on version.
        We handle both formats defensively.
        """
        examples: list[SQLExample] = []
        for item in raw:
            try:
                if isinstance(item, dict):
                    q = item.get("question", "")
                    s = item.get("sql", "")
                elif isinstance(item, (list, tuple)) and len(item) >= 2:
                    q, s = str(item[0]), str(item[1])
                else:
                    continue

                if q and s:
                    examples.append(SQLExample(question=q, sql=s))
            except Exception:
                logger.debug("Skipping unparseable SQL example: %s", item)

        return examples
