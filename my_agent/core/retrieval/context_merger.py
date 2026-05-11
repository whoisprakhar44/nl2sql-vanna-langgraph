"""Context assembly helpers for the retrieval pipeline."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from my_agent.core.models.schema_context import BusinessRule, SchemaContext, SQLExample, TableInfo
from my_agent.core.retrieval.metadata_store import MetadataStore
from my_agent.core.retrieval.ranking import RankedTable, TableCandidate

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CandidateMergeResult:
    """Merged table candidates plus stage-level table names for telemetry."""

    candidates: list[TableCandidate]
    semantic_tables: list[str]
    keyword_tables: list[str]
    expanded_tables: list[str]


class ContextMerger:
    """Merge retrieval outputs and assemble the final SchemaContext."""

    def __init__(self, metadata_store: MetadataStore) -> None:
        self._meta = metadata_store

    def collect_table_candidates(
        self,
        ddl_chunks: list[str],
        keyword_tables: list[TableInfo],
        expanded_names: list[str],
        seed_names: list[str],
    ) -> CandidateMergeResult:
        """
        Convert semantic, keyword, and relationship signals into candidates.

        Keyword metadata wins when the same table appears in multiple sources,
        because YAML carries richer descriptions than raw DDL chunks.
        """
        candidates_by_name: dict[str, TableCandidate] = {}
        semantic_tables: list[str] = []
        keyword_table_names: list[str] = []
        total_ddl = max(len(ddl_chunks), 1)

        for index, ddl in enumerate(ddl_chunks):
            table_name = self.extract_table_name_from_ddl(ddl)
            if not table_name:
                continue

            semantic_tables.append(table_name)
            table = self._meta.find_table(table_name) or TableInfo(
                name=table_name,
                ddl=ddl,
                description="(from semantic retrieval)",
            )
            candidate = self._get_or_create_candidate(candidates_by_name, table)
            candidate.semantic_score = max(
                candidate.semantic_score,
                1.0 - (index / total_ddl),
            )
            candidate.sources.add("semantic")

        for table in keyword_tables:
            keyword_table_names.append(table.name)
            candidate = self._get_or_create_candidate(candidates_by_name, table)
            candidate.table = table
            candidate.keyword_score = max(candidate.keyword_score, 1.0)
            candidate.sources.add("keyword")

        seed_set = {name.lower() for name in seed_names}
        expanded_table_names: list[str] = []
        for table_name in expanded_names:
            name = table_name.lower()
            if name in seed_set:
                continue

            expanded_table_names.append(name)
            table = self._meta.find_table(name)
            if not table:
                continue

            candidate = self._get_or_create_candidate(candidates_by_name, table)
            candidate.relationship_score = max(candidate.relationship_score, 1.0)
            candidate.sources.add("relationship")

        return CandidateMergeResult(
            candidates=list(candidates_by_name.values()),
            semantic_tables=self._unique(semantic_tables),
            keyword_tables=self._unique(keyword_table_names),
            expanded_tables=self._unique(expanded_table_names),
        )

    def build_context(
        self,
        ranked_tables: list[RankedTable],
        keyword_rules: list[BusinessRule],
        doc_chunks: list[str],
        sql_examples_raw: list[Any],
    ) -> SchemaContext:
        """Assemble a SchemaContext from reranked tables and supporting metadata."""
        tables = [ranked.table for ranked in ranked_tables]
        table_names = [table.name for table in tables]

        columns = []
        for name in table_names:
            columns.extend(self._meta.find_columns(name))

        business_rules = []
        business_rules.extend(keyword_rules)
        for doc in doc_chunks:
            business_rules.append(
                BusinessRule(term="(retrieved documentation)", definition=doc)
            )
        business_rules = self.deduplicate_rules(business_rules)

        return SchemaContext(
            tables=tables,
            columns=columns,
            relationships=self._meta.find_relationships(table_names),
            business_rules=business_rules,
            time_columns=self._meta.find_time_columns(table_names),
            filters=self._meta.find_filters(table_names),
            examples=self.parse_sql_examples(sql_examples_raw),
        )

    @staticmethod
    def extract_table_name_from_ddl(ddl: str) -> str | None:
        """Extract a table name from a CREATE TABLE statement."""
        match = re.search(
            r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?[`\"\[]?([\w.]+)",
            ddl,
            re.IGNORECASE,
        )
        if not match:
            return None
        return match.group(1).split(".")[-1].strip("`\"[]").lower()

    @staticmethod
    def deduplicate_rules(rules: list[BusinessRule]) -> list[BusinessRule]:
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
    def parse_sql_examples(raw: list[Any]) -> list[SQLExample]:
        """Parse Vanna's get_similar_question_sql output into typed models."""
        examples: list[SQLExample] = []
        for item in raw:
            try:
                if isinstance(item, dict):
                    question = item.get("question", "")
                    sql = item.get("sql", "")
                elif isinstance(item, (list, tuple)) and len(item) >= 2:
                    question, sql = str(item[0]), str(item[1])
                else:
                    continue

                if question and sql:
                    examples.append(SQLExample(question=question, sql=sql))
            except Exception:
                logger.debug("Skipping unparseable SQL example: %s", item)

        return examples

    @staticmethod
    def _get_or_create_candidate(
        candidates_by_name: dict[str, TableCandidate],
        table: TableInfo,
    ) -> TableCandidate:
        key = table.name.lower()
        candidate = candidates_by_name.get(key)
        if candidate is None:
            candidate = TableCandidate(table=table)
            candidates_by_name[key] = candidate
        return candidate

    @staticmethod
    def _unique(items: list[str]) -> list[str]:
        seen: set[str] = set()
        unique: list[str] = []
        for item in items:
            key = item.lower()
            if key not in seen:
                seen.add(key)
                unique.append(key)
        return unique
