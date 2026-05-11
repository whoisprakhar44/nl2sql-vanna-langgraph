"""Stage-based retrieval pipeline for Step 2."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from src.config.settings import Settings, get_settings
from src.models.schema_context import SchemaContext
from src.retrieval.context_merger import ContextMerger
from src.retrieval.keyword_extractor import KeywordExtractor
from src.retrieval.metadata_store import MetadataStore
from src.retrieval.ranking import TableReranker
from src.retrieval.synonym_mapper import SynonymMapper
from src.retrieval.telemetry import RetrievalTelemetry

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievalPipelineResult:
    """The final context plus structured telemetry for debugging."""

    context: SchemaContext
    telemetry: RetrievalTelemetry


class RetrievalPipeline:
    """
    Run retrieval as explicit stages:

    semantic -> keyword -> FK expansion -> rerank -> context build
    """

    def __init__(
        self,
        vanna_retriever: Any,
        metadata_store: MetadataStore,
        settings: Settings | None = None,
        keyword_extractor: KeywordExtractor | None = None,
        reranker: TableReranker | None = None,
        context_merger: ContextMerger | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._vanna = vanna_retriever
        self._meta = metadata_store
        self._keyword_extractor = keyword_extractor or KeywordExtractor(
            SynonymMapper(metadata_store.get_synonym_groups())
        )
        self._reranker = reranker or TableReranker(self._settings)
        self._context_merger = context_merger or ContextMerger(metadata_store)

    def run(self, question: str) -> RetrievalPipelineResult:
        """Run all retrieval stages for one natural-language question."""
        logger.info("Step 2 retrieval started for: %s", question[:120])

        keywords = self._keyword_extractor.extract(question)
        logger.debug("Extracted keywords: %s", keywords)

        ddl_chunks = self._vanna.retrieve_ddl(question)
        doc_chunks = self._vanna.retrieve_documentation(question)

        sql_examples_enabled = (
            self._settings.retrieval_include_sql_examples
            and self._settings.retrieval_sql_limit > 0
        )
        sql_examples_raw = []
        if sql_examples_enabled:
            sql_examples_raw = self._vanna.retrieve_sql_examples(
                question,
                n=self._settings.retrieval_sql_limit,
            )

        keyword_tables = self._meta.find_tables(keywords)
        keyword_rules = self._meta.find_business_rules(keywords)

        semantic_seed_names = [
            table_name
            for table_name in (
                self._context_merger.extract_table_name_from_ddl(ddl)
                for ddl in ddl_chunks
            )
            if table_name
        ]
        keyword_seed_names = [table.name for table in keyword_tables]
        seed_names = self._unique([*semantic_seed_names, *keyword_seed_names])

        expanded_names = self._meta.schema_graph.expand_related_tables(
            seed_names,
            max_hops=self._settings.relationship_expansion_hops,
        )

        merge_result = self._context_merger.collect_table_candidates(
            ddl_chunks=ddl_chunks,
            keyword_tables=keyword_tables,
            expanded_names=expanded_names,
            seed_names=seed_names,
        )
        ranked_tables = self._reranker.rank(merge_result.candidates)

        context = self._context_merger.build_context(
            ranked_tables=ranked_tables,
            keyword_rules=keyword_rules,
            doc_chunks=doc_chunks,
            sql_examples_raw=sql_examples_raw,
        )

        telemetry = RetrievalTelemetry(
            question=question,
            keywords=keywords,
            semantic_tables=merge_result.semantic_tables,
            keyword_tables=merge_result.keyword_tables,
            expanded_tables=merge_result.expanded_tables,
            final_ranked_tables=ranked_tables,
            documentation_chunks=len(doc_chunks),
            business_rules=len(context.business_rules),
            sql_examples_enabled=sql_examples_enabled,
            sql_examples=len(context.examples),
        )

        logger.info(
            "Retrieval telemetry: %s",
            json.dumps(telemetry.to_log_dict(), sort_keys=True),
        )
        logger.info(
            "Step 2 retrieval complete - %d tables, %d columns, "
            "%d relationships, %d rules, %d examples",
            len(context.tables),
            len(context.columns),
            len(context.relationships),
            len(context.business_rules),
            len(context.examples),
        )

        return RetrievalPipelineResult(context=context, telemetry=telemetry)

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
