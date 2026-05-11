"""
SchemaContextBuilder - stable public API for Step 2 retrieval.

The implementation now delegates to RetrievalPipeline so the retrieval stages
remain explicit and observable while LangGraph-facing code can keep calling
builder.retrieve(question).
"""

from __future__ import annotations

from typing import Any

from my_agent.core.config.settings import Settings, get_settings
from my_agent.core.models.schema_context import BusinessRule, SchemaContext, SQLExample, TableInfo
from my_agent.core.retrieval.context_merger import ContextMerger
from my_agent.core.retrieval.keyword_extractor import KeywordExtractor
from my_agent.core.retrieval.metadata_store import MetadataStore
from my_agent.core.retrieval.retrieval_pipeline import RetrievalPipeline, RetrievalPipelineResult
from my_agent.core.retrieval.telemetry import RetrievalTelemetry
from my_agent.core.retrieval.vanna_retriever import VannaRetriever


class SchemaContextBuilder:
    """
    Public entry point for Step 2.

    Downstream agents receive the same SchemaContext as before. Call
    retrieve_with_telemetry() when debugging retrieval quality.
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
        self._pipeline = RetrievalPipeline(
            vanna_retriever=self._vanna,
            metadata_store=self._meta,
            settings=self._settings,
        )
        self.last_telemetry: RetrievalTelemetry | None = None

    def retrieve(self, question: str) -> SchemaContext:
        """Run Step 2 and return only the SchemaContext contract."""
        return self.retrieve_with_telemetry(question).context

    def retrieve_with_telemetry(self, question: str) -> RetrievalPipelineResult:
        """Run Step 2 and return context plus retrieval telemetry."""
        result = self._pipeline.run(question)
        self.last_telemetry = result.telemetry
        return result

    # Backwards-compatible helpers used by existing tests and notebooks.

    @staticmethod
    def _extract_keywords(question: str) -> list[str]:
        return KeywordExtractor().extract(question)

    @staticmethod
    def _merge_tables(
        ddl_chunks: list[str],
        keyword_tables: list[TableInfo],
    ) -> list[TableInfo]:
        seen: set[str] = set()
        merged: list[TableInfo] = []

        for table in keyword_tables:
            key = table.name.lower()
            if key not in seen:
                seen.add(key)
                merged.append(table)

        for ddl in ddl_chunks:
            table_name = ContextMerger.extract_table_name_from_ddl(ddl)
            if table_name and table_name not in seen:
                seen.add(table_name)
                merged.append(
                    TableInfo(
                        name=table_name,
                        ddl=ddl,
                        description="(from semantic retrieval)",
                    )
                )

        return merged

    @staticmethod
    def _deduplicate_rules(rules: list[BusinessRule]) -> list[BusinessRule]:
        return ContextMerger.deduplicate_rules(rules)

    @staticmethod
    def _parse_sql_examples(raw: list[Any]) -> list[SQLExample]:
        return ContextMerger.parse_sql_examples(raw)
