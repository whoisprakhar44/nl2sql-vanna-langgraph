"""
Tests for the Step 2 retrieval layer.

These tests validate the MetadataStore and SchemaContext models
without requiring Ollama or ChromaDB to be running (unit tests).

For integration tests that exercise VannaRetriever, run:
    pytest tests/ -m integration
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config.settings import Settings
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
from src.retrieval.context_builder import SchemaContextBuilder
from src.retrieval.keyword_extractor import KeywordExtractor
from src.retrieval.metadata_store import MetadataStore
from src.retrieval.ranking import TableCandidate, TableReranker
from src.retrieval.schema_graph import SchemaGraph
from src.retrieval.synonym_mapper import SynonymMapper


class FakeVannaRetriever:
    """Test double that behaves like the retrieval-only Vanna wrapper."""

    def __init__(self) -> None:
        self.sql_called = False

    def retrieve_ddl(self, question: str, n: int | None = None) -> list[str]:
        return ["CREATE TABLE orders (id INT, customer_id INT, total DECIMAL);"]

    def retrieve_documentation(self, question: str, n: int | None = None) -> list[str]:
        return []

    def retrieve_sql_examples(self, question: str, n: int | None = None) -> list[dict[str, str]]:
        self.sql_called = True
        return [{"question": question, "sql": "SELECT * FROM orders"}]


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def sample_settings(tmp_path: Path) -> Settings:
    """Create settings pointing at the real sample metadata."""
    return Settings(
        metadata_dir="./data/metadata",
        chroma_persist_dir=str(tmp_path / "chroma"),
        database_path="./data/sample.db",
    )


@pytest.fixture
def metadata_store(sample_settings: Settings) -> MetadataStore:
    """Create a MetadataStore using the sample YAML metadata."""
    return MetadataStore(settings=sample_settings)


# ── SchemaContext model tests ─────────────────────────────────────────────


class TestSchemaContext:
    """Tests for the Pydantic data models."""

    def test_empty_context(self) -> None:
        """An empty SchemaContext should serialise cleanly."""
        ctx = SchemaContext()
        data = ctx.model_dump()
        assert data["tables"] == []
        assert data["columns"] == []
        assert data["relationships"] == []
        assert data["business_rules"] == []
        assert data["time_columns"] == []
        assert data["filters"] == []
        assert data["examples"] == []

    def test_context_with_data(self) -> None:
        """SchemaContext should hold typed entities."""
        ctx = SchemaContext(
            tables=[
                TableInfo(
                    name="orders",
                    ddl="CREATE TABLE orders (id INT);",
                    description="Orders",
                )
            ],
            columns=[ColumnInfo(table="orders", name="id", type="INT", description="PK")],
            relationships=[
                RelationshipInfo(
                    from_table="orders", from_column="customer_id",
                    to_table="customers", to_column="id",
                )
            ],
            business_rules=[BusinessRule(term="revenue", definition="SUM of totals")],
            time_columns=[TimeColumnInfo(table="orders", column="order_date", granularity="day")],
            filters=[FilterInfo(table="orders", column="status", values=["pending", "shipped"])],
            examples=[SQLExample(question="count orders", sql="SELECT COUNT(*) FROM orders")],
        )
        assert len(ctx.tables) == 1
        assert ctx.table_names == ["orders"]
        assert len(ctx.relationships) == 1
        assert ctx.relationships[0].type == "foreign_key"

    def test_to_prompt_str(self) -> None:
        """to_prompt_str should produce a non-empty string."""
        ctx = SchemaContext(
            tables=[TableInfo(name="orders", ddl="CREATE TABLE orders (id INT);")],
            business_rules=[BusinessRule(term="revenue", definition="SUM of totals")],
        )
        prompt = ctx.to_prompt_str()
        assert "orders" in prompt
        assert "revenue" in prompt
        assert "RELEVANT TABLES" in prompt

    def test_model_dump_json_roundtrip(self) -> None:
        """SchemaContext should survive JSON serialisation roundtrip."""
        ctx = SchemaContext(
            tables=[TableInfo(name="test", ddl="CREATE TABLE test (x INT);")],
        )
        json_str = ctx.model_dump_json()
        restored = SchemaContext.model_validate_json(json_str)
        assert restored.tables[0].name == "test"


# ── MetadataStore tests ───────────────────────────────────────────────────


class TestMetadataStore:
    """Tests for keyword-based metadata retrieval."""

    def test_loads_tables(self, metadata_store: MetadataStore) -> None:
        """Should load all 4 sample tables."""
        names = metadata_store.get_all_table_names()
        assert "customers" in names
        assert "orders" in names
        assert "products" in names
        assert "order_items" in names

    def test_find_tables_by_name(self, metadata_store: MetadataStore) -> None:
        """Direct name match should return the table."""
        tables = metadata_store.find_tables(["orders"])
        assert len(tables) >= 1
        assert any(t.name == "orders" for t in tables)

    def test_find_tables_by_column(self, metadata_store: MetadataStore) -> None:
        """A keyword matching a column name should return its parent table."""
        tables = metadata_store.find_tables(["customer_id"])
        assert any(t.name == "orders" for t in tables)

    def test_find_columns(self, metadata_store: MetadataStore) -> None:
        """Should return all columns for a known table."""
        cols = metadata_store.find_columns("orders")
        col_names = [c.name for c in cols]
        assert "id" in col_names
        assert "customer_id" in col_names
        assert "total" in col_names
        assert "status" in col_names

    def test_find_relationships(self, metadata_store: MetadataStore) -> None:
        """Should find FK from orders → customers."""
        rels = metadata_store.find_relationships(["orders"])
        assert len(rels) >= 1
        assert any(
            r.from_table == "orders" and r.to_table == "customers"
            for r in rels
        )

    def test_find_time_columns(self, metadata_store: MetadataStore) -> None:
        """Should find date columns with granularity metadata."""
        tcs = metadata_store.find_time_columns(["orders"])
        assert len(tcs) >= 1
        assert any(tc.column == "order_date" for tc in tcs)

    def test_find_filters(self, metadata_store: MetadataStore) -> None:
        """Should find enum values for the status column."""
        filters = metadata_store.find_filters(["orders"])
        assert len(filters) >= 1
        status_filter = next(f for f in filters if f.column == "status")
        assert "pending" in status_filter.values
        assert "cancelled" in status_filter.values

    def test_find_business_rules(self, metadata_store: MetadataStore) -> None:
        """Should find glossary entries matching keywords."""
        rules = metadata_store.find_business_rules(["revenue"])
        assert len(rules) >= 1
        assert any("revenue" in r.term.lower() for r in rules)

    def test_expand_related_tables(self, metadata_store: MetadataStore) -> None:
        """FK expansion from orders should include customers."""
        expanded = metadata_store.expand_related_tables(["orders"])
        assert "customers" in expanded
        assert "orders" in expanded

    def test_expand_related_tables_reverse(self, metadata_store: MetadataStore) -> None:
        """FK expansion from customers should include orders (reverse direction)."""
        expanded = metadata_store.expand_related_tables(["customers"])
        assert "orders" in expanded

    def test_find_tables_partial_match(self, metadata_store: MetadataStore) -> None:
        """Keywords in table descriptions should match."""
        tables = metadata_store.find_tables(["purchase"])
        assert any(t.name == "orders" for t in tables)

    def test_schema_graph_expands_relationships(self, metadata_store: MetadataStore) -> None:
        """SchemaGraph should power FK expansion."""
        graph = metadata_store.schema_graph
        assert isinstance(graph, SchemaGraph)
        assert "customers" in graph.neighbors("orders")
        assert "order_items" in graph.neighbors("orders")

    def test_schema_graph_shortest_join_path(self, metadata_store: MetadataStore) -> None:
        """Should expose shortest join paths for future schema linking."""
        path = metadata_store.schema_graph.shortest_join_path("customers", "products")
        assert [rel.from_table for rel in path] == ["orders", "order_items", "order_items"]
        assert [rel.to_table for rel in path] == ["customers", "orders", "products"]

    def test_synonym_groups_loaded(self, metadata_store: MetadataStore) -> None:
        """Synonym-only metadata files should be loaded without becoming tables."""
        groups = metadata_store.get_synonym_groups()
        assert "revenue" in groups
        assert "sales" in groups["revenue"]
        assert "synonyms" not in metadata_store.get_all_table_names()


# ── SchemaContextBuilder tests (keyword path only) ────────────────────────


class TestContextBuilderKeywords:
    """Test keyword extraction and table merging logic."""

    def test_extract_keywords(self) -> None:
        """Should strip stop words and return meaningful tokens."""
        keywords = SchemaContextBuilder._extract_keywords(
            "Show me revenue by region vs last month"
        )
        assert "revenue" in keywords
        assert "region" in keywords
        assert "month" in keywords
        # Stop words should be removed.
        assert "show" not in keywords
        assert "me" not in keywords
        assert "by" not in keywords

    def test_keyword_extractor_expands_synonyms(self, metadata_store: MetadataStore) -> None:
        """Business synonyms should add canonical lookup terms."""
        mapper = SynonymMapper(metadata_store.get_synonym_groups())
        keywords = KeywordExtractor(mapper).extract("Show sales by region")
        assert "sales" in keywords
        assert "revenue" in keywords

    def test_merge_tables_deduplication(self) -> None:
        """Tables from both sources should be merged without duplicates."""
        ddl_chunks = ["CREATE TABLE orders (id INT, total DECIMAL);"]
        keyword_tables = [
            TableInfo(name="orders", ddl="CREATE TABLE orders (...);", description="Orders")
        ]
        merged = SchemaContextBuilder._merge_tables(ddl_chunks, keyword_tables)
        # Should have exactly one "orders" entry (keyword version wins).
        order_tables = [t for t in merged if t.name == "orders"]
        assert len(order_tables) == 1

    def test_deduplicate_rules(self) -> None:
        """Duplicate business rules should be removed."""
        rules = [
            BusinessRule(term="revenue", definition="SUM of totals"),
            # Same content, different case.
            BusinessRule(term="Revenue", definition="sum of totals"),
            BusinessRule(term="profit", definition="Revenue minus cost"),
        ]
        unique = SchemaContextBuilder._deduplicate_rules(rules)
        assert len(unique) == 2

    def test_parse_sql_examples_dict_format(self) -> None:
        """Should parse Vanna's dict-format SQL examples."""
        raw = [{"question": "count orders", "sql": "SELECT COUNT(*) FROM orders"}]
        examples = SchemaContextBuilder._parse_sql_examples(raw)
        assert len(examples) == 1
        assert examples[0].question == "count orders"

    def test_parse_sql_examples_tuple_format(self) -> None:
        """Should parse Vanna's tuple-format SQL examples."""
        raw = [("count orders", "SELECT COUNT(*) FROM orders")]
        examples = SchemaContextBuilder._parse_sql_examples(raw)
        assert len(examples) == 1

    def test_parse_sql_examples_empty(self) -> None:
        """Empty input should return empty list."""
        assert SchemaContextBuilder._parse_sql_examples([]) == []

    def test_parse_sql_examples_malformed(self) -> None:
        """Malformed items should be skipped gracefully."""
        raw = [None, 42, {"question": "", "sql": ""}, "garbage"]
        examples = SchemaContextBuilder._parse_sql_examples(raw)
        assert len(examples) == 0


class TestReranking:
    """Tests for weighted table reranking."""

    def test_reranker_uses_weighted_signals(self, sample_settings: Settings) -> None:
        """Semantic, keyword, and relationship scores should combine predictably."""
        candidates = [
            TableCandidate(
                table=TableInfo(name="orders"),
                semantic_score=1.0,
                sources={"semantic"},
            ),
            TableCandidate(
                table=TableInfo(name="customers"),
                keyword_score=1.0,
                relationship_score=1.0,
                sources={"keyword", "relationship"},
            ),
            TableCandidate(
                table=TableInfo(name="order_items"),
                relationship_score=1.0,
                sources={"relationship"},
            ),
        ]

        ranked = TableReranker(sample_settings).rank(candidates)
        scores = {item.table.name: item.score for item in ranked}

        assert scores["orders"] == pytest.approx(0.5)
        assert scores["customers"] == pytest.approx(0.5)
        assert scores["order_items"] == pytest.approx(0.2)


class TestRetrievalPipeline:
    """Tests for the stage-based retrieval pipeline via the public builder."""

    def test_sql_examples_disabled_by_default(
        self,
        sample_settings: Settings,
        metadata_store: MetadataStore,
    ) -> None:
        """SQL examples should remain empty unless explicitly enabled."""
        fake_vanna = FakeVannaRetriever()
        settings = sample_settings.model_copy(
            update={"retrieval_include_sql_examples": False, "retrieval_sql_limit": 2}
        )
        builder = SchemaContextBuilder(
            vanna_retriever=fake_vanna,
            metadata_store=metadata_store,
            settings=settings,
        )

        result = builder.retrieve_with_telemetry("Show sales by region")

        assert result.context.examples == []
        assert fake_vanna.sql_called is False
        assert "orders" in result.telemetry.semantic_tables
        assert "customers" in result.context.table_names
        assert result.telemetry.final_ranked_tables

    def test_sql_examples_can_be_enabled(
        self,
        sample_settings: Settings,
        metadata_store: MetadataStore,
    ) -> None:
        """SQL examples remain available as an explicit opt-in path."""
        fake_vanna = FakeVannaRetriever()
        settings = sample_settings.model_copy(
            update={"retrieval_include_sql_examples": True, "retrieval_sql_limit": 1}
        )
        builder = SchemaContextBuilder(
            vanna_retriever=fake_vanna,
            metadata_store=metadata_store,
            settings=settings,
        )

        result = builder.retrieve_with_telemetry("Show revenue")

        assert fake_vanna.sql_called is True
        assert len(result.context.examples) == 1
        assert result.telemetry.sql_examples_enabled is True
