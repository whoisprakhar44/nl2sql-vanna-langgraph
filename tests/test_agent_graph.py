"""Tests for the LangGraph agent node scaffold."""

from __future__ import annotations

from types import SimpleNamespace

from my_agent.agent import build_graph
from my_agent.utils.nodes import retrieve_context_node
from my_agent.core.models.schema_context import SchemaContext, TableInfo
from my_agent.core.retrieval.telemetry import RetrievalTelemetry


class FakeRetrievalBuilder:
    """Small test double for SchemaContextBuilder."""

    def retrieve_with_telemetry(self, question: str):
        return SimpleNamespace(
            context=SchemaContext(tables=[TableInfo(name="orders")]),
            telemetry=RetrievalTelemetry(question=question),
        )


def test_graph_compiles() -> None:
    """The default retrieval graph should compile without constructing Vanna."""
    graph = build_graph()
    assert graph is not None


def test_placeholder_graph_compiles() -> None:
    """Future Step 3+ placeholder nodes should also be wireable."""
    graph = build_graph(include_placeholders=True)
    assert graph is not None


def test_retrieve_context_node_uses_injected_builder() -> None:
    """The retrieval node should be testable without Ollama or ChromaDB."""
    output = retrieve_context_node(
        {"question": "Show revenue by region"},
        builder=FakeRetrievalBuilder(),
    )

    assert output["pipeline_stage"] == "context_retrieved"
    assert output["schema_context"]["tables"][0]["name"] == "orders"
    assert output["retrieval_telemetry"]["question"] == "Show revenue by region"


def test_retrieve_context_node_requires_question() -> None:
    """Missing question should produce an explicit state error."""
    output = retrieve_context_node({})

    assert output["pipeline_stage"] == "retrieval_failed"
    assert "Missing required state field: question" in output["errors"]
