"""Tools and reusable helpers for the NL2SQL graph."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from langchain_core.tools import tool

from my_agent.core.retrieval import SchemaContextBuilder


@lru_cache(maxsize=1)
def get_schema_context_builder() -> SchemaContextBuilder:
    """Return a cached retrieval builder for graph nodes and tools."""
    return SchemaContextBuilder()


def retrieve_schema_context(
    question: str,
    builder: Any | None = None,
) -> dict[str, Any]:
    """Run Step 2 retrieval and return graph-state friendly dictionaries."""
    retrieval_builder = builder or get_schema_context_builder()
    result = retrieval_builder.retrieve_with_telemetry(question)
    return {
        "schema_context": result.context.model_dump(),
        "retrieval_telemetry": result.telemetry.to_log_dict(),
    }


@tool
def retrieve_schema_context_tool(question: str) -> dict[str, Any]:
    """Retrieve schema context and telemetry for a natural-language SQL question."""
    return retrieve_schema_context(question)
