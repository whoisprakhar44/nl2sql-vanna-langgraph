"""State definition for the NL2SQL LangGraph agent."""

from __future__ import annotations

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    """Shared state passed between LangGraph nodes."""

    question: str
    schema_context: dict[str, Any]
    retrieval_telemetry: dict[str, Any]
    linked_schema: dict[str, Any]
    generated_sql: str
    execution_result: dict[str, Any]
    answer: str
    pipeline_stage: str
    errors: list[str]
