"""Node functions for the NL2SQL LangGraph agent."""

from __future__ import annotations

from typing import Any

from my_agent.utils.state import AgentState
from my_agent.utils.tools import retrieve_schema_context


def retrieve_context_node(
    state: AgentState,
    builder: Any | None = None,
) -> dict[str, Any]:
    """Step 2 node: retrieve schema context for the user's question."""
    question = (state.get("question") or "").strip()
    if not question:
        return {
            "pipeline_stage": "retrieval_failed",
            "errors": [*state.get("errors", []), "Missing required state field: question"],
        }

    retrieval_payload = retrieve_schema_context(question, builder=builder)
    return {
        **retrieval_payload,
        "pipeline_stage": "context_retrieved",
    }


def schema_linker_node(state: AgentState) -> dict[str, Any]:
    """Step 3 placeholder: resolve tables, columns, joins, and ambiguity."""
    return {
        "pipeline_stage": "schema_linking_pending",
        "linked_schema": state.get("linked_schema", {}),
    }


def generate_sql_node(state: AgentState) -> dict[str, Any]:
    """Step 4 placeholder: generate SQL from linked schema."""
    return {
        "pipeline_stage": "sql_generation_pending",
        "generated_sql": state.get("generated_sql", ""),
    }


def execute_sql_node(state: AgentState) -> dict[str, Any]:
    """Step 4 placeholder: execute generated SQL safely."""
    return {
        "pipeline_stage": "execution_pending",
        "execution_result": state.get("execution_result", {}),
    }


def analytics_node(state: AgentState) -> dict[str, Any]:
    """Step 5 placeholder: turn execution results into a final answer."""
    return {
        "pipeline_stage": "analytics_pending",
        "answer": state.get("answer", ""),
    }
