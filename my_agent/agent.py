"""Construct the LangGraph graph for the NL2SQL agent."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from my_agent.utils.nodes import (
    analytics_node,
    execute_sql_node,
    generate_sql_node,
    retrieve_context_node,
    schema_linker_node,
)
from my_agent.utils.state import AgentState


def build_graph(include_placeholders: bool = False):
    """
    Build the NL2SQL graph.

    The default graph runs only Step 2 retrieval, which is implemented today.
    Set include_placeholders=True to wire the future Step 3+ placeholder nodes
    into the graph while they are being implemented.
    """
    workflow = StateGraph(AgentState)

    workflow.add_node("retrieve_context", retrieve_context_node)
    workflow.add_edge(START, "retrieve_context")

    if include_placeholders:
        workflow.add_node("schema_linker", schema_linker_node)
        workflow.add_node("generate_sql", generate_sql_node)
        workflow.add_node("execute_sql", execute_sql_node)
        workflow.add_node("analytics", analytics_node)

        workflow.add_edge("retrieve_context", "schema_linker")
        workflow.add_edge("schema_linker", "generate_sql")
        workflow.add_edge("generate_sql", "execute_sql")
        workflow.add_edge("execute_sql", "analytics")
        workflow.add_edge("analytics", END)
    else:
        workflow.add_edge("retrieve_context", END)

    return workflow.compile()


graph = build_graph()
