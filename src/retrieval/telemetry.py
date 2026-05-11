"""Structured retrieval telemetry for logs and debugging."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.retrieval.ranking import RankedTable


@dataclass(frozen=True)
class RetrievalTelemetry:
    """Explain what each retrieval stage contributed to the final context."""

    question: str
    keywords: list[str] = field(default_factory=list)
    semantic_tables: list[str] = field(default_factory=list)
    keyword_tables: list[str] = field(default_factory=list)
    expanded_tables: list[str] = field(default_factory=list)
    final_ranked_tables: list[RankedTable] = field(default_factory=list)
    documentation_chunks: int = 0
    business_rules: int = 0
    sql_examples_enabled: bool = False
    sql_examples: int = 0

    def to_log_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable snapshot suitable for logs."""
        return {
            "question": self.question,
            "keywords": self.keywords,
            "semantic_tables": self.semantic_tables,
            "keyword_tables": self.keyword_tables,
            "expanded_tables": self.expanded_tables,
            "final_ranked_tables": [
                {
                    "table": item.table.name,
                    "score": round(item.score, 4),
                    "semantic_score": round(item.semantic_score, 4),
                    "keyword_score": round(item.keyword_score, 4),
                    "relationship_score": round(item.relationship_score, 4),
                    "sources": list(item.sources),
                }
                for item in self.final_ranked_tables
            ],
            "documentation_chunks": self.documentation_chunks,
            "business_rules": self.business_rules,
            "sql_examples_enabled": self.sql_examples_enabled,
            "sql_examples": self.sql_examples,
        }
