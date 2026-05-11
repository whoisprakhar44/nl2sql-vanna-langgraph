"""
Table reranking for the retrieval pipeline.

Vanna returns nearest chunks, not a final schema decision. This layer converts
candidate signals into a stable table ranking that can be logged, tested, and
tuned without changing the downstream SchemaContext contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.config.settings import Settings, get_settings
from src.models.schema_context import TableInfo


@dataclass
class TableCandidate:
    """A table plus retrieval signals collected across pipeline stages."""

    table: TableInfo
    semantic_score: float = 0.0
    keyword_score: float = 0.0
    relationship_score: float = 0.0
    sources: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class RankedTable:
    """A reranked table with explainable component scores."""

    table: TableInfo
    score: float
    semantic_score: float
    keyword_score: float
    relationship_score: float
    sources: tuple[str, ...]


class TableReranker:
    """
    Score table candidates using weighted retrieval signals.

    Default formula:
        semantic_score * 0.5 + keyword_score * 0.3 + relationship_score * 0.2
    """

    def __init__(self, settings: Settings | None = None) -> None:
        settings = settings or get_settings()
        self._settings = settings
        self._semantic_weight = settings.retrieval_score_semantic_weight
        self._keyword_weight = settings.retrieval_score_keyword_weight
        self._relationship_weight = settings.retrieval_score_relationship_weight

    def rank(
        self,
        candidates: list[TableCandidate],
        limit: int | None = None,
    ) -> list[RankedTable]:
        """Return candidates ordered by weighted score, then table name."""
        ranked: list[RankedTable] = []

        for candidate in candidates:
            semantic = self._clamp(candidate.semantic_score)
            keyword = self._clamp(candidate.keyword_score)
            relationship = self._clamp(candidate.relationship_score)
            score = (
                semantic * self._semantic_weight
                + keyword * self._keyword_weight
                + relationship * self._relationship_weight
            )
            ranked.append(
                RankedTable(
                    table=candidate.table,
                    score=score,
                    semantic_score=semantic,
                    keyword_score=keyword,
                    relationship_score=relationship,
                    sources=tuple(sorted(candidate.sources)),
                )
            )

        ranked.sort(key=lambda item: (-item.score, item.table.name))

        table_limit = self._settings.retrieval_table_limit if limit is None else limit
        if table_limit <= 0:
            return ranked
        return ranked[:table_limit]

    @staticmethod
    def _clamp(value: float) -> float:
        """Keep signal scores in the 0..1 range."""
        return max(0.0, min(1.0, value))
