"""Schema relationship graph for join-aware retrieval."""

from __future__ import annotations

from collections import deque
from typing import Any

try:
    import networkx as nx
except ImportError:  # pragma: no cover - exercised only when dependency is absent.
    nx = None

from src.models.schema_context import RelationshipInfo


class SchemaGraph:
    """
    A graph abstraction over table relationships.

    The adjacency map is kept as the portable source of truth. When networkx is
    installed, the same edges are also mirrored into a networkx.Graph so future
    join-path and ambiguity logic can use its algorithms directly.
    """

    def __init__(self) -> None:
        self._adjacency: dict[str, set[str]] = {}
        self._relationships: dict[tuple[str, str], list[RelationshipInfo]] = {}
        self._graph: Any = nx.Graph() if nx else None

    @classmethod
    def from_metadata(cls, tables: dict[str, dict[str, Any]]) -> SchemaGraph:
        """Build a graph from loaded YAML metadata."""
        graph = cls()

        for table_name, data in tables.items():
            graph.add_table(table_name)
            for rel in data.get("relationships", []):
                to_table = str(rel.get("to_table", "")).lower()
                if not to_table:
                    continue
                graph.add_relationship(
                    RelationshipInfo(
                        from_table=table_name,
                        from_column=str(rel.get("from_column", "")),
                        to_table=to_table,
                        to_column=str(rel.get("to_column", "")),
                        type=str(rel.get("type", "foreign_key")),
                    )
                )

        return graph

    def add_table(self, table_name: str) -> None:
        """Add a table node if it does not already exist."""
        name = self._normalise(table_name)
        if not name:
            return
        self._adjacency.setdefault(name, set())
        if self._graph is not None:
            self._graph.add_node(name)

    def add_relationship(self, relationship: RelationshipInfo) -> None:
        """Add an undirected relationship edge for join traversal."""
        left = self._normalise(relationship.from_table)
        right = self._normalise(relationship.to_table)
        if not left or not right:
            return

        self.add_table(left)
        self.add_table(right)
        self._adjacency[left].add(right)
        self._adjacency[right].add(left)

        key = self._edge_key(left, right)
        self._relationships.setdefault(key, []).append(relationship)

        if self._graph is not None:
            self._graph.add_edge(left, right)

    def neighbors(self, table_name: str) -> list[str]:
        """Return directly connected tables."""
        name = self._normalise(table_name)
        return sorted(self._adjacency.get(name, set()))

    def expand_related_tables(self, table_names: list[str], max_hops: int = 1) -> list[str]:
        """Return seed tables plus neighbors within max_hops."""
        seeds = [self._normalise(name) for name in table_names if self._normalise(name)]
        expanded: set[str] = set(seeds)

        if max_hops <= 0:
            return sorted(expanded)

        queue: deque[tuple[str, int]] = deque((seed, 0) for seed in seeds)
        while queue:
            table_name, distance = queue.popleft()
            if distance >= max_hops:
                continue
            for neighbor in self._adjacency.get(table_name, set()):
                if neighbor not in expanded:
                    expanded.add(neighbor)
                    queue.append((neighbor, distance + 1))

        return sorted(expanded)

    def shortest_join_path(self, from_table: str, to_table: str) -> list[RelationshipInfo]:
        """Return relationships along the shortest join path, or an empty list."""
        start = self._normalise(from_table)
        end = self._normalise(to_table)
        if not start or not end or start == end:
            return []

        path = self._shortest_node_path(start, end)
        if len(path) < 2:
            return []

        relationships: list[RelationshipInfo] = []
        for left, right in zip(path, path[1:]):
            edge_relationships = self._relationships.get(self._edge_key(left, right), [])
            if edge_relationships:
                relationships.append(edge_relationships[0])
        return relationships

    def has_ambiguous_shortest_paths(self, from_table: str, to_table: str) -> bool:
        """Return True when more than one shortest join path exists."""
        start = self._normalise(from_table)
        end = self._normalise(to_table)
        if not start or not end or start == end:
            return False

        if self._graph is not None:
            try:
                paths = nx.all_shortest_paths(self._graph, start, end)
                first = next(paths, None)
                if first is None:
                    return False
                return next(paths, None) is not None
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                return False

        return self._count_shortest_paths(start, end) > 1

    def to_adjacency_map(self) -> dict[str, list[str]]:
        """Return a serialisable adjacency map for debugging."""
        return {table: sorted(neighbors) for table, neighbors in sorted(self._adjacency.items())}

    def _shortest_node_path(self, start: str, end: str) -> list[str]:
        if self._graph is not None:
            try:
                return list(nx.shortest_path(self._graph, start, end))
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                return []

        queue: deque[list[str]] = deque([[start]])
        seen = {start}
        while queue:
            path = queue.popleft()
            table_name = path[-1]
            if table_name == end:
                return path
            for neighbor in self._adjacency.get(table_name, set()):
                if neighbor in seen:
                    continue
                seen.add(neighbor)
                queue.append([*path, neighbor])
        return []

    def _count_shortest_paths(self, start: str, end: str) -> int:
        shortest_length: int | None = None
        path_count = 0
        queue: deque[list[str]] = deque([[start]])

        while queue:
            path = queue.popleft()
            if shortest_length is not None and len(path) > shortest_length:
                continue

            table_name = path[-1]
            if table_name == end:
                shortest_length = len(path)
                path_count += 1
                if path_count > 1:
                    return path_count
                continue

            for neighbor in self._adjacency.get(table_name, set()):
                if neighbor not in path:
                    queue.append([*path, neighbor])

        return path_count

    @staticmethod
    def _edge_key(left: str, right: str) -> tuple[str, str]:
        return tuple(sorted((left, right)))

    @staticmethod
    def _normalise(table_name: str) -> str:
        return table_name.lower().strip()
