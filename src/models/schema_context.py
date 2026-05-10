"""
SchemaContext — the structured output contract of Step 2.

This Pydantic model is the ONLY thing downstream LangGraph agents receive
from the retrieval layer.  It is designed to be:

  - Typed and validated (catches malformed context early)
  - Serialisable to JSON via .model_dump()
  - Frozen after construction (immutable — prevents accidental mutation)
  - Lightweight (top 3-5 entities, not the whole schema)

The shape matches the contract defined in the architecture doc:
  {tables, columns, relationships, business_rules, time_columns, filters, examples}
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ── Atomic building blocks ──────────────────────────────────────────────────


class TableInfo(BaseModel):
    """A single relevant table retrieved from the schema."""

    name: str = Field(..., description="Table name as it appears in the database")
    ddl: str = Field(default="", description="CREATE TABLE statement (may be partial)")
    description: str = Field(default="", description="Human-readable purpose of this table")


class ColumnInfo(BaseModel):
    """A single relevant column with type and semantic description."""

    table: str = Field(..., description="Parent table name")
    name: str = Field(..., description="Column name")
    type: str = Field(default="", description="SQL data type (e.g. INTEGER, VARCHAR(255))")
    description: str = Field(default="", description="What this column represents")


class RelationshipInfo(BaseModel):
    """A foreign-key or logical relationship between two tables."""

    from_table: str
    from_column: str
    to_table: str
    to_column: str
    type: str = Field(
        default="foreign_key",
        description="Relationship type: foreign_key | logical | many_to_many",
    )


class BusinessRule(BaseModel):
    """A business glossary entry or domain rule."""

    term: str = Field(..., description="Business term or concept name")
    definition: str = Field(..., description="Plain-English definition / SQL interpretation")


class TimeColumnInfo(BaseModel):
    """Metadata about a date/time column for temporal query understanding."""

    table: str
    column: str
    granularity: str = Field(
        default="day",
        description="Finest useful granularity: second | minute | hour | day | month | year",
    )
    format: str = Field(default="", description="Storage format hint, e.g. YYYY-MM-DD")


class FilterInfo(BaseModel):
    """Known enum / categorical values for a column."""

    table: str
    column: str
    values: list[str] = Field(
        default_factory=list,
        description="Allowed / known values for this column",
    )


class SQLExample(BaseModel):
    """An example question → SQL pair (used sparingly)."""

    question: str
    sql: str


# ── Composite context ──────────────────────────────────────────────────────


class SchemaContext(BaseModel):
    """
    The complete structured context produced by Step 2.

    Consumed by LangGraph agents in Steps 3+.
    Designed for JSON serialisation via .model_dump().
    """

    tables: list[TableInfo] = Field(default_factory=list)
    columns: list[ColumnInfo] = Field(default_factory=list)
    relationships: list[RelationshipInfo] = Field(default_factory=list)
    business_rules: list[BusinessRule] = Field(default_factory=list)
    time_columns: list[TimeColumnInfo] = Field(default_factory=list)
    filters: list[FilterInfo] = Field(default_factory=list)
    examples: list[SQLExample] = Field(default_factory=list)

    # -- Convenience -----------------------------------------------------------

    @property
    def table_names(self) -> list[str]:
        """Quick access to the list of retrieved table names."""
        return [t.name for t in self.tables]

    def to_prompt_str(self) -> str:
        """
        Render context as a compact string suitable for injection into
        an LLM prompt.  Downstream agents can use this directly.
        """
        parts: list[str] = []

        if self.tables:
            parts.append("=== RELEVANT TABLES ===")
            for t in self.tables:
                header = f"-- {t.name}"
                if t.description:
                    header += f": {t.description}"
                parts.append(header)
                if t.ddl:
                    parts.append(t.ddl)
                parts.append("")

        if self.relationships:
            parts.append("=== RELATIONSHIPS ===")
            for r in self.relationships:
                parts.append(
                    f"{r.from_table}.{r.from_column} -> "
                    f"{r.to_table}.{r.to_column} ({r.type})"
                )
            parts.append("")

        if self.business_rules:
            parts.append("=== BUSINESS RULES ===")
            for b in self.business_rules:
                parts.append(f"- {b.term}: {b.definition}")
            parts.append("")

        if self.time_columns:
            parts.append("=== TIME COLUMNS ===")
            for tc in self.time_columns:
                line = f"- {tc.table}.{tc.column} (granularity={tc.granularity})"
                if tc.format:
                    line += f" format={tc.format}"
                parts.append(line)
            parts.append("")

        if self.filters:
            parts.append("=== KNOWN FILTER VALUES ===")
            for f in self.filters:
                parts.append(f"- {f.table}.{f.column}: {f.values}")
            parts.append("")

        if self.examples:
            parts.append("=== EXAMPLE QUERIES ===")
            for ex in self.examples:
                parts.append(f"Q: {ex.question}")
                parts.append(f"SQL: {ex.sql}")
                parts.append("")

        return "\n".join(parts)
