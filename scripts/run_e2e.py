"""
End-to-end integration script for Step 2.

Trains the retrieval layer, then runs a sample retrieval and
prints the structured SchemaContext output.

Usage:
    source .venv/bin/activate
    python -m scripts.run_e2e
"""

import json
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(name)s | %(levelname)s | %(message)s")


def main() -> None:
    from my_agent.core.training import SchemaIngestor
    from my_agent.core.retrieval import SchemaContextBuilder

    print("=" * 60)
    print("STEP 2 — END-TO-END INTEGRATION TEST")
    print("=" * 60)

    # ── Phase 1: Train ────────────────────────────────────────────────
    print("\n[1/3] Training retrieval layer...")
    ingestor = SchemaIngestor()

    ddl_count = ingestor.ingest_from_sqlite("./data/sample.db")
    print(f"  Ingested {ddl_count} DDL statements from SQLite")

    meta_count = ingestor.ingest_metadata_directory()
    print(f"  Ingested {meta_count} YAML metadata files")

    ddl_file_count = ingestor.ingest_ddl_directory("./data/ddl")
    print(f"  Ingested {ddl_file_count} DDL .sql files")

    print("  Training complete.\n")

    # ── Phase 2: Retrieve ─────────────────────────────────────────────
    test_questions = [
        "Show me revenue by region vs last month",
        "How many active customers do we have?",
        "What are the top selling products by category?",
    ]

    builder = SchemaContextBuilder()

    for i, question in enumerate(test_questions, 1):
        print(f"[2/{len(test_questions)}] Retrieving context for: \"{question}\"")
        ctx = builder.retrieve(question)

        print(f"  Tables:        {ctx.table_names}")
        print(f"  Columns:       {len(ctx.columns)}")
        print(f"  Relationships: {len(ctx.relationships)}")
        print(f"  Business rules:{len(ctx.business_rules)}")
        print(f"  Time columns:  {len(ctx.time_columns)}")
        print(f"  Filters:       {len(ctx.filters)}")
        print(f"  Examples:      {len(ctx.examples)}")
        print()

    # ── Phase 3: Print full JSON output ───────────────────────────────
    print("[3/3] Full SchemaContext JSON for first question:\n")
    ctx = builder.retrieve(test_questions[0])
    print(json.dumps(ctx.model_dump(), indent=2, default=str))

    print("\n" + "=" * 60)
    print("Prompt string for LLM injection:")
    print("=" * 60)
    print(ctx.to_prompt_str())

    print("\n[DONE] Step 2 integration test complete.")


if __name__ == "__main__":
    main()
