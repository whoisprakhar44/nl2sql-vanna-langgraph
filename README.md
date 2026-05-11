# NL2SQL — Reasoning-First Natural Language to SQL

> An agentic NL-to-SQL system using **LangGraph** for reasoning and **Vanna 0.7.x** as a retrieval subsystem.

## Architecture

```
User Question
     │
     ▼
┌────────────────────────────────────┐
│  Step 2 — Context Retrieval        │  ← YOU ARE HERE
│  ┌──────────────┐ ┌─────────────┐ │
│  │ VannaRetriever│ │MetadataStore│ │
│  │ (ChromaDB)    │ │ (YAML)      │ │
│  └──────────────┘ └─────────────┘ │
│         ↓                ↓         │
│  RetrievalPipeline                 │
│  semantic → keyword → FK → rank     │
│         ↓                          │
│  SchemaContextBuilder              │
│         ↓                          │
│  SchemaContext (JSON)              │
└────────────────────────────────────┘
     │
     ▼
┌────────────────────────────────────┐
│  Step 3 — Schema Linker Agent      │  ← LangGraph (future)
│  Step 4 — SQL Gen + Execution      │
│  Step 5 — Analytics Agent          │
└────────────────────────────────────┘
```

**Key principle:** Retrieval assists reasoning. Retrieval does NOT replace reasoning.

- **Vanna** is used ONLY for semantic retrieval against ChromaDB
- **LangGraph** owns all reasoning, SQL generation, and execution
- Step 2 outputs a structured `SchemaContext` JSON consumed by downstream agents

## Project Structure

```
my_agent/
├── agent.py                # LangGraph construction
├── __init__.py
└── utils/
    ├── state.py            # AgentState shared across nodes
    ├── nodes.py            # LangGraph node functions
    └── tools.py            # Reusable graph tools/helpers

├── core/                     # The Retrieval Subsystem (formerly src/)
│   ├── config/settings.py        # Pydantic Settings (env-based config)
│   ├── models/schema_context.py  # SchemaContext Pydantic model (output contract)
│   ├── retrieval/
│   │   ├── vanna_retriever.py    # Vanna wrapper (retrieval-only, SQL gen blocked)
│   │   ├── metadata_store.py     # YAML metadata + keyword search
│   │   ├── schema_graph.py       # FK graph + join-path helpers
│   │   ├── keyword_extractor.py  # Stop-word filtering + synonym expansion
│   │   ├── synonym_mapper.py     # Business synonym normalisation
│   │   ├── context_merger.py     # Candidate merge + SchemaContext assembly
│   │   ├── retrieval_pipeline.py # Stage-based retrieval pipeline
│   │   ├── ranking/
│   │   │   └── reranker.py       # Weighted table reranking
│   │   └── context_builder.py    # Stable public API → SchemaContext
│   └── training/
│       └── ingest.py             # Schema ingestion scripts

data/
├── metadata/                 # YAML metadata per table
├── ddl/                      # Raw DDL .sql files
└── sample.db                 # SQLite test database

tests/
├── test_agent_graph.py       # LangGraph node scaffold tests
└── test_retrieval.py         # Retrieval unit tests (no Ollama required)

langgraph.json                # LangGraph app configuration
```

## Quick Start

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. Create the sample database

```bash
python -m scripts.create_sample_db
```

### 3. Start Ollama (for embeddings)

```bash
ollama serve
ollama pull llama3.2
```

### 4. Train the retrieval layer

```python
from my_agent.core.training import SchemaIngestor

ingestor = SchemaIngestor()

# Auto-ingest DDL from the SQLite database
ingestor.ingest_from_sqlite()

# Ingest YAML metadata (descriptions, business rules)
ingestor.ingest_metadata_directory()
```

### 5. Retrieve context for a question

```python
from my_agent.core.retrieval import SchemaContextBuilder

builder = SchemaContextBuilder()
ctx = builder.retrieve("Show me revenue by region vs last month")

# Structured JSON for LangGraph state
print(ctx.model_dump_json(indent=2))

# Compact string for LLM prompt injection
print(ctx.to_prompt_str())
```

### 6. Run tests

```bash
pytest tests/ -v
```

## LangGraph App

The repository exposes a LangGraph app through `langgraph.json`:

```python
from my_agent.agent import graph

result = graph.invoke({"question": "Show me revenue by region"})
print(result["schema_context"])
```

The current default graph runs the implemented Step 2 retrieval node:

```
START -> retrieve_context -> END
```

Future nodes are already scaffolded in `my_agent/utils/nodes.py`:

```
retrieve_context -> schema_linker -> generate_sql -> execute_sql -> analytics
```

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| Vanna `generate_sql()` is blocked | Enforces architecture: Vanna = retrieval only |
| Hybrid retrieval (semantic + keyword) | Embeddings alone miss exact table/column names |
| FK relationship expansion | Ensures join tables are included in context |
| Schema graph abstraction | Enables shortest join paths and ambiguity checks |
| Weighted reranking | Controls noisy table leakage before prompt assembly |
| SQL examples disabled by default | Avoids template copying and nearest-SQL overfitting |
| Retrieval telemetry | Logs semantic, keyword, expanded, and final ranked tables |
| Pydantic models for output | Type safety + JSON serialisation + validation |
| YAML metadata files | Human-readable, version-controllable schema docs |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `llama3.2` | Model for embeddings |
| `CHROMA_PERSIST_DIR` | `./data/chroma_store` | ChromaDB persistence path |
| `DATABASE_PATH` | `./data/sample.db` | SQLite database path |
| `METADATA_DIR` | `./data/metadata` | YAML metadata directory |
| `RETRIEVAL_DDL_LIMIT` | `5` | Max DDL chunks per query |
| `RETRIEVAL_DOC_LIMIT` | `3` | Max doc chunks per query |
| `RETRIEVAL_INCLUDE_SQL_EXAMPLES` | `false` | Opt-in switch for SQL example retrieval |
| `RETRIEVAL_SQL_LIMIT` | `0` | Max SQL examples per query when enabled |
| `RETRIEVAL_TABLE_LIMIT` | `8` | Max reranked tables in context |
| `RELATIONSHIP_EXPANSION_HOPS` | `1` | FK graph expansion depth |
| `RETRIEVAL_SCORE_SEMANTIC_WEIGHT` | `0.5` | Reranker semantic signal weight |
| `RETRIEVAL_SCORE_KEYWORD_WEIGHT` | `0.3` | Reranker keyword signal weight |
| `RETRIEVAL_SCORE_RELATIONSHIP_WEIGHT` | `0.2` | Reranker relationship signal weight |

## License

MIT
