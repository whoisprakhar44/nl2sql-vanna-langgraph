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
src/
├── config/settings.py        # Pydantic Settings (env-based config)
├── models/schema_context.py  # SchemaContext Pydantic model (output contract)
├── retrieval/
│   ├── vanna_retriever.py    # Vanna wrapper (retrieval-only, SQL gen blocked)
│   ├── metadata_store.py     # YAML metadata + keyword search
│   └── context_builder.py    # Orchestrator → SchemaContext
└── training/
    └── ingest.py             # Schema ingestion scripts

data/
├── metadata/                 # YAML metadata per table
├── ddl/                      # Raw DDL .sql files
└── sample.db                 # SQLite test database

tests/
└── test_retrieval.py         # Unit tests (no Ollama required)
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
from src.training import SchemaIngestor

ingestor = SchemaIngestor()

# Auto-ingest DDL from the SQLite database
ingestor.ingest_from_sqlite()

# Ingest YAML metadata (descriptions, business rules)
ingestor.ingest_metadata_directory()
```

### 5. Retrieve context for a question

```python
from src.retrieval import SchemaContextBuilder

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

## How This Connects to LangGraph

Step 2 is designed to slot into a LangGraph pipeline as a node or tool:

```python
# Future: LangGraph integration (Step 3+)
from langgraph.graph import StateGraph
from src.retrieval import SchemaContextBuilder

builder = SchemaContextBuilder()

def retrieve_context(state):
    """LangGraph node that runs Step 2 retrieval."""
    question = state["question"]
    ctx = builder.retrieve(question)
    return {"schema_context": ctx.model_dump()}

# Add to your graph
graph = StateGraph(...)
graph.add_node("retrieve_context", retrieve_context)
```

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| Vanna `generate_sql()` is blocked | Enforces architecture: Vanna = retrieval only |
| Hybrid retrieval (semantic + keyword) | Embeddings alone miss exact table/column names |
| FK relationship expansion | Ensures join tables are included in context |
| Low retrieval limits (5 DDL, 3 docs, 2 SQL) | Minimises noise for downstream reasoning |
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
| `RETRIEVAL_SQL_LIMIT` | `2` | Max SQL examples per query |

## License

MIT
