# Music Store Customer Support Bot

A customer support bot for a music store (Chinook database) that compares **two agent frameworks** (LangGraph and OpenHands SDK) and **two tool strategies** (structured pre-built tools vs. deep agent with raw SQL). Includes human-in-the-loop approval, observability via Laminar, and benchmarks for each approach.

## Repo Structure

```
sql-support-bot/
├── agent.ipynb            # LangGraph workflow notebook with demo scenarios
├── deep_agent.ipynb       # LangGraph deep agent notebook (LLM writes SQL)
├── benchmark.py           # LangGraph benchmark: Workflow vs Deep Agent
├── oh_structured.ipynb    # OpenHands SDK structured agent (6 pre-built tools + HITL)
├── oh_deep_agent.ipynb    # OpenHands SDK deep agent (2 generic tools, LLM writes SQL)
├── oh_benchmark.py        # OpenHands SDK benchmark: Structured vs Deep Agent
├── oh_tool_helper.py      # @simple_tool decorator for OpenHands SDK
├── oh_prompts/
│   ├── structured_agent.j2  # System prompt for structured agent
│   └── deep_agent.j2        # System prompt for deep agent
├── agent/
│   ├── __init__.py        # Re-exports the compiled graph for LangGraph Studio
│   ├── graph.py           # Full LangGraph graph definition
│   └── db.py              # Chinook database loader (shared across all notebooks)
├── langgraph.json         # LangGraph Studio configuration
├── pyproject.toml         # Project dependencies
├── PLAN.md                # Architecture plan and implementation checklist
└── .env                   # API keys (see Setup)
```

## Approach Comparison

The project implements the same customer support bot four ways — two frameworks x two tool strategies:

|  | **Structured Tools** (6 pre-built SQL queries) | **Deep Agent** (2 generic tools, LLM writes SQL) |
|---|---|---|
| **LangGraph** | `agent.ipynb` + `agent/graph.py` | `deep_agent.ipynb` |
| **OpenHands SDK** | `oh_structured.ipynb` | `oh_deep_agent.ipynb` |

### Structured Tools Approach

Pre-built tools with fixed SQL queries — the LLM never writes SQL:

- `search_catalog` — search by artist, song, or album
- `get_recommendations` — popular tracks in a genre
- `get_purchase_history` — customer's recent invoices
- `get_invoice_details` — line items for an invoice
- `get_profile` / `update_profile` — view/update customer info

The LangGraph version uses a multi-node graph with routing, guardrails, and specialized agents. The OpenHands SDK version uses a single agent with all 6 tools plus a `SecurityAnalyzer` for human-in-the-loop approval on profile updates.

### Deep Agent Approach

Two generic tools — the LLM writes its own SQL:

- `get_schema` — returns the full database schema
- `run_sql` — executes any SQL query (with safety checks)

Maximum flexibility with minimal code. The LLM handles routing, query generation, and result formatting on its own.

### Human-in-the-Loop (HITL)

Both frameworks support HITL approval for sensitive operations (profile updates):

- **LangGraph**: Uses `interrupt()` from `langgraph.types` to pause the graph and wait for user approval
- **OpenHands SDK**: Uses `ConfirmRisky` policy with a custom `SecurityAnalyzer` that flags `update_profile` calls as high-risk

### `@simple_tool` Decorator (`oh_tool_helper.py`)

The OpenHands SDK requires verbose class-based tool definitions (Action, Observation, Executor, ToolDefinition). The `@simple_tool` decorator simplifies this to a single function with a docstring — similar to LangGraph's `@tool`:

```python
from oh_tool_helper import simple_tool, tool_spec

@simple_tool
def get_schema() -> str:
    """Get the database schema."""
    return db.get_table_info()

@simple_tool(read_only=False, destructive=True)
def update_profile(field: str, new_value: str) -> str:
    """Update a profile field.

    Args:
        field: The field to update.
        new_value: The new value.
    """
    db.run(f"UPDATE Customer SET \"{field}\" = '{new_value}' WHERE CustomerId = {CID};")
    return f"Updated {field}."

agent = Agent(llm=llm, tools=[tool_spec(get_schema), tool_spec(update_profile)])
```

## Benchmark Results

### LangGraph: Workflow vs Deep Agent

| Metric | Workflow (Structured) | Deep Agent (Raw SQL) |
|---|---|---|
| **Latency** | 27.2s | 38.6s (1.4x slower) |
| **Tokens** | 6,457 | 114,519 (17.7x more) |
| **Cost** | $0.022 | $0.209 (9.4x more) |

### OpenHands SDK: Structured vs Deep Agent

| Metric | Structured (6 tools) | Deep Agent (Raw SQL) |
|---|---|---|
| **Latency** | 15.2s | 33.5s (2.2x slower) |
| **Tokens** | 15,917 | 46,636 (2.9x more) |
| **Cost** | $0.039 | $0.106 (2.7x more) |

**Key takeaway**: Structured tools consistently win on speed, tokens, and cost. The deep agent trades efficiency for flexibility — it can answer ad-hoc queries (e.g., "top 5 genres by track count") that structured tools can't handle without a dedicated tool.

## Observability

All approaches are instrumented with **Laminar** for trace monitoring. Traces are sent automatically via the `lmnr` package. 

## Setup

```bash
# Create virtual environment and install dependencies
uv venv
uv sync

# Set up environment variables
cp .env.example .env  # Then fill in your API keys
```

Required environment variables in `.env`:

```
OPENAI_API_KEY=sk-...
LMNR_PROJECT_API_KEY=...         # For Laminar tracing
```

## Running

### Notebooks

Open any notebook in Jupyter or VS Code and run all cells:

- `agent.ipynb` — LangGraph structured workflow
- `deep_agent.ipynb` — LangGraph deep agent
- `oh_structured.ipynb` — OpenHands SDK structured agent
- `oh_deep_agent.ipynb` — OpenHands SDK deep agent

### LangGraph Studio

The graph is configured for Studio via `langgraph.json`. Open the project in LangGraph Studio to visualize and interact with the workflow.

### Benchmarks

```bash
# LangGraph benchmark
uv run python benchmark.py

# OpenHands SDK benchmark
uv run python oh_benchmark.py
```
