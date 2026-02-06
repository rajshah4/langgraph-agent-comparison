# 🎵 Music Store Customer Support Bot

A customer support bot for a music store (Chinook database) built with **LangGraph**, **LangChain**, and **LangSmith**. Demonstrates multi-agent orchestration, human-in-the-loop approval, and LLM-driven SQL — with a benchmark comparing both approaches.

## Repo Structure

```
langgraph-agent-comparison/
├── agent.ipynb          # ★ Main notebook — full LangGraph workflow with demo scenarios
├── deep_agent.ipynb     # Alternative: Deep Agents + raw SQL (minimal code, LLM writes SQL)
├── benchmark.py         # Compares Workflow vs. Deep Agent on 6 queries (latency, tokens, cost)
├── agent/
│   ├── __init__.py      # Re-exports the compiled graph for LangGraph Studio
│   ├── graph.py         # Full graph definition (Studio entry point)
│   └── db.py            # Chinook database loader (shared by notebook + Studio)
├── langgraph.json       # LangGraph Studio configuration
├── pyproject.toml       # Project dependencies
├── PLAN.md              # Architecture plan and implementation checklist
└── .env                 # API keys (OPENAI_API_KEY, LANGSMITH_API_KEY)
```

## Two Approaches

### 1. LangGraph Workflow (`agent.ipynb` + `agent/graph.py`)

A multi-node graph with explicit routing, specialized agents, and human-in-the-loop:

- **Guardrail** → moderation filter on every input
- **Router** → classifies intent and dispatches to the right agent
- **Music Agent** → catalog search, genre recommendations (2 tools)
- **Orders Agent** → purchase history, invoice details (2 tools)
- **Account Agent** → profile view/update with HITL approval via `interrupt()` (2 tools)

**6 tools, 8 nodes, conditional edges, ReAct loops.**

### 2. Deep Agents + Raw SQL (`deep_agent.ipynb`)

A single agent with just 2 generic tools (`get_schema` + `run_query`). The LLM writes its own SQL — no routing, no pre-built queries. Maximum simplicity (~30 lines of agent code).

### Benchmark Results (6 queries)

| Metric | Graph (Workflow) | Deep Agents + Raw SQL |
|---|---|---|
| **Latency** | 27.2s | 38.6s (1.4× slower) |
| **Tokens** | 6,457 | 114,519 (17.7× more) |
| **Cost** | $0.022 | $0.209 (9.4× more) |

Run `uv run python benchmark.py` to reproduce. Traces appear in LangSmith under the `music-store-benchmark` project.

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
LANGSMITH_API_KEY=lsv2_...
```

## Running

### Notebooks

Open `agent.ipynb` or `deep_agent.ipynb` in Jupyter/VS Code and run all cells.

### LangGraph Studio

The graph is configured for Studio via `langgraph.json`. Open the project in LangGraph Studio to visualize and interact with the workflow.

### Benchmark

```bash
uv run python benchmark.py
```
