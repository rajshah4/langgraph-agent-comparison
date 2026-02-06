# 🎯 Game Plan: Music Store Customer Support Bot

## Analysis of the Starter Code (Problems to Fix)

The existing notebook has several **intentional issues** that need to be addressed:

1. **SQL injection vulnerabilities** — all queries use raw f-strings (`f"SELECT * FROM Customer WHERE CustomerID = {customer_id}"`)
2. **Broken routing logic** — `_is_tool_call` uses `content_blocks` which isn't the correct LangChain API; should use `tool_calls`
3. **No customer authentication** — anyone can look up any customer's data
4. **No human-in-the-loop** — required by the spec
5. **No LangSmith tracing** — required by the spec
6. **No LangGraph Studio support** — no `langgraph.json`, no proper module structure
7. **Flat, tangled graph** — every node has conditional edges to every other node; hard to reason about
8. **Customer agent says "update profile" but has no update tool**
9. **No checkpointer** passed to `compile()` — so no memory across turns
10. **Deprecated APIs** — `set_conditional_entry_point`, `SqliteSaver` (should use `MemorySaver` or newer)

---

## Bot Capabilities (3 Focused Areas)

| Area | What it does | Tools | HITL? |
|------|-------------|-------|-------|
| **🎵 Music Discovery** | Search songs, albums, artists; get genre-based recommendations | `search_catalog`, `get_recommendations_by_genre` | No |
| **🧾 Order History** | View past purchases, invoices, spending summary | `get_purchase_history`, `get_invoice_details` | No |
| **👤 Account Management** | View profile, update email/phone/address | `get_my_profile`, `update_my_profile` | **Yes** — updates require human approval |

This gives us **6 tools** across **3 areas**, hitting all the requirements:
- ✅ At least 2 areas of work
- ✅ Human-in-the-loop (profile updates)
- ✅ Customer data isolation (customer_id in config, all queries filtered)
- ✅ Realistic use cases

---

## Cognitive Architecture (Redesigned Graph)

```
                    ┌─────────────┐
                    │   START     │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  Guardrail  │──── (blocked) ──→ END
                    │ (moderation)│
                    └──────┬──────┘
                           │ (ok)
                    ┌──────▼──────┐
                    │   Router    │
                    └──┬───┬───┬──┘
                       │   │   │
            ┌──────────┘   │   └──────────┐
            ▼              ▼              ▼
     ┌─────────────┐ ┌──────────┐ ┌──────────────┐
     │ Music Agent │ │  Order   │ │   Account    │
     │             │ │  Agent   │ │   Agent      │
     └──────┬──────┘ └────┬─────┘ └──────┬───────┘
            │              │              │
     ┌──────▼──────┐ ┌────▼─────┐ ┌──────▼───────┐
     │ Music Tools │ │Order Tools│ │Account Tools │
     └──────┬──────┘ └────┬─────┘ └──────┬───────┘
            │              │              │
            │              │        ┌─────▼──────┐
            │              │        │  HITL Gate  │ ← interrupt()
            │              │        │ (approve?)  │
            │              │        └─────┬──────┘
            │              │              │
            └──────────────┴──────────────┘
                           │
                    ┌──────▼──────┐
                    │    END      │
                    └─────────────┘
```

**Key LangGraph features showcased:**
- **Cycles**: Each agent ↔ tools loop (agent calls tool → gets result → decides if done or needs another tool call)
- **Human-in-the-loop**: `interrupt()` before executing profile updates
- **Conditional routing**: Router classifies intent and dispatches to the right sub-agent
- **Guardrail node**: Basic moderation before routing

---

## Project Structure

```
sql-support-bot/
├── agent.ipynb            # ★ Primary build — everything lives here
├── agent/
│   ├── __init__.py        # Re-exports the compiled graph for Studio
│   ├── graph.py           # Thin wrapper that builds the same graph for Studio
│   ├── state.py           # State schema (shared by notebook + Studio)
│   └── db.py              # Database setup (shared by notebook + Studio)
├── langgraph.json         # LangGraph Studio configuration
├── pyproject.toml         # Dependencies
└── .env                   # OPENAI_API_KEY, LANGSMITH_API_KEY, etc.
```

**Approach:** Everything is built and demoed in the notebook. The `agent/` module
is a thin re-export layer so LangGraph Studio can point at `agent/graph.py:graph`.

---

## Implementation Steps

| # | Task | Status | Notes |
|---|------|--------|-------|
| 1 | Set up project structure | ✅ | `agent/` stubs, `langgraph.json`, deps |
| 2 | Database & schema exploration | ✅ | Load Chinook, inspect tables in notebook |
| 3 | State definition | ✅ | Extend `MessagesState` with `current_agent` + `customer_id` |
| 4 | Tools | ✅ | 6 tools with InjectedState for customer_id filtering |
| 5 | Prompts | ✅ | System prompts for router + 3 agents + guardrail |
| 6 | Nodes & routing | ✅ | Router, 3 agent nodes, guardrail, HITL via interrupt() |
| 7 | Graph wiring & compile | ✅ | Connected with cycles, checkpointer, conditional edges |
| 8 | LangSmith tracing | ✅ | Env vars set in notebook cell 1 |
| 9 | Demo conversations | ✅ | 5 test scenarios in notebook |
| 10 | Studio export | ✅ | Full graph in `agent/graph.py`, `langgraph.json` configured |
| 11 | Polish & test | ✅ | Notebook runs end-to-end, Studio loads graph |

---

## Customer Data Isolation Strategy

- Customer ID is passed via LangGraph's **`config`** (i.e. `configurable: {"customer_id": 5}`)
- Every tool that touches customer data reads `customer_id` from config — the user never gets to specify someone else's ID
- The Account agent's prompt tells it to never reveal other customers' data

---

## Human-in-the-Loop Strategy

When the Account Agent calls `update_my_profile`:
1. The graph **interrupts** before executing the update
2. The human reviewer sees: "Customer wants to change email from X to Y. Approve?"
3. If approved → execute the SQL update → confirm to user
4. If rejected → tell user the update was denied

This uses LangGraph's `interrupt()` function — clean, built-in, and very demo-friendly.

---

## Demo Scenarios (for the 25-30 min presentation)

1. **"What rock albums do you have?"** → Music agent searches catalog, returns results
2. **"What have I purchased recently?"** → Order agent looks up invoices (filtered to logged-in customer)
3. **"Can you update my email to newemail@example.com?"** → Account agent triggers HITL → approve → done
4. **"Show me my profile"** → Account agent returns only the authenticated customer's data
5. **Try an off-topic request** → Guardrail catches it or router politely redirects
