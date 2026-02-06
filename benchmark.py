"""
Benchmark: LangGraph Workflow vs Deep Agent (Raw SQL)
=====================================================
Runs 6 identical queries through both approaches, tracks them in LangSmith,
and compares latency and token cost side by side.

Usage:
    python benchmark.py

Both approaches use gpt-4o. Traces land in LangSmith project "music-store-benchmark".
"""

from __future__ import annotations

import os
import time
import uuid

from dotenv import load_dotenv

load_dotenv()

# ── LangSmith Config ─────────────────────────────────────────────────────────
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "music-store-benchmark"

from langchain_core.tracers.langchain import wait_for_all_tracers  # noqa: E402
from langsmith import Client  # noqa: E402

from agent.db import db  # noqa: E402

CUSTOMER_ID = 5

# ═════════════════════════════════════════════════════════════════════════════
# Approach 1: LangGraph Workflow  (from agent/graph.py)
# ═════════════════════════════════════════════════════════════════════════════
from agent.graph import graph as workflow_graph  # noqa: E402

# ═════════════════════════════════════════════════════════════════════════════
# Approach 2: Deep Agent + Raw SQL  (2 generic tools)
# ═════════════════════════════════════════════════════════════════════════════
from deepagents import create_deep_agent  # noqa: E402


def get_schema() -> str:
    """Get the database schema — table names and their columns.
    Call this first to understand what data is available before writing queries."""
    return db.get_table_info()


def run_query(sql: str) -> str:
    """Execute a SQL query against the music store database and return results.

    Args:
        sql: A SQL query. Use SELECT for reads. UPDATE only on Customer table.
    """
    sql_upper = sql.strip().upper()

    if any(kw in sql_upper for kw in ("DROP", "DELETE", "ALTER", "CREATE", "INSERT", "TRUNCATE")):
        return "❌ Blocked: Only SELECT and limited UPDATE queries are allowed."
    if sql_upper.startswith("UPDATE") and "CUSTOMER" not in sql_upper:
        return "❌ Blocked: UPDATE is only allowed on the Customer table."

    try:
        result = db.run(sql, include_columns=True)
        return result if result else "Query returned no results."
    except Exception as e:
        return f"❌ SQL Error: {e}"


DEEP_AGENT_PROMPT = f"""\
You are a customer support assistant for a music store with a SQLite database (Chinook).
The logged-in customer has CustomerId = {CUSTOMER_ID}.

You have two custom tools:
- get_schema — returns table names and columns. Call this FIRST.
- run_query  — executes a SQL query.

Rules:
- Always call get_schema before writing SQL if you haven't seen the schema yet.
- Filter customer-specific queries by CustomerId = {CUSTOMER_ID}.
- NEVER expose other customers' data.
- Use LIKE with wildcards for fuzzy search.
- LIMIT results to 20.
- Be concise and friendly.
"""

deep_agent = create_deep_agent(
    model="openai:gpt-4o",
    tools=[get_schema, run_query],
    system_prompt=DEEP_AGENT_PROMPT,
)

# ═════════════════════════════════════════════════════════════════════════════
# Test Queries — same 6 for both approaches
# ═════════════════════════════════════════════════════════════════════════════
QUERIES = [
    "What AC/DC albums do you have?",
    "Recommend some Jazz tracks",
    "What have I purchased recently?",
    "Show me my profile",
    "What are the top 5 genres by number of tracks?",
    "How much have I spent in total?",
]


# ═════════════════════════════════════════════════════════════════════════════
# Run Benchmark
# ═════════════════════════════════════════════════════════════════════════════
def run_benchmark():
    results: list[dict] = []

    print("\n🏁 Starting benchmark: 6 queries × 2 approaches\n")

    for i, query in enumerate(QUERIES, 1):
        print(f"{'─'*60}")
        print(f"  Q{i}: {query}")
        print(f"{'─'*60}")

        # ── Workflow ──────────────────────────────────────────────────
        w_run_id = str(uuid.uuid4())
        t0 = time.time()
        try:
            w_result = workflow_graph.invoke(
                {
                    "messages": [{"role": "user", "content": query}],
                    "customer_id": CUSTOMER_ID,
                },
                config={"run_id": w_run_id, "run_name": f"workflow_q{i}"},
            )
            w_time = time.time() - t0
            w_answer = w_result["messages"][-1].content[:80]
        except Exception as e:
            w_time = time.time() - t0
            w_answer = f"ERROR: {e}"[:80]

        print(f"  ✅ Workflow:   {w_time:>6.2f}s  →  {w_answer}...")

        # ── Deep Agent ────────────────────────────────────────────────
        d_run_id = str(uuid.uuid4())
        t0 = time.time()
        try:
            d_result = deep_agent.invoke(
                {"messages": [{"role": "user", "content": query}]},
                config={"run_id": d_run_id, "run_name": f"deep_q{i}"},
            )
            d_time = time.time() - t0
            d_answer = d_result["messages"][-1].content[:80]
        except Exception as e:
            d_time = time.time() - t0
            d_answer = f"ERROR: {e}"[:80]

        print(f"  ✅ Deep Agent: {d_time:>6.2f}s  →  {d_answer}...")

        results.append({
            "i": i,
            "query": query,
            "w_time": w_time,
            "d_time": d_time,
            "w_run_id": w_run_id,
            "d_run_id": d_run_id,
        })

    # ── Flush traces to LangSmith ────────────────────────────────────────
    print("\n⏳ Flushing traces to LangSmith...")
    wait_for_all_tracers()
    time.sleep(5)  # give LangSmith a moment to process

    # ── Fetch token / cost data from LangSmith ───────────────────────────
    client = Client()

    for r in results:
        for prefix, run_id_key in [("w", "w_run_id"), ("d", "d_run_id")]:
            try:
                run = client.read_run(r[run_id_key])
                r[f"{prefix}_in_tokens"] = run.input_tokens or 0
                r[f"{prefix}_out_tokens"] = run.output_tokens or 0
                r[f"{prefix}_total_tokens"] = r[f"{prefix}_in_tokens"] + r[f"{prefix}_out_tokens"]
                r[f"{prefix}_cost"] = (run.input_cost or 0) + (run.output_cost or 0)
            except Exception:
                r[f"{prefix}_in_tokens"] = 0
                r[f"{prefix}_out_tokens"] = 0
                r[f"{prefix}_total_tokens"] = 0
                r[f"{prefix}_cost"] = 0

    # ── Print Results Table ──────────────────────────────────────────────
    print("\n")
    print("=" * 110)
    print(f"{'':>4} {'Query':<40}  {'Workflow':^28}  {'Deep Agent':^28}  {'Winner':^8}")
    print(f"{'':>4} {'':40}  {'Time':>8} {'Tokens':>8} {'Cost':>10}  {'Time':>8} {'Tokens':>8} {'Cost':>10}  {'':^8}")
    print("=" * 110)

    totals = {"w_time": 0, "d_time": 0, "w_total_tokens": 0, "d_total_tokens": 0, "w_cost": 0, "d_cost": 0}

    for r in results:
        q = r["query"][:38] + ".." if len(r["query"]) > 40 else r["query"]
        winner = "⚡ WF" if r["w_time"] < r["d_time"] else "⚡ DA"

        print(
            f"  Q{r['i']} {q:<40}"
            f"  {r['w_time']:>7.1f}s {r['w_total_tokens']:>7,} ${r['w_cost']:>8.4f}"
            f"  {r['d_time']:>7.1f}s {r['d_total_tokens']:>7,} ${r['d_cost']:>8.4f}"
            f"  {winner:^8}"
        )

        for key in totals:
            totals[key] += r[key]

    print("─" * 110)
    print(
        f"     {'TOTAL':<40}"
        f"  {totals['w_time']:>7.1f}s {totals['w_total_tokens']:>7,} ${totals['w_cost']:>8.4f}"
        f"  {totals['d_time']:>7.1f}s {totals['d_total_tokens']:>7,} ${totals['d_cost']:>8.4f}"
        f"  {'⚡ WF' if totals['w_time'] < totals['d_time'] else '⚡ DA':^8}"
    )
    print("=" * 110)

    # ── Summary ──────────────────────────────────────────────────────────
    speedup = totals["d_time"] / totals["w_time"] if totals["w_time"] > 0 else 0
    token_ratio = totals["d_total_tokens"] / totals["w_total_tokens"] if totals["w_total_tokens"] > 0 else 0

    print(f"\n📊 Summary:")
    print(f"   Workflow total:   {totals['w_time']:.1f}s | {totals['w_total_tokens']:,} tokens | ${totals['w_cost']:.4f}")
    print(f"   Deep Agent total: {totals['d_time']:.1f}s | {totals['d_total_tokens']:,} tokens | ${totals['d_cost']:.4f}")
    print(f"   Time ratio:       Deep Agent is {speedup:.1f}x {'slower' if speedup > 1 else 'faster'} than Workflow")
    print(f"   Token ratio:      Deep Agent uses {token_ratio:.1f}x {'more' if token_ratio > 1 else 'fewer'} tokens")
    print(f"\n🔍 View full traces in LangSmith → project: 'music-store-benchmark'")


if __name__ == "__main__":
    run_benchmark()
