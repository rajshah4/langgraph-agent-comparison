"""
Benchmark: OpenHands SDK — Structured Tools vs Deep Agent (Raw SQL)
===================================================================
Runs 6 identical queries through both approaches and compares
latency, token usage, and cost side by side.

Usage:
    python oh_benchmark.py

Both approaches use gpt-4o via the OpenHands SDK.
Traces are sent to Laminar.
"""

from __future__ import annotations

import os
import time
from typing import Literal

from dotenv import load_dotenv
from lmnr import Laminar

load_dotenv()
Laminar.initialize()

from agent.db import db  # noqa: E402
from oh_tool_helper import simple_tool, tool_spec  # noqa: E402
from openhands.sdk import (  # noqa: E402
    Agent,
    AgentContext,
    Conversation,
    LLM,
    MessageEvent,
)
from openhands.sdk.event.llm_convertible.action import ActionEvent  # noqa: E402
from openhands.sdk.event.llm_convertible.observation import ObservationEvent  # noqa: E402

CUSTOMER_ID = 5

# ═════════════════════════════════════════════════════════════════════════════
# Approach 1: Structured Tools  (6 pre-built SQL tools — LLM never writes SQL)
# ═════════════════════════════════════════════════════════════════════════════


@simple_tool
def search_catalog(query: str, search_by: Literal["artist", "song", "album"] = "artist") -> str:
    """Search the music catalog by artist, song title, or album name.

    Args:
        query: The search term (artist name, song title, or album name).
        search_by: What to search by: 'artist', 'song', or 'album'.
    """
    q = query.replace("'", "''")
    if search_by == "artist":
        sql = (
            f"SELECT DISTINCT Artist.Name AS Artist, Album.Title AS Album "
            f"FROM Artist JOIN Album ON Album.ArtistId = Artist.ArtistId "
            f"WHERE Artist.Name LIKE '%{q}%' LIMIT 20;"
        )
    elif search_by == "song":
        sql = (
            f"SELECT Track.Name AS Song, Artist.Name AS Artist, Album.Title AS Album "
            f"FROM Track JOIN Album ON Track.AlbumId = Album.AlbumId "
            f"JOIN Artist ON Album.ArtistId = Artist.ArtistId "
            f"WHERE Track.Name LIKE '%{q}%' LIMIT 20;"
        )
    else:
        sql = (
            f"SELECT Album.Title AS Album, Artist.Name AS Artist "
            f"FROM Album JOIN Artist ON Album.ArtistId = Artist.ArtistId "
            f"WHERE Album.Title LIKE '%{q}%' LIMIT 20;"
        )
    result = db.run(sql, include_columns=True)
    return str(result) if result else "No results found."


@simple_tool
def get_recommendations(genre: str) -> str:
    """Get popular tracks in a given genre.

    Args:
        genre: The genre name (e.g. 'Rock', 'Jazz', 'Blues', 'Pop', 'Latin').
    """
    g = genre.replace("'", "''")
    sql = (
        f"SELECT Track.Name AS Song, Artist.Name AS Artist, Genre.Name AS Genre "
        f"FROM Track JOIN Album ON Track.AlbumId = Album.AlbumId "
        f"JOIN Artist ON Album.ArtistId = Artist.ArtistId "
        f"JOIN Genre ON Track.GenreId = Genre.GenreId "
        f"WHERE Genre.Name LIKE '%{g}%' ORDER BY RANDOM() LIMIT 10;"
    )
    result = db.run(sql, include_columns=True)
    return str(result) if result else f"No tracks found for genre '{genre}'."


@simple_tool
def get_purchase_history() -> str:
    """View the customer's recent purchases and invoices."""
    sql = (
        f"SELECT InvoiceId, InvoiceDate, Total, BillingCity, BillingCountry "
        f"FROM Invoice WHERE CustomerId = {CUSTOMER_ID} "
        f"ORDER BY InvoiceDate DESC LIMIT 10;"
    )
    result = db.run(sql, include_columns=True)
    return str(result) if result else "No purchases found."


@simple_tool
def get_invoice_details(invoice_id: int) -> str:
    """Get the line-item details (songs purchased) for a specific invoice.

    Args:
        invoice_id: The invoice number to look up.
    """
    sql = (
        f"SELECT Track.Name AS Song, Artist.Name AS Artist, "
        f"InvoiceLine.UnitPrice, InvoiceLine.Quantity "
        f"FROM InvoiceLine "
        f"JOIN Invoice ON InvoiceLine.InvoiceId = Invoice.InvoiceId "
        f"JOIN Track ON InvoiceLine.TrackId = Track.TrackId "
        f"JOIN Album ON Track.AlbumId = Album.AlbumId "
        f"JOIN Artist ON Album.ArtistId = Artist.ArtistId "
        f"WHERE InvoiceLine.InvoiceId = {int(invoice_id)} "
        f"AND Invoice.CustomerId = {CUSTOMER_ID};"
    )
    result = db.run(sql, include_columns=True)
    return str(result) if result else "Invoice not found."


@simple_tool
def get_profile() -> str:
    """View the customer's current profile information (name, email, phone, address)."""
    sql = (
        f"SELECT FirstName, LastName, Email, Phone, Address, City, State, "
        f"Country, PostalCode, Company "
        f"FROM Customer WHERE CustomerId = {CUSTOMER_ID};"
    )
    result = db.run(sql, include_columns=True)
    return str(result) if result else "Profile not found."


@simple_tool(read_only=False, destructive=True)
def update_profile(field: str, new_value: str) -> str:
    """Update a profile field. Requires manager approval.

    Args:
        field: The profile field to update.
        new_value: The new value for the field.
    """
    safe = new_value.replace("'", "''")
    db.run(f'UPDATE Customer SET "{field}" = \'{safe}\' WHERE CustomerId = {CUSTOMER_ID};')
    return f"Updated {field} to '{new_value}'."


# ═════════════════════════════════════════════════════════════════════════════
# Approach 2: Deep Agent + Raw SQL  (2 generic tools — LLM writes its own SQL)
# ═════════════════════════════════════════════════════════════════════════════

DANGEROUS_KW = ["DROP", "DELETE", "TRUNCATE", "INSERT", "ALTER", "CREATE"]


@simple_tool
def get_schema() -> str:
    """Get the database schema - table names, columns, and sample data."""
    return db.get_table_info()


@simple_tool(read_only=False)
def run_sql(query: str) -> str:
    """Execute a SQL query against the music store database.

    Args:
        query: The SQL query to execute. Use SELECT for reads, UPDATE only on Customer.
    """
    sql = query.strip()
    sql_upper = sql.upper()
    for kw in DANGEROUS_KW:
        if kw in sql_upper:
            return f"Blocked: dangerous keyword '{kw}'."
    if sql_upper.startswith("UPDATE") and "CUSTOMER" not in sql_upper:
        return "Blocked: UPDATE only allowed on Customer table."
    result = db.run(sql, include_columns=True)
    return str(result) if result else "Query returned no results."


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
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

STRUCTURED_TEMPLATE = os.path.abspath("oh_prompts/structured_agent.j2")
DEEP_TEMPLATE = os.path.abspath("oh_prompts/deep_agent.j2")

STRUCTURED_CONTEXT = (
    f"The currently logged-in customer has CustomerId = {CUSTOMER_ID}.\n"
    f"All order and profile queries are automatically scoped to this customer.\n"
)
DEEP_CONTEXT = (
    f"The currently logged-in customer has CustomerId = {CUSTOMER_ID}.\n"
    f"ALWAYS filter customer-specific queries by CustomerId = {CUSTOMER_ID}.\n"
    f"NEVER expose other customers' data.\n"
)


def get_response(conversation: Conversation) -> str:
    """Extract the last agent response from a conversation."""
    events = list(conversation.state.events)
    for event in reversed(events):
        if isinstance(event, MessageEvent) and event.source == "agent":
            return "".join(c.text for c in event.llm_message.content if hasattr(c, "text"))
    # Fallback: FinishTool observation
    for i in range(len(events) - 1, -1, -1):
        if isinstance(events[i], ActionEvent) and events[i].tool_name == "finish":
            if i + 1 < len(events) and isinstance(events[i + 1], ObservationEvent):
                obs = events[i + 1].observation
                if obs and hasattr(obs, "content"):
                    return "".join(c.text for c in obs.content if hasattr(c, "text"))
    return "(no response)"


def snapshot_metrics(llm: LLM) -> tuple[float, int, int]:
    """Return (cost, prompt_tokens, completion_tokens) snapshot."""
    usage = llm.metrics.accumulated_token_usage
    prompt = usage.prompt_tokens if usage else 0
    completion = usage.completion_tokens if usage else 0
    return llm.metrics.accumulated_cost, prompt, completion


def run_single_query(
    llm: LLM,
    agent: Agent,
    query: str,
) -> dict:
    """Run one query in a fresh conversation, return timing + token stats."""
    cost0, prompt0, comp0 = snapshot_metrics(llm)

    conv = Conversation(agent=agent, workspace=os.getcwd(), visualizer=None)
    t0 = time.time()
    try:
        conv.send_message(query)
        conv.run()
        elapsed = time.time() - t0
        answer = get_response(conv)[:80]
    except Exception as e:
        elapsed = time.time() - t0
        answer = f"ERROR: {e}"[:80]
    finally:
        conv.close()

    cost1, prompt1, comp1 = snapshot_metrics(llm)
    return {
        "time": elapsed,
        "answer": answer,
        "prompt_tokens": prompt1 - prompt0,
        "completion_tokens": comp1 - comp0,
        "total_tokens": (prompt1 - prompt0) + (comp1 - comp0),
        "cost": cost1 - cost0,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Main Benchmark
# ═════════════════════════════════════════════════════════════════════════════

def run_benchmark():
    # ── Create separate LLM instances so metrics don't overlap ────────────
    api_key = os.getenv("OPENAI_API_KEY")

    llm_s = LLM(model="openai/gpt-4o", api_key=api_key)
    llm_d = LLM(model="openai/gpt-4o", api_key=api_key)

    agent_s = Agent(
        llm=llm_s,
        tools=[
            tool_spec(search_catalog),
            tool_spec(get_recommendations),
            tool_spec(get_purchase_history),
            tool_spec(get_invoice_details),
            tool_spec(get_profile),
            tool_spec(update_profile),
        ],
        system_prompt_filename=STRUCTURED_TEMPLATE,
        agent_context=AgentContext(skills=[], system_message_suffix=STRUCTURED_CONTEXT),
    )

    agent_d = Agent(
        llm=llm_d,
        tools=[tool_spec(get_schema), tool_spec(run_sql)],
        system_prompt_filename=DEEP_TEMPLATE,
        agent_context=AgentContext(skills=[], system_message_suffix=DEEP_CONTEXT),
    )

    results: list[dict] = []

    print("\nStarting benchmark: 6 queries x 2 approaches (OpenHands SDK)\n")

    for i, query in enumerate(QUERIES, 1):
        print(f"{'─' * 60}")
        print(f"  Q{i}: {query}")
        print(f"{'─' * 60}")

        # ── Structured ──────────────────────────────────────────────
        s = run_single_query(llm_s, agent_s, query)
        print(f"  Structured: {s['time']:>6.2f}s  {s['total_tokens']:>6,} tok  ${s['cost']:.4f}  ->  {s['answer']}...")

        # ── Deep Agent ──────────────────────────────────────────────
        d = run_single_query(llm_d, agent_d, query)
        print(f"  Deep Agent: {d['time']:>6.2f}s  {d['total_tokens']:>6,} tok  ${d['cost']:.4f}  ->  {d['answer']}...")

        results.append({"i": i, "query": query, "s": s, "d": d})

    # ── Print Results Table ────────────────────────────────────────────────
    print("\n")
    W = 115
    print("=" * W)
    print(f"{'':>4} {'Query':<40}  {'Structured':^28}  {'Deep Agent':^28}  {'Winner':^8}")
    print(f"{'':>4} {'':40}  {'Time':>8} {'Tokens':>8} {'Cost':>10}  {'Time':>8} {'Tokens':>8} {'Cost':>10}  {'':^8}")
    print("=" * W)

    tot = {"s_time": 0, "d_time": 0, "s_tok": 0, "d_tok": 0, "s_cost": 0, "d_cost": 0}

    for r in results:
        s, d = r["s"], r["d"]
        q = r["query"][:38] + ".." if len(r["query"]) > 40 else r["query"]
        winner = "ST" if s["time"] < d["time"] else "DA"

        print(
            f"  Q{r['i']} {q:<40}"
            f"  {s['time']:>7.1f}s {s['total_tokens']:>7,} ${s['cost']:>8.4f}"
            f"  {d['time']:>7.1f}s {d['total_tokens']:>7,} ${d['cost']:>8.4f}"
            f"  {'<-- ' + winner:>8}"
        )

        tot["s_time"] += s["time"]
        tot["d_time"] += d["time"]
        tot["s_tok"] += s["total_tokens"]
        tot["d_tok"] += d["total_tokens"]
        tot["s_cost"] += s["cost"]
        tot["d_cost"] += d["cost"]

    print("─" * W)
    winner = "ST" if tot["s_time"] < tot["d_time"] else "DA"
    print(
        f"     {'TOTAL':<40}"
        f"  {tot['s_time']:>7.1f}s {tot['s_tok']:>7,} ${tot['s_cost']:>8.4f}"
        f"  {tot['d_time']:>7.1f}s {tot['d_tok']:>7,} ${tot['d_cost']:>8.4f}"
        f"  {'<-- ' + winner:>8}"
    )
    print("=" * W)

    # ── Summary ────────────────────────────────────────────────────────────
    time_ratio = tot["d_time"] / tot["s_time"] if tot["s_time"] > 0 else 0
    tok_ratio = tot["d_tok"] / tot["s_tok"] if tot["s_tok"] > 0 else 0
    cost_ratio = tot["d_cost"] / tot["s_cost"] if tot["s_cost"] > 0 else 0

    print(f"\nSummary:")
    print(f"  Structured total:  {tot['s_time']:.1f}s | {tot['s_tok']:,} tokens | ${tot['s_cost']:.4f}")
    print(f"  Deep Agent total:  {tot['d_time']:.1f}s | {tot['d_tok']:,} tokens | ${tot['d_cost']:.4f}")
    print(f"  Time ratio:        Deep Agent is {time_ratio:.1f}x {'slower' if time_ratio > 1 else 'faster'}")
    print(f"  Token ratio:       Deep Agent uses {tok_ratio:.1f}x {'more' if tok_ratio > 1 else 'fewer'} tokens")
    print(f"  Cost ratio:        Deep Agent costs {cost_ratio:.1f}x {'more' if cost_ratio > 1 else 'less'}")
    print(f"\nView traces in Laminar.")


if __name__ == "__main__":
    run_benchmark()
