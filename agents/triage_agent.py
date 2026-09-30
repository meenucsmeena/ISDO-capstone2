"""
ISDO Lab C3 - Triage Agent
Reads an IT ticket and assigns category, priority, assignment group and a PII flag
using the Anthropic SDK tool-calling API and an agentic (ReAct) loop.

Run from the project root (C:\\ISDO Batch 2):
    python agents/triage_agent.py

Needs ANTHROPIC_API_KEY in a .env file in the project root.
Optional: ISDO_MODEL in .env to override the model name.
"""

import csv
import json
import os
import sys
from pathlib import Path

import anthropic
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

MODEL = os.environ.get("ISDO_MODEL", "claude-opus-5")
INCIDENTS_CSV = ROOT / "data" / "incidents.csv"
MAX_TURNS = 5  # safety stop for the agentic loop

if not os.environ.get("ANTHROPIC_API_KEY"):
    sys.exit("ANTHROPIC_API_KEY not set. Add it to .env in the project root:\n"
             "  ANTHROPIC_API_KEY=sk-ant-...")

client = anthropic.Anthropic()

# ── TOOL DEFINITIONS ──────────────────────────────────────────────────────────

TOOLS = [
    {
        "name": "classify_ticket",
        "description": "Classify an IT support ticket. Records category, priority, "
                       "assignment_group, whether PII was detected, and the reasoning.",
        "input_schema": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "enum": ["Network", "Application", "Hardware", "Access",
                             "Email", "Server", "Software"],
                    "description": "The ticket category",
                },
                "priority": {
                    "type": "string",
                    "enum": ["P1", "P2", "P3", "P4"],
                    "description": "P1=Critical/many users, P2=High/one department, "
                                   "P3=Medium/single user, P4=Low/request",
                },
                "assignment_group": {
                    "type": "string",
                    "enum": ["Network-Ops", "App-Support", "Desktop-Support", "Service-Desk",
                             "Security-Ops", "Server-Ops", "Email-Support", "DBA-Team"],
                    "description": "Team that should receive the ticket",
                },
                "pii_detected": {
                    "type": "boolean",
                    "description": "True if the text contains personal names, email addresses, "
                                   "employee IDs, phone numbers or IP addresses",
                },
                "reasoning": {
                    "type": "string",
                    "description": "One sentence explaining the classification decision",
                },
            },
            "required": ["category", "priority", "assignment_group", "pii_detected", "reasoning"],
        },
    },
    {
        "name": "get_open_tickets",
        "description": "Get the number of currently Open incidents grouped by category. "
                       "Use it to check current workload before classifying.",
        "input_schema": {"type": "object", "properties": {}},
    },
]

# ── TOOL IMPLEMENTATIONS ──────────────────────────────────────────────────────


_warned = False


def load_incidents() -> list:
    """Read data/incidents.csv, skipping malformed rows (wrong column count)."""
    with open(INCIDENTS_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    good = [r for r in rows if None not in r and None not in r.values()]
    global _warned
    if len(good) < len(rows) and not _warned:
        _warned = True
        bad = [r.get("number") for r in rows if r not in good]
        print(f"WARNING: skipped malformed rows in incidents.csv: {bad} (fix the CSV quoting)")
    return good


def get_open_tickets() -> dict:
    """Count Open incidents in data/incidents.csv grouped by category."""
    counts = {}
    for r in load_incidents():
        if r["state"] == "Open":
            counts[r["category"]] = counts.get(r["category"], 0) + 1
    return {"open_by_category": dict(sorted(counts.items())), "total_open": sum(counts.values())}


def handle_tool_call(name: str, tool_input: dict) -> dict:
    """Route a tool call from Claude to its local implementation."""
    if name == "get_open_tickets":
        return get_open_tickets()
    if name == "classify_ticket":
        # The structured input IS the classification - acknowledge and return it.
        return {"status": "recorded", **tool_input}
    return {"error": f"Unknown tool: {name}"}


# ── TRIAGE AGENT ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are the ISDO Triage Agent for Zensar's IT Service Desk.

For every ticket you MUST call the classify_ticket tool exactly once to record
category, priority, assignment group and PII flag. Do not answer in free text
instead of calling the tool.

Priority rules:
- P1: Service down, many users / whole site affected, or security breach
- P2: Significant impact on one department or function, or urgent single-user blocker
- P3: Single user impacted, workaround exists
- P4: Request (new software, access, equipment) with no outage

Assignment groups: Network-Ops (network, VPN, Wi-Fi), App-Support (business apps,
ERP, CRM, SharePoint), Desktop-Support (laptops, printers, workstations),
Service-Desk (passwords, accounts, onboarding), Security-Ops (MFA, security),
Server-Ops (servers), Email-Support (email/Outlook), DBA-Team (databases).

PII: flag true for personal names, email addresses, employee IDs, phone numbers
or IP addresses. Placeholders like [REDACTED] are not PII.

After the tool result comes back, reply with one short confirmation line."""


def call_claude(messages):
    """One Messages API call. temperature=0 for repeatable classification."""
    kwargs = dict(model=MODEL, max_tokens=1024, system=SYSTEM_PROMPT,
                  tools=TOOLS, messages=messages, temperature=0.0)
    try:
        return client.messages.create(**kwargs)
    except anthropic.BadRequestError as e:
        if "temperature" not in str(e):
            raise
        kwargs.pop("temperature")  # some newer models fix sampling themselves
        return client.messages.create(**kwargs)


def triage_ticket(ticket_number: str, short_description: str, description: str) -> dict:
    """Run the agentic loop on one ticket and return the classification dict."""
    print(f"\n{'=' * 55}\nTriaging: {ticket_number}\n{'=' * 55}")
    print(f"Description: {short_description}")

    messages = [{"role": "user", "content":
                 f"Please triage this ticket:\n\nTicket: {ticket_number}\n"
                 f"Summary: {short_description}\nDetails: {description}"}]
    classification = {}
    turns = 0

    while True:                                      # Reason -> Act -> Observe loop
        turns += 1
        if turns > MAX_TURNS:                        # safety stop, never loop forever
            print(f"  !! Stopped after {MAX_TURNS} turns without end_turn")
            break
        response = call_claude(messages)

        if response.stop_reason == "end_turn":
            for block in response.content:
                if block.type == "text" and block.text.strip():
                    print(f"  Agent: {block.text.strip()}")
            break
        if response.stop_reason != "tool_use":       # e.g. max_tokens - stop, don't resend
            print(f"  !! Unexpected stop_reason: {response.stop_reason}")
            break

        messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            print(f"  -> Tool called: {block.name}")
            result = handle_tool_call(block.name, block.input)
            if block.name == "classify_ticket":
                classification = dict(block.input, ticket=ticket_number)
                print(f"  -> Category:   {result.get('category')}")
                print(f"  -> Priority:   {result.get('priority')}")
                print(f"  -> Assign To:  {result.get('assignment_group')}")
                print(f"  -> PII Found:  {result.get('pii_detected')}")
                print(f"  -> Reason:     {result.get('reasoning')}")
            else:
                print(f"     Result: {json.dumps(result)}")
            tool_results.append({"type": "tool_result", "tool_use_id": block.id,
                                 "content": json.dumps(result)})
        messages.append({"role": "user", "content": tool_results})

    if not classification:
        print("  !! Agent did not call classify_ticket")
    return classification


# ── RUN ON SAMPLE TICKETS ─────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"ISDO Triage Agent  |  model: {MODEL}")

    # 5 sample tickets taken straight from data/incidents.csv
    SAMPLE_IDS = ["INC0001001", "INC0001002", "INC0001008", "INC0001006", "INC0001004"]
    incidents = {r["number"]: r for r in load_incidents()}
    test_tickets = [(n, incidents[n]["short_description"], incidents[n]["description"])
                    for n in SAMPLE_IDS]

    # Extra tickets from the lab document
    test_tickets += [
        # PII test - contractor emp-id + email
        ("REQ-1002", "VPN access for new contractor joining project Phoenix",
         "New contractor [REDACTED NAME] emp-id ZEN-9823 joining next Monday. "
         "Email: contractor@client.com"),
        # Step 5 - your own ticket
        ("TEST-006", "Salesforce CRM access issue",
         "User cannot access Salesforce CRM from company laptop since this morning."),
        # Step 5 - same issue, many users: watch the priority change
        # ("TEST-007", "Salesforce CRM access issue",
        #  "Entire Sales team (25 users) cannot access Salesforce CRM since this morning."),
    ]

    results = [triage_ticket(*t) for t in test_tickets]

    print(f"\n{'=' * 55}\nSUMMARY\n{'=' * 55}")
    print(f"  {'Ticket':<12}{'Category':<13}{'Pri':<5}{'Assign To':<17}PII")
    for r in results:
        if r:
            print(f"  {r['ticket']:<12}{r['category']:<13}{r['priority']:<5}"
                  f"{r['assignment_group']:<17}{r['pii_detected']}")

    print(f"\n{'=' * 55}\nOPEN TICKET COUNTS BY CATEGORY\n{'=' * 55}")
    summary = get_open_tickets()
    for cat, count in summary["open_by_category"].items():
        print(f"  {cat:<20} {count} open")
    print(f"  {'TOTAL':<20} {summary['total_open']} open")