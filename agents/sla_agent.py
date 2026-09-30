"""
ISDO Lab C5 - SLA & Escalation Agent
Checks SLA breach risk, escalates CRITICAL/BREACHED P1/P2 tickets, and pauses at a
Human-in-the-Loop (HITL) gate before ANY P1 escalation.

Run from the project root (C:\\ISDO Batch 2):
    python agents/sla_agent.py

Needs ANTHROPIC_API_KEY in .env (optional ISDO_MODEL to override the model).
If the Lab C2 ServiceNow shim is running on :5001, update_ticket sends a real PATCH to it;
otherwise the update is simulated.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import anthropic
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

MODEL = os.environ.get("ISDO_MODEL", "claude-opus-5")
SNOW_URL = "http://localhost:5001/api/now/table/incident"
AUDIT_LOG = ROOT / "logs" / "hitl_audit.jsonl"
MAX_TURNS = 6

NOW = datetime(2024, 1, 15, 10, 30)                       # simulated 'now' for the demo
SLA_MINUTES = {"P1": 60, "P2": 240, "P3": 480, "P4": 1440}
ESCALATE_RISKS = {"BREACHED", "CRITICAL"}
ESCALATE_PRIORITIES = {"P1", "P2"}                        # P3/P4 are monitored only
HITL_PRIORITIES = {"P1"}                                  # human must approve these

if not os.environ.get("ANTHROPIC_API_KEY"):
    sys.exit("ANTHROPIC_API_KEY not set. Add it to .env in the project root.")

client = anthropic.Anthropic()

# ── TOOL DEFINITIONS ──────────────────────────────────────────────────────────

TOOLS = [
    {
        "name": "get_sla_status",
        "description": "Check a ticket's SLA: minutes remaining, breach_risk "
                       "(BREACHED/CRITICAL/AT_RISK/ON_TRACK) and requires_escalation.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticket_number": {"type": "string"},
                "sla_due": {"type": "string", "description": "YYYY-MM-DD HH:MM:SS"},
                "priority": {"type": "string", "enum": ["P1", "P2", "P3", "P4"]},
            },
            "required": ["ticket_number", "sla_due", "priority"],
        },
    },
    {
        "name": "update_ticket",
        "description": "Update a ticket in ServiceNow: escalate it, add a work note, or change state.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticket_number": {"type": "string"},
                "action": {"type": "string", "enum": ["escalate", "add_note", "update_state"]},
                "escalation_team": {"type": "string",
                                    "description": "Required for escalate, e.g. L2-Network-Ops"},
                "note": {"type": "string", "description": "Work note text"},
                "new_state": {"type": "string",
                              "description": "e.g. In Progress, Escalated, Resolved"},
            },
            "required": ["ticket_number", "action"],
        },
    },
]

# ── TOOL IMPLEMENTATIONS ──────────────────────────────────────────────────────


def get_sla_status(ticket_number: str, sla_due: str, priority: str) -> dict:
    """Minutes remaining vs the priority's SLA target -> breach risk level."""
    try:
        due = datetime.strptime(sla_due.strip(), "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return {"error": f"Invalid sla_due '{sla_due}', expected YYYY-MM-DD HH:MM:SS"}
    if priority not in SLA_MINUTES:
        return {"error": f"Unknown priority '{priority}'"}

    target = SLA_MINUTES[priority]
    remaining = int((due - NOW).total_seconds() // 60)
    pct_left = remaining / target

    if remaining < 0:
        risk, msg = "BREACHED", f"SLA breached {abs(remaining)} minutes ago"
    elif pct_left < 0.2:
        risk, msg = "CRITICAL", f"Only {remaining} minutes remaining - breach imminent"
    elif pct_left < 0.5:
        risk, msg = "AT_RISK", f"{remaining} minutes remaining - at risk"
    else:
        risk, msg = "ON_TRACK", f"{remaining} minutes remaining - on track"

    return {
        "ticket_number": ticket_number, "priority": priority, "sla_due": sla_due,
        "sla_target_minutes": target, "minutes_remaining": remaining,
        "percent_time_left": round(max(pct_left, 0) * 100),
        "breach_risk": risk, "status_message": msg,
        "requires_escalation": risk in ESCALATE_RISKS and priority in ESCALATE_PRIORITIES,
    }


def update_ticket(ticket_number, action, escalation_team=None, note=None, new_state=None) -> dict:
    """ServiceNow update: real PATCH to the Lab C2 shim if it's running, else simulated."""
    if action == "escalate":
        if not escalation_team:
            return {"success": False, "error": "escalation_team is required for escalate"}
        body = {"state": "Escalated", "assignment_group": escalation_team,
                "work_notes": f"Escalated to {escalation_team} by ISDO SLA Agent"}
        label = f"ESCALATED {ticket_number} -> {escalation_team}"
    elif action == "add_note":
        body, label = {"work_notes": note or ""}, f"NOTE ADDED to {ticket_number}"
    elif action == "update_state":
        body, label = {"state": new_state or ""}, f"STATE CHANGED {ticket_number} -> {new_state}"
    else:
        return {"success": False, "error": f"Unknown action '{action}'"}

    try:
        r = requests.patch(f"{SNOW_URL}/{ticket_number}", json=body, timeout=3)
        mode = "ServiceNow shim" if r.ok else f"simulated (shim returned {r.status_code})"
    except requests.RequestException:
        mode = "simulated (shim not running)"

    print(f"  [ServiceNow Mock] {label}   ({mode})")
    return {"success": True, "ticket_number": ticket_number, "action": action,
            "update": body, "via": mode, "timestamp": datetime.now().isoformat(timespec="seconds")}


# ── HITL GATE ─────────────────────────────────────────────────────────────────


def hitl_approve(ticket_number: str, action: str, detail: str) -> bool:
    """Pause for a human decision. Anything but 'y' (or no terminal) = rejected."""
    print("\n  !!! !!! !!!  HITL APPROVAL REQUIRED")
    print(f"  Ticket:  {ticket_number}")
    print(f"  Action:  {action}")
    print(f"  Detail:  {detail}")
    print("  !!! !!! !!!")
    try:
        answer = input("  Approve escalation? [y/n]: ").strip().lower()
    except EOFError:
        answer = ""
    approved = answer == "y"
    print(f"  Decision: {'APPROVED' if approved else 'REJECTED'}")
    log_decision(ticket_number, detail, approved)
    return approved


def log_decision(ticket_number: str, detail: str, approved: bool) -> None:
    """Append every HITL decision to logs/hitl_audit.jsonl (audit trail for Lab C7)."""
    AUDIT_LOG.parent.mkdir(exist_ok=True)
    entry = {"timestamp": datetime.now().isoformat(timespec="seconds"),
             "ticket": ticket_number, "proposed": detail,
             "decision": "APPROVED" if approved else "REJECTED"}
    with open(AUDIT_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


# ── SLA AGENT ─────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are the ISDO SLA & Escalation Agent for Zensar's IT Service Desk.

For each ticket:
1. Call get_sla_status once.
2. If requires_escalation is true, call update_ticket with action "escalate" and the
   right escalation_team. Otherwise do NOT escalate (P3/P4 and ON_TRACK/AT_RISK
   tickets are monitored only); you may add a short work note instead.
3. If an escalation is rejected by the human approver, do not retry it. Add a work
   note saying escalation was rejected and needs manual follow-up.
4. Finish with one sentence summarising what you did.

Escalation teams by category:
- Network -> L2-Network-Ops
- Application -> L2-App-Support
- Server -> L2-Server-Ops
- Access / Security -> L2-Security-Ops
- Email -> L2-Email-Support
- Anything else (Hardware, Software, ...) -> L2-Service-Desk"""


def call_claude(messages):
    kwargs = dict(model=MODEL, max_tokens=1024, system=SYSTEM_PROMPT,
                  tools=TOOLS, messages=messages, temperature=0.0)
    try:
        return client.messages.create(**kwargs)
    except anthropic.BadRequestError as e:
        if "temperature" not in str(e):
            raise
        kwargs.pop("temperature")
        return client.messages.create(**kwargs)


def run_tool(block, ticket: dict, state: dict) -> dict:
    """Execute one tool call with the guardrails enforced in code."""
    inp = block.input
    if block.name == "get_sla_status":
        result = get_sla_status(inp["ticket_number"], inp["sla_due"], inp["priority"])
        if "breach_risk" in result:
            state["sla"] = result
            print(f"  -> Risk Level: {result['breach_risk']}")
            print(f"  -> Status:     {result['status_message']}")
        return result

    if block.name == "update_ticket":
        if inp.get("action") == "escalate":
            sla = state.get("sla")
            # Guardrail 1: only escalate what the SLA rules say needs it
            if not sla or not sla["requires_escalation"]:
                print("  -> Escalation blocked: SLA rules do not require escalation")
                return {"success": False,
                        "error": "Escalation not allowed: call get_sla_status first; only "
                                 "CRITICAL/BREACHED P1/P2 tickets are escalated."}
            # Guardrail 2: HITL gate - decided by the ticket's real priority, not by the model
            if ticket["priority"] in HITL_PRIORITIES:
                team = inp.get("escalation_team", "L2 team")
                if not hitl_approve(ticket["number"], "Escalate ticket", f"Escalate to {team}"):
                    state["decision"] = "REJECTED"
                    print("  Escalation cancelled.")
                    return {"success": False, "message": "Escalation REJECTED by human approver"}
                state["decision"] = "APPROVED"
            else:
                state["decision"] = "AUTO"
        result = update_ticket(inp["ticket_number"], inp["action"], inp.get("escalation_team"),
                               inp.get("note"), inp.get("new_state"))
        if inp.get("action") == "escalate" and result.get("success"):
            state["escalated_to"] = inp.get("escalation_team")
        return result

    return {"error": f"Unknown tool {block.name}"}


def monitor_ticket(ticket: dict) -> dict:
    """Run the SLA agent on one ticket. Returns a summary dict for reporting / Lab C6."""
    print(f"\n{'=' * 55}")
    print(f"SLA Check: {ticket['number']} | {ticket['priority']} | Category: {ticket['category']}")
    print(f"{'=' * 55}")
    messages = [{"role": "user", "content":
                 "Monitor SLA for this ticket and escalate if needed:\n\n"
                 f"Ticket: {ticket['number']}\nDescription: {ticket['short_description']}\n"
                 f"Category: {ticket['category']}\nPriority: {ticket['priority']}\n"
                 f"SLA Due: {ticket['sla_due']}"}]
    state = {"sla": None, "decision": "NONE", "escalated_to": None}
    turns = 0

    while True:                                   # agentic loop until end_turn
        turns += 1
        if turns > MAX_TURNS:
            print(f"  !! Stopped after {MAX_TURNS} turns")
            break
        response = call_claude(messages)
        if response.stop_reason != "tool_use":
            for b in response.content:
                if b.type == "text" and b.text.strip():
                    print(f"  Agent: {b.text.strip()}")
            if response.stop_reason != "end_turn":
                print(f"  !! Unexpected stop_reason: {response.stop_reason}")
            break
        messages.append({"role": "assistant", "content": response.content})
        results = [{"type": "tool_result", "tool_use_id": b.id,
                    "content": json.dumps(run_tool(b, ticket, state))}
                   for b in response.content if b.type == "tool_use"]
        messages.append({"role": "user", "content": results})

    sla = state["sla"] or {}
    return {"ticket": ticket["number"], "priority": ticket["priority"],
            "breach_risk": sla.get("breach_risk", "UNKNOWN"),
            "minutes_remaining": sla.get("minutes_remaining"),
            "decision": state["decision"], "escalated_to": state["escalated_to"]}


# ── RUN SLA MONITORING ────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"ISDO SLA Agent  |  model: {MODEL}  |  simulated now: {NOW:%Y-%m-%d %H:%M}")

    test_tickets = [
        # NOTE: the lab doc's times (11:00 / 14:00) give 50% and 87% time left, which are
        # ON_TRACK under the Step 1 rules. Times below produce the Step 4 demo results.
        # P1, 10 of 60 min left (17%) -> CRITICAL -> HITL prompt (type y)
        {"number": "INC0001002", "short_description": "Cannot access ERP - SAP login failure",
         "category": "Application", "priority": "P1", "sla_due": "2024-01-15 10:40:00"},
        # P1, due 09:30 -> BREACHED -> HITL prompt (type n)
        {"number": "INC0001010", "short_description": "Exchange server high CPU",
         "category": "Server", "priority": "P1", "sla_due": "2024-01-15 09:30:00"},
        # P2, 60 of 240 min left (25%) -> AT_RISK, monitored (no HITL for P2)
        # Step 5: change sla_due to "2024-01-15 10:00:00" -> BREACHED -> auto-escalates
        {"number": "INC0001001", "short_description": "VPN not connecting",
         "category": "Network", "priority": "P2", "sla_due": "2024-01-15 11:30:00"},
        # P3, due in 2 days -> ON_TRACK, monitored only
        {"number": "INC0001003", "short_description": "Laptop running slowly",
         "category": "Hardware", "priority": "P3", "sla_due": "2024-01-17 09:00:00"},
    ]

    summary = [monitor_ticket(t) for t in test_tickets]

    print(f"\n{'=' * 55}\nSLA SUMMARY\n{'=' * 55}")
    print(f"  {'Ticket':<12}{'Pri':<5}{'Risk':<10}{'Min left':<10}{'Decision':<10}Escalated to")
    for s in summary:
        print(f"  {s['ticket']:<12}{s['priority']:<5}{s['breach_risk']:<10}"
              f"{str(s['minutes_remaining']):<10}{s['decision']:<10}{s['escalated_to'] or '-'}")
    print(f"\n  HITL decisions logged to {AUDIT_LOG.relative_to(ROOT)}")