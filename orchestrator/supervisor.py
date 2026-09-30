"""
ISDO Lab C6 - LangGraph Orchestrator (Supervisor)
Routes a ticket through the ISDO agents built in Labs C3-C5 using a LangGraph StateGraph:

    triage -> resolution -> sla --(hitl_required?)--> hitl -> communication -> END
                                  \\-----------(no)----------> communication -> END

Run from the project root (C:\\ISDO Batch 2):
    python orchestrator/supervisor.py

Needs: .env with ANTHROPIC_API_KEY, the agents/ folder from Labs C3-C5,
data/kb/ (Lab C1), and ideally the Lab C2 shims running (snow_shim.py on :5001).
"""

import operator
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Annotated, TypedDict

import anthropic
from langgraph.graph import END, START, StateGraph

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agents"))

# Reuse the agents built in Labs C3, C4 and C5 - one tested implementation each
import triage_agent       # noqa: E402  Lab C3: triage_ticket()
import resolution_agent   # noqa: E402  Lab C4: resolve_ticket()  (loads ChromaDB 'isdo_kb')
import sla_agent          # noqa: E402  Lab C5: get_sla_status(), update_ticket()

MODEL = os.environ.get("ISDO_MODEL", "claude-opus-5")
client = anthropic.Anthropic()

ESCALATION_TEAMS = {"Network": "L2-Network-Ops", "Application": "L2-App-Support",
                    "Server": "L2-Server-Ops", "Access": "L2-Security-Ops",
                    "Email": "L2-Email-Support"}


# ── SHARED STATE ──────────────────────────────────────────────────────────────

class TicketState(TypedDict, total=False):
    # input
    ticket_number: str
    short_description: str
    description: str
    category: str
    priority: str
    sla_due: str
    # Triage agent
    triage_category: str
    triage_priority: str
    triage_assignment_group: str
    pii_detected: bool
    # Resolution agent
    kb_article: str
    resolution_text: str
    auto_resolve: bool
    confidence: str
    # SLA agent
    sla_breach_risk: str
    escalation_required: bool
    hitl_required: bool
    # HITL node
    hitl_approved: bool
    # Communication agent
    user_message: str
    final_status: str
    # every node appends; operator.add merges the lists instead of overwriting
    audit_log: Annotated[list, operator.add]


def audit(agent: str, action: str, detail: str) -> list:
    """One audit entry (returned as a list so LangGraph appends it to audit_log)."""
    print(f"  [AUDIT] {agent}: {action}")
    return [{"timestamp": datetime.now().isoformat(timespec="seconds"),
             "agent": agent, "action": action, "detail": detail}]


def header(title: str) -> None:
    print(f"\n▶ {title}")


# ── NODES ─────────────────────────────────────────────────────────────────────

def triage_node(state: TicketState) -> dict:
    header(f"TRIAGE AGENT — {state['ticket_number']}")
    result = triage_agent.triage_ticket(state["ticket_number"], state["short_description"],
                                        state["description"])
    if not result:  # agent failed to classify - fall back to the ticket's own values
        result = {"category": state["category"], "priority": state["priority"],
                  "assignment_group": "Service-Desk", "pii_detected": False}
    print(f"  Category: {result['category']}  Priority: {result['priority']}")
    print(f"  Assign To: {result['assignment_group']}  PII: {result['pii_detected']}")
    return {
        "triage_category": result["category"],
        "triage_priority": result["priority"],
        "triage_assignment_group": result["assignment_group"],
        "pii_detected": bool(result["pii_detected"]),
        "audit_log": audit("TriageAgent", "classify_ticket",
                           f"{result['category']} / {result['priority']} -> "
                           f"{result['assignment_group']}, PII={result['pii_detected']}"),
    }


def resolution_node(state: TicketState) -> dict:
    header("RESOLUTION AGENT — searching KB")
    d = resolution_agent.resolve_ticket(state["ticket_number"], state["short_description"],
                                        state["description"], state["triage_category"],
                                        state["triage_priority"])
    print(f"  KB Article: {d.get('kb_article_used')}")
    print(f"  Confidence: {d['confidence']} | Auto-resolve: {d['auto_resolve']}")
    return {
        "kb_article": d.get("kb_article_used", "none"),
        "resolution_text": d.get("resolution_text", ""),
        "auto_resolve": bool(d["auto_resolve"]),
        "confidence": d["confidence"],
        "audit_log": audit("ResolutionAgent", "search_kb",
                           f"{d.get('kb_article_used')} ({d['confidence']}), "
                           f"auto_resolve={d['auto_resolve']}"),
    }


def sla_node(state: TicketState) -> dict:
    header("SLA AGENT — checking deadline")
    priority = state["triage_priority"]
    s = sla_agent.get_sla_status(state["ticket_number"], state["sla_due"], priority)
    if "error" in s:
        print(f"  !! {s['error']}")
        risk, escalate = "UNKNOWN", False
    else:
        risk, escalate = s["breach_risk"], s["requires_escalation"]
        print(f"  SLA Risk: {risk} | Minutes remaining: {s['minutes_remaining']}")
    # an auto-resolved ticket is being closed, so it isn't escalated
    escalate = escalate and not state.get("auto_resolve", False)
    hitl = escalate and priority == "P1"
    updates = {"sla_breach_risk": risk, "escalation_required": escalate, "hitl_required": hitl}
    entries = audit("SLAAgent", "get_sla_status",
                    f"{risk}; escalation_required={escalate}, hitl_required={hitl}")

    if escalate and not hitl:  # P2 CRITICAL/BREACHED: escalate automatically, no human gate
        team = ESCALATION_TEAMS.get(state["triage_category"], "L2-Service-Desk")
        sla_agent.update_ticket(state["ticket_number"], "escalate", escalation_team=team)
        entries += audit("SLAAgent", "update_ticket", f"auto-escalated to {team}")
    updates["audit_log"] = entries
    return updates


def hitl_node(state: TicketState) -> dict:
    header("HITL GATE — human approval required")
    team = ESCALATION_TEAMS.get(state["triage_category"], "L2-Service-Desk")
    approved = sla_agent.hitl_approve(
        state["ticket_number"], "Escalate ticket",
        f"Escalate {state['triage_priority']} {state['sla_breach_risk']} ticket to {team}")
    entries = audit("HITLGate", "approval", "APPROVED" if approved else "REJECTED")
    if approved:
        sla_agent.update_ticket(state["ticket_number"], "escalate", escalation_team=team)
        entries += audit("SLAAgent", "update_ticket", f"escalated to {team}")
    else:
        sla_agent.update_ticket(state["ticket_number"], "add_note",
                                note="Escalation rejected by human approver - manual follow-up")
    return {"hitl_approved": approved, "audit_log": entries}


def communication_node(state: TicketState) -> dict:
    header("COMMUNICATION AGENT")
    if state.get("auto_resolve"):
        status, kind = "RESOLVED", ("a self-service resolution message with these steps:\n"
                                    + state.get("resolution_text", ""))
        sla_agent.update_ticket(state["ticket_number"], "update_state", new_state="Resolved")
    elif state.get("hitl_approved"):
        status, kind = "ESCALATED", ("an escalation confirmation: the ticket was escalated to "
                                     "the L2 team after manager approval and is being worked "
                                     "on urgently")
    elif state.get("escalation_required") and not state.get("hitl_required"):
        status, kind = "ESCALATED", "an escalation confirmation: the ticket was escalated to L2"
    else:
        status, kind = "ASSIGNED", (f"a standard assignment notification: the ticket is assigned "
                                    f"to {state.get('triage_assignment_group')}, who will contact "
                                    f"the user")

    prompt = (f"Write {kind}\n\nTicket: {state['ticket_number']} - {state['short_description']}\n"
              f"Priority: {state.get('triage_priority')}\n\nRules: address it 'Dear User', max "
              f"120 words, plain text, no personal data, sign off 'ISDO Service Desk'.")
    message = draft_message(prompt, state, status)
    print(f"  USER MESSAGE:\n    " + message.replace("\n", "\n    "))
    print(f"\n  ✅ FINAL STATUS: {status}")
    return {"user_message": message, "final_status": status,
            "audit_log": audit("CommunicationAgent", "draft_message", f"final_status={status}")}


def draft_message(prompt: str, state: TicketState, status: str) -> str:
    """Ask Claude for the user message; fall back to a template if the call fails."""
    try:
        kwargs = dict(model=MODEL, max_tokens=400, temperature=0.0,
                      messages=[{"role": "user", "content": prompt}])
        try:
            r = client.messages.create(**kwargs)
        except anthropic.BadRequestError as e:
            if "temperature" not in str(e):
                raise
            kwargs.pop("temperature")
            r = client.messages.create(**kwargs)
        return "".join(b.text for b in r.content if b.type == "text").strip()
    except anthropic.APIError as e:
        print(f"  !! Claude call failed ({e.__class__.__name__}) - using template")
        return (f"Dear User,\n\nYour ticket {state['ticket_number']} "
                f"({state['short_description']}) is now {status}.\n\nISDO Service Desk")


# ── ROUTING ───────────────────────────────────────────────────────────────────

def route_after_sla(state: TicketState) -> str:
    return "hitl" if state.get("hitl_required") else "communication"


def build_graph():
    g = StateGraph(TicketState)
    g.add_node("triage", triage_node)
    g.add_node("resolution", resolution_node)
    g.add_node("sla", sla_node)
    g.add_node("hitl", hitl_node)
    g.add_node("communication", communication_node)

    g.add_edge(START, "triage")
    g.add_edge("triage", "resolution")
    g.add_edge("resolution", "sla")
    g.add_conditional_edges("sla", route_after_sla,
                            {"hitl": "hitl", "communication": "communication"})
    g.add_edge("hitl", "communication")
    g.add_edge("communication", END)
    return g.compile()


# ── RUN ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    graph = build_graph()

    # Simulated 'now' = 2024-01-15 10:30 (Lab C5). SLA times chosen so the risks match the lab.
    tickets = [
        {"ticket_number": "INC0001001", "short_description": "VPN not connecting after password change",
         "description": "User reports VPN client fails to connect after AD password was reset. "
                        "Error: authentication failed.",
         "category": "Network", "priority": "P2", "sla_due": "2024-01-15 11:30:00"},
        {"ticket_number": "INC0001002", "short_description": "Cannot access ERP system - login error",
         "description": "Multiple users in Finance unable to login to SAP. Error code: DBCON_FAIL. "
                        "Started 09:00 today.",
         "category": "Application", "priority": "P1", "sla_due": "2024-01-15 10:40:00"},
    ]

    finals = []
    for t in tickets:
        print(f"\n{'═' * 55}\nPROCESSING TICKET: {t['ticket_number']}\n{'═' * 55}")
        finals.append(graph.invoke({**t, "audit_log": []}))

    for f in finals:
        print(f"\n{'═' * 55}\nAUDIT LOG — {f['ticket_number']}  (final: {f['final_status']})\n{'═' * 55}")
        for e in f["audit_log"]:
            print(f"  {e['timestamp']}  {e['agent']:<19}{e['action']:<16}{e['detail']}")