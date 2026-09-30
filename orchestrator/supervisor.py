"""
ISDO Lab C6 + C7 + C8 - LangGraph Orchestrator (Supervisor) with HITL gate and A2A
Routes a ticket through the ISDO agents built in Labs C3-C5 using a LangGraph StateGraph:

    triage -> resolution -> sla --(hitl_required?)--> hitl -> communication -> END
                                  \\-----------(no)----------> communication -> END

HITL triggers (Lab C7):
  1. P1 ticket with SLA CRITICAL or BREACHED          -> approve escalation
  2. Access grant (category Access + request_type 'Access Grant') -> approve access
  3. Resolution confidence LOW (any priority)         -> approve hand-off to L2

A2A (Lab C8): when ChromaDB confidence is LOW, resolution_node asks the Knowledge
Specialist agent (a2a/knowledge_specialist.py on :8001) for a deeper answer. If it
comes back MEDIUM/HIGH, the LOW-confidence HITL gate is no longer needed. If the A2A
server is down or also LOW, the ticket still goes to the HITL gate.

Run from the project root (C:\\ISDO Batch 2):
    python orchestrator/supervisor.py                    # all test tickets
    python orchestrator/supervisor.py REQ-1002           # just one (or several) tickets

Needs: .env with ANTHROPIC_API_KEY, the agents/ folder from Labs C3-C5,
data/kb/ (Lab C1), and ideally the Lab C2 shims (:5001, :5002) and the Lab C8
Knowledge Specialist (uvicorn a2a.knowledge_specialist:app --port 8001) running.
"""

import operator
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Annotated, TypedDict

import anthropic
import requests
from langgraph.graph import END, START, StateGraph

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agents"))

# Reuse the agents built in Labs C3, C4 and C5 - one tested implementation each
import triage_agent       # noqa: E402  Lab C3: triage_ticket()
import resolution_agent   # noqa: E402  Lab C4: resolve_ticket()  (loads ChromaDB 'isdo_kb')
import sla_agent          # noqa: E402  Lab C5: get_sla_status(), update_ticket(), log_decision()

MODEL = os.environ.get("ISDO_MODEL", "claude-opus-5")
JIRA_URL = "http://localhost:5002/rest/api/2/issue"
A2A_URL = "http://localhost:8001"          # Lab C8 Knowledge Specialist
A2A_TIMEOUT = 90                           # the specialist calls Claude inside POST /tasks
client = anthropic.Anthropic()

ESCALATION_TEAMS = {"Network": "L2-Network-Ops", "Application": "L2-App-Support",
                    "Server": "L2-Server-Ops", "Access": "L2-Security-Ops",
                    "Email": "L2-Email-Support"}

# HITL trigger codes -> (label shown to the approver, proposed action)
HITL_TRIGGERS = {
    "P1_SLA":       ("P1 SLA {risk}", "Escalate to {team}"),
    "ACCESS_GRANT": ("ACCESS GRANT -- access request requires security approval",
                     "Grant the requested access"),
    "LOW_CONF":     ("LOW KB CONFIDENCE -- no reliable KB fix found",
                     "Hand off to {team} for manual investigation"),
}


# ── SHARED STATE ──────────────────────────────────────────────────────────────

class TicketState(TypedDict, total=False):
    # input
    ticket_number: str
    short_description: str
    description: str
    category: str
    priority: str
    sla_due: str
    request_type: str             # Jira request type, e.g. 'Access Grant' (C7)
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
    kb_source: str                # 'chromadb' or 'a2a' (C8)
    a2a_status: str               # not_needed / used / unavailable / error (C8)
    # SLA agent
    sla_breach_risk: str
    escalation_required: bool
    hitl_required: bool
    hitl_reason: str              # human-readable reason(s) the gate fired (C7)
    hitl_triggers: list           # trigger codes: P1_SLA / ACCESS_GRANT / LOW_CONF (C7)
    # HITL node
    hitl_approved: bool
    # Communication agent
    user_message: str
    final_status: str
    # every node appends; operator.add merges the lists instead of overwriting
    audit_log: Annotated[list, operator.add]


def audit(agent: str, action: str, detail: str) -> list:
    """One audit entry (returned as a list so LangGraph appends it to audit_log)."""
    print(f"  [AUDIT] {agent}: {action} -- {detail}")
    return [{"timestamp": datetime.now().isoformat(timespec="seconds"),
             "agent": agent, "action": action, "detail": detail}]


def header(title: str) -> None:
    print(f"\n▶ {title}")


def is_request(state: TicketState) -> bool:
    return state["ticket_number"].upper().startswith("REQ-")


def escalation_team(state: TicketState) -> str:
    return ESCALATION_TEAMS.get(state.get("triage_category") or state.get("category"),
                                "L2-Service-Desk")


def update_record(state: TicketState, action: str, **kw) -> None:
    """INC tickets -> ServiceNow shim (via Lab C5 update_ticket); REQ tickets -> Jira shim."""
    if not is_request(state):
        sla_agent.update_ticket(state["ticket_number"], action, **kw)
        return
    fields = {"escalate": {"status": "Escalated", "assignee": kw.get("escalation_team", "")},
              "update_state": {"status": kw.get("new_state", "")},
              "add_note": {"comment": kw.get("note", "")}}[action]
    try:
        r = requests.put(f"{JIRA_URL}/{state['ticket_number']}", json={"fields": fields}, timeout=3)
        via = "Jira shim" if r.ok else f"simulated (Jira shim returned {r.status_code})"
    except requests.RequestException:
        via = "simulated (Jira shim not running)"
    print(f"  [Jira Mock] {action.upper()} {state['ticket_number']}: {fields}   ({via})")


def lookup_request_type(key: str) -> str:
    """Fetch a REQ ticket's request type from the Jira shim (Lab C2) if not given."""
    try:
        r = requests.get(f"{JIRA_URL}/{key}", timeout=3)
        return r.json()["fields"]["issuetype"]["name"] if r.ok else ""
    except (requests.RequestException, KeyError, ValueError):
        return ""


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
    updates = {
        "triage_category": result["category"],
        "triage_priority": result["priority"],
        "triage_assignment_group": result["assignment_group"],
        "pii_detected": bool(result["pii_detected"]),
        "audit_log": audit("TriageAgent", "classify_ticket",
                           f"{result['category']} / {result['priority']} -> "
                           f"{result['assignment_group']}, PII={result['pii_detected']}"),
    }
    if is_request(state) and not state.get("request_type"):
        updates["request_type"] = lookup_request_type(state["ticket_number"])
    return updates


def ask_knowledge_specialist(state: TicketState) -> dict:
    """A2A call (Lab C8): POST /tasks -> task_id, then GET /tasks/{task_id} -> result.
    Returns {"status": "used", ...result fields} or {"status": "unavailable"/"error", ...}."""
    query = f"{state['short_description']}. {state['description']}"
    context = f"category={state.get('triage_category')}, priority={state.get('triage_priority')}"
    try:
        r = requests.post(f"{A2A_URL}/tasks", timeout=A2A_TIMEOUT,
                          json={"query": query, "ticket_number": state["ticket_number"],
                                "context": context})
        r.raise_for_status()
        task_id = r.json()["task_id"]
        print(f"  -> A2A task submitted: {task_id} (status: {r.json().get('status')})")

        task = requests.get(f"{A2A_URL}/tasks/{task_id}", timeout=A2A_TIMEOUT)
        task.raise_for_status()
        result = task.json().get("result") or {}
        return {"status": "used", "task_id": task_id,
                "confidence": result.get("confidence", "LOW"),
                "resolution": result.get("resolution", ""),
                "best_match": result.get("best_match", "none"),
                "score": result.get("confidence_score"),
                "escalate_to_l2": result.get("escalate_to_l2", True)}
    except requests.exceptions.ConnectionError:
        return {"status": "unavailable", "error": f"A2A server not running at {A2A_URL}"}
    except (requests.RequestException, KeyError, ValueError) as e:
        return {"status": "error", "error": f"{e.__class__.__name__}: {e}"}


def resolution_node(state: TicketState) -> dict:
    header("RESOLUTION AGENT — searching KB")
    d = resolution_agent.resolve_ticket(state["ticket_number"], state["short_description"],
                                        state["description"], state["triage_category"],
                                        state["triage_priority"])
    print(f"  KB Article: {d.get('kb_article_used')}")
    print(f"  Confidence: {d['confidence']} | Auto-resolve: {d['auto_resolve']}")
    updates = {
        "kb_article": d.get("kb_article_used", "none"),
        "resolution_text": d.get("resolution_text", ""),
        "auto_resolve": bool(d["auto_resolve"]),
        "confidence": d["confidence"],
        "kb_source": "chromadb",
        "a2a_status": "not_needed",
    }
    entries = audit("ResolutionAgent", "search_kb",
                    f"{d.get('kb_article_used')} ({d['confidence']}), "
                    f"auto_resolve={d['auto_resolve']}")

    # ── Lab C8: LOW confidence -> ask the Knowledge Specialist via A2A ──
    if d["confidence"] == "LOW":
        print(f"\n▶ A2A CALL — Knowledge Specialist ({A2A_URL})")
        a2a = ask_knowledge_specialist(state)
        updates["a2a_status"] = a2a["status"]
        if a2a["status"] == "used":
            score = f" ({a2a['score']:.0%})" if isinstance(a2a.get("score"), (int, float)) else ""
            print(f"  -> A2A Confidence: {a2a['confidence']}{score} | Best match: {a2a['best_match']}")
            print(f"  -> Escalate to L2: {a2a['escalate_to_l2']}")
            updates.update({
                "resolution_text": a2a["resolution"] or updates["resolution_text"],
                "confidence": a2a["confidence"],
                "kb_article": a2a["best_match"],
                "kb_source": "a2a",
                "auto_resolve": False,   # A2A answers are for an L2 engineer, never auto-sent
            })
            entries += audit("ResolutionAgent", "a2a_knowledge_specialist",
                             f"task {a2a['task_id']}: {a2a['best_match']} -> "
                             f"{a2a['confidence']}, escalate_to_l2={a2a['escalate_to_l2']}")
        else:
            print(f"  !! A2A unavailable - {a2a['error']}")
            print("  -> Keeping LOW confidence: ticket will go to the HITL gate")
            entries += audit("ResolutionAgent", "a2a_knowledge_specialist",
                             f"{a2a['status'].upper()} - fallback to HITL ({a2a['error']})")

    updates["audit_log"] = entries
    return updates


def sla_node(state: TicketState) -> dict:
    header("SLA AGENT — checking deadline + HITL triggers")
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

    # ── HITL trigger conditions (C6 + C7) ──
    team = escalation_team(state)
    triggers, reasons = [], []
    if escalate and priority == "P1":
        triggers.append("P1_SLA")
    access_category = "Access" in (state.get("category"), state.get("triage_category"))
    if access_category and (state.get("request_type") or "").lower() == "access grant":
        triggers.append("ACCESS_GRANT")
    if state.get("confidence") == "LOW":
        triggers.append("LOW_CONF")
    for t in triggers:
        reasons.append(HITL_TRIGGERS[t][0].format(risk=risk, team=team))
    hitl = bool(triggers)
    reason = "; ".join(reasons)
    if hitl:
        print(f"  HITL required: {reason}")

    entries = audit("SLAAgent", "get_sla_status",
                    f"{risk}; escalation_required={escalate}, hitl_required={hitl}"
                    + (f" ({', '.join(triggers)})" if triggers else ""))

    if escalate and "P1_SLA" not in triggers:  # P2 CRITICAL/BREACHED: escalate, no human gate
        update_record(state, "escalate", escalation_team=team)
        entries += audit("SLAAgent", "update_ticket", f"auto-escalated to {team}")
    return {"sla_breach_risk": risk, "escalation_required": escalate,
            "hitl_required": hitl, "hitl_reason": reason, "hitl_triggers": triggers,
            "audit_log": entries}


def hitl_node(state: TicketState) -> dict:
    header("HITL GATE — human approval required")
    team = escalation_team(state)
    triggers = state.get("hitl_triggers", [])
    actions = [HITL_TRIGGERS[t][1].format(team=team) for t in triggers]

    print("  " + "WARNING " * 8)
    print(f"  Ticket:  {state['ticket_number']} | Priority: {state.get('triage_priority')}")
    print(f"  Reason:  {state.get('hitl_reason')}")
    print(f"  Action:  {' + '.join(actions)}")
    print("  " + "WARNING " * 8)
    try:
        approved = input("  Approve action? [y/n]: ").strip().lower() == "y"
    except EOFError:            # no terminal attached -> safe default is REJECT
        approved = False
    decision = "APPROVED" if approved else "REJECTED"
    print(f"  Decision: {decision}")

    sla_agent.log_decision(state["ticket_number"], f"{state.get('hitl_reason')} -> "
                           f"{' + '.join(actions)}", approved)          # logs/hitl_audit.jsonl
    entries = audit("HITLGate", "approval_decision",
                    f"{decision} -- reason: {state.get('hitl_reason')}")

    if approved:
        if "P1_SLA" in triggers or "LOW_CONF" in triggers:
            update_record(state, "escalate", escalation_team=team)
            entries += audit("HITLGate", "update_ticket", f"escalated to {team}")
        if "ACCESS_GRANT" in triggers:
            update_record(state, "update_state", new_state="Approved")
            entries += audit("HITLGate", "update_ticket", "access grant approved")
    else:
        update_record(state, "add_note",
                      note=f"HITL rejected ({state.get('hitl_reason')}) - pending manual review")
    return {"hitl_approved": approved, "audit_log": entries}


def communication_node(state: TicketState) -> dict:
    header("COMMUNICATION AGENT")
    triggers = state.get("hitl_triggers", [])
    group = state.get("triage_assignment_group")

    if state.get("auto_resolve") and not state.get("hitl_required"):
        status = "RESOLVED"
        kind = ("a self-service resolution message with these steps:\n"
                + state.get("resolution_text", ""))
        update_record(state, "update_state", new_state="Resolved")
    elif state.get("hitl_required") and not state.get("hitl_approved"):
        status = "PENDING_APPROVAL"
        kind = ("a 'pending approval' message: the request is waiting for approval by the "
                "service desk team, no action has been taken yet, and the user will be "
                "updated as soon as a decision is made")
    elif "ACCESS_GRANT" in triggers:
        status = "ACCESS_APPROVED"
        kind = ("an access-grant approval message addressed 'Dear Requester': the access "
                "request was approved by security and access will be provisioned shortly")
    elif "P1_SLA" in triggers:
        status = "ESCALATED"
        kind = ("an escalation confirmation: the ticket was escalated to the L2 team after "
                "manager approval and is being worked on urgently")
    elif "LOW_CONF" in triggers:
        status = "ASSIGNED_L2"
        kind = ("a notification that no standard fix was found, so the ticket has been passed "
                "to a specialist L2 team who will investigate and contact the user")
    elif state.get("escalation_required"):
        status = "ESCALATED"
        kind = "an escalation confirmation: the ticket was escalated to L2"
    elif state.get("kb_source") == "a2a":
        status = "ASSIGNED"
        kind = (f"a notification that a knowledge specialist has prepared a detailed fix and "
                f"the ticket is assigned to {group}, who will apply it and contact the user")
    else:
        status = "ASSIGNED"
        kind = (f"a standard assignment notification: the ticket is assigned to {group}, "
                f"who will contact the user")

    greeting = "Dear Requester" if is_request(state) else "Dear User"
    prompt = (f"Write {kind}\n\nTicket: {state['ticket_number']} - {state['short_description']}\n"
              f"Priority: {state.get('triage_priority')}\n\nRules: address it '{greeting}', max "
              f"120 words, plain text, do not repeat any personal data (names, emails, IDs), "
              f"sign off 'ISDO Service Desk'.")
    message = draft_message(prompt, state, status, greeting)
    print("  USER MESSAGE:\n    " + message.replace("\n", "\n    "))
    print(f"\n  ✅ FINAL STATUS: {status}")
    return {"user_message": message, "final_status": status,
            "audit_log": audit("CommunicationAgent", "draft_message", f"final_status={status}")}


def draft_message(prompt: str, state: TicketState, status: str, greeting: str) -> str:
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
        return (f"{greeting},\n\nYour ticket {state['ticket_number']} "
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

# Simulated 'now' = 2024-01-15 10:30 (Lab C5). SLA times chosen so the risks match the lab.
TEST_TICKETS = [
    # C6 ticket 1 - P2 VPN: HIGH confidence, auto-resolve, no HITL
    {"ticket_number": "INC0001001", "short_description": "VPN not connecting after password change",
     "description": "User reports VPN client fails to connect after AD password was reset. "
                    "Error: authentication failed.",
     "category": "Network", "priority": "P2", "sla_due": "2024-01-15 11:30:00"},
    # C6 ticket 2 / C7 Step 2 - P1 SAP: SLA CRITICAL -> HITL (P1_SLA)
    {"ticket_number": "INC0001002", "short_description": "Cannot access ERP system - login error",
     "description": "Multiple users in Finance unable to login to SAP. Error code: DBCON_FAIL. "
                    "Started 09:00 today.",
     "category": "Application", "priority": "P1", "sla_due": "2024-01-15 10:40:00"},
    # C7 Step 4 - access grant request -> HITL (ACCESS_GRANT) regardless of priority
    {"ticket_number": "REQ-1002", "short_description": "VPN access for new contractor",
     "description": "Contractor needs VPN access. Email: contractor@client.com",
     "category": "Access", "priority": "P2", "sla_due": "2024-01-15 15:00:00",
     "request_type": "Access Grant"},
    # C7 Step 3 / C8 - no KB article covers this -> LOW confidence -> A2A Knowledge Specialist;
    # if A2A is down or also LOW -> HITL (LOW_CONF) even for P3
    # (summary AND description changed - a VPN description would still match the VPN article)
    {"ticket_number": "INC0001016", "short_description":
        "Cisco Webex not launching on MacBook M2 after Sonoma update",
     "description": "Cisco Webex app crashes on launch on a MacBook M2 since the macOS Sonoma "
                    "update. Reinstalling did not help.",
     "category": "Software", "priority": "P3", "sla_due": "2024-01-15 18:00:00"},
]

if __name__ == "__main__":
    wanted = {a.upper() for a in sys.argv[1:]}
    tickets = [t for t in TEST_TICKETS if not wanted or t["ticket_number"].upper() in wanted]
    if not tickets:
        sys.exit(f"No test ticket matches {sorted(wanted)}. "
                 f"Choose from: {[t['ticket_number'] for t in TEST_TICKETS]}")

    graph = build_graph()
    finals = []
    for t in tickets:
        print(f"\n{'═' * 55}\nPROCESSING TICKET: {t['ticket_number']}\n{'═' * 55}")
        finals.append(graph.invoke({**t, "audit_log": []}))

    for f in finals:
        print(f"\n{'═' * 55}\nAUDIT LOG — {f['ticket_number']}  (final: {f['final_status']})\n{'═' * 55}")
        for e in f["audit_log"]:
            print(f"  {e['timestamp']}  {e['agent']:<19}{e['action']:<26}{e['detail']}")

    print(f"\n{'═' * 55}\nSUMMARY\n{'═' * 55}")
    print(f"  {'Ticket':<12}{'Pri':<5}{'Conf':<8}{'A2A':<13}{'HITL':<24}{'Decision':<10}Final")
    for f in finals:
        hitl = ",".join(f.get("hitl_triggers") or []) or "-"
        dec = ("-" if not f.get("hitl_required")
               else "APPROVED" if f.get("hitl_approved") else "REJECTED")
        print(f"  {f['ticket_number']:<12}{f.get('triage_priority', ''):<5}"
              f"{f.get('confidence', ''):<8}{f.get('a2a_status', ''):<13}"
              f"{hitl:<24}{dec:<10}{f['final_status']}")