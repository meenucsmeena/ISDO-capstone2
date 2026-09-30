"""
ISDO Lab C4 - Resolution / KB Agent
Searches ChromaDB ('isdo_kb') for matching KB articles and drafts a resolution.
HIGH confidence on a non-P1 L1 issue -> auto-resolve.
Anything else -> Human-in-the-Loop (HITL) flag.

Run from the project root (C:\\ISDO Batch 2):
    python agents/resolution_agent.py

Needs ANTHROPIC_API_KEY in .env (optional ISDO_MODEL to override the model).
"""

import json
import os
import sys
from pathlib import Path

import anthropic
import chromadb
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

MODEL = os.environ.get("ISDO_MODEL", "claude-opus-5")
KB_DIR = ROOT / "data" / "kb"
MAX_TURNS = 4

# Confidence thresholds (Lab C4, Step 3) - applied in code, not left to the model
HIGH_THRESHOLD = 0.60
MEDIUM_THRESHOLD = 0.35
# Priorities allowed to auto-resolve. Lab Step 3 says P3/P4 only; the expected results
# and the KB articles also allow P2 (VPN, password reset). P1 never auto-resolves.
AUTO_RESOLVE_PRIORITIES = {"P2", "P3", "P4"}

if not os.environ.get("ANTHROPIC_API_KEY"):
    sys.exit("ANTHROPIC_API_KEY not set. Add it to .env in the project root.")

client = anthropic.Anthropic()

# ── CHROMADB KNOWLEDGE BASE (same articles and chunking as Lab C1) ────────────


def chunk_article(text: str) -> list[tuple[str, str]]:
    """Split markdown at '## ' headings -> [(heading, chunk_text), ...]."""
    chunks, heading, lines = [], "Introduction", []
    for line in text.splitlines():
        if line.startswith("## ") and lines:
            chunks.append((heading, "\n".join(lines).strip()))
            heading, lines = line[3:].strip(), []
        lines.append(line)
    if lines:
        chunks.append((heading, "\n".join(lines).strip()))
    return chunks


def build_kb():
    """(Re)build the 'isdo_kb' collection with COSINE distance so 1 - distance
    is a real 0-1 similarity score the thresholds can use."""
    db = chromadb.Client()
    try:
        db.delete_collection("isdo_kb")
    except Exception:
        pass
    kb = db.create_collection("isdo_kb", metadata={"hnsw:space": "cosine"})

    docs, ids, metas = [], [], []
    files = sorted(KB_DIR.glob("*.md"))
    if not files:
        sys.exit(f"No KB articles found in {KB_DIR} - copy the Lab C1 data/kb folder here.")
    for md in files:
        text = md.read_text(encoding="utf-8")
        title = text.splitlines()[0].lstrip("# ").strip()
        for i, (heading, chunk) in enumerate(chunk_article(text)):
            docs.append(f"{title}\n\n{chunk}")  # keep article context in every chunk
            ids.append(f"{md.stem}_{i}")
            metas.append({"filename": md.name, "heading": heading})
    kb.add(documents=docs, ids=ids, metadatas=metas)
    print(f"KB loaded: {len(docs)} chunks from {len(files)} articles (collection 'isdo_kb')")
    return kb


KB = build_kb()


def confidence_level(score: float) -> str:
    if score > HIGH_THRESHOLD:
        return "HIGH"
    if score > MEDIUM_THRESHOLD:
        return "MEDIUM"
    return "LOW"


# ── TOOL DEFINITIONS ──────────────────────────────────────────────────────────

TOOLS = [
    {
        "name": "search_kb",
        "description": "Search the IT knowledge base. Returns the top 2 matching KB articles "
                       "(full text) with confidence scores (1 - cosine distance).",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string",
                                     "description": "The ticket's summary and description text"}},
            "required": ["query"],
        },
    },
    {
        "name": "draft_resolution",
        "description": "Record the resolution for the ticket, based on the matched KB article.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticket_number": {"type": "string"},
                "resolution_text": {
                    "type": "string",
                    "description": "3-4 numbered steps taken from the KB article, written for "
                                   "the requester. For LOW confidence: what L2 should check.",
                },
                "auto_resolve": {
                    "type": "boolean",
                    "description": "True only if the KB article says this issue is L1 "
                                   "auto-resolvable for this situation",
                },
                "confidence": {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"],
                               "description": "The confidence_level returned by search_kb"},
                "kb_article_used": {"type": "string",
                                    "description": "KB filename used, or 'none'"},
            },
            "required": ["ticket_number", "resolution_text", "auto_resolve",
                         "confidence", "kb_article_used"],
        },
    },
]

# ── TOOL IMPLEMENTATIONS ──────────────────────────────────────────────────────


def search_kb(query: str) -> dict:
    """Top 2 articles by best-matching chunk; returns full article text so the
    model sees the Resolution Steps even if the best chunk was 'Symptoms'."""
    raw = KB.query(query_texts=[query], n_results=10)
    best = {}
    for meta, dist in zip(raw["metadatas"][0], raw["distances"][0]):
        f = meta["filename"]
        best[f] = min(dist, best.get(f, 2.0))
    articles = []
    for fname, dist in sorted(best.items(), key=lambda kv: kv[1])[:2]:
        score = round(max(0.0, 1.0 - dist), 2)
        articles.append({"article": fname, "confidence_score": score,
                         "content": (KB_DIR / fname).read_text(encoding="utf-8")})
    top = articles[0]["confidence_score"] if articles else 0.0
    return {"query": query, "confidence_level": confidence_level(top), "articles": articles}


# ── RESOLUTION AGENT ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are the ISDO Resolution Agent for Zensar's IT Service Desk.

For each ticket:
1. Call search_kb ONCE, using the ticket summary and details as the query.
   Do not rephrase and search again - a LOW result is a valid outcome.
2. Call draft_resolution ONCE.
   - confidence: use the confidence_level returned by search_kb.
   - resolution_text: 3-4 numbered, specific steps copied from the matched
     article's Resolution Steps (not generic advice). If confidence is LOW,
     write what L2 should investigate instead.
   - auto_resolve: true only if the article's "Auto-Resolve Eligibility" section
     says this situation is L1 auto-resolvable. Multi-user outages and P1
     tickets are never auto-resolvable.
3. Finish with one short sentence."""


def call_claude(messages):
    kwargs = dict(model=MODEL, max_tokens=1500, system=SYSTEM_PROMPT,
                  tools=TOOLS, messages=messages, temperature=0.0)
    try:
        return client.messages.create(**kwargs)
    except anthropic.BadRequestError as e:
        if "temperature" not in str(e):
            raise
        kwargs.pop("temperature")
        return client.messages.create(**kwargs)


def apply_guardrails(draft: dict, kb_level: str, priority: str) -> tuple[dict, list]:
    """Enforce the HITL rules in code. Returns (final decision, reasons for HITL)."""
    final = dict(draft)
    final["confidence"] = kb_level            # the score decides, not the model
    reasons = []
    if kb_level != "HIGH":
        reasons.append(f"{kb_level} KB confidence")
    if priority not in AUTO_RESOLVE_PRIORITIES:
        reasons.append(f"{priority} ticket")
    if not draft.get("auto_resolve"):
        reasons.append("KB article says not L1 auto-resolvable")
    final["auto_resolve"] = not reasons
    return final, reasons


def resolve_ticket(ticket_number, short_description, description, category, priority) -> dict:
    """Run the agent on one ticket (input = Lab C3 triage output). Returns the decision."""
    print(f"\n{'=' * 55}\nResolving: {ticket_number} | Category: {category} | Priority: {priority}")
    print(f"{'=' * 55}\nIssue: {short_description}")

    messages = [{"role": "user", "content":
                 f"Find a resolution for this ticket:\n\nTicket: {ticket_number}\n"
                 f"Category: {category}\nPriority: {priority}\n"
                 f"Summary: {short_description}\nDetails: {description}"}]
    kb_level, draft = "LOW", None
    turns = 0

    while True:                                   # agentic loop until end_turn
        turns += 1
        if turns > MAX_TURNS:
            print(f"  !! Stopped after {MAX_TURNS} turns")
            break
        response = call_claude(messages)
        if response.stop_reason != "tool_use":
            if response.stop_reason != "end_turn":
                print(f"  !! Unexpected stop_reason: {response.stop_reason}")
            break

        messages.append({"role": "assistant", "content": response.content})
        results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            if block.name == "search_kb":
                result = search_kb(block.input["query"])
                kb_level = result["confidence_level"]
                print(f"  -> KB search: '{' '.join(block.input['query'].split())[:70]}'")
                for a in result["articles"]:
                    print(f"     [{a['confidence_score']:.0%}] {a['article']}")
            elif block.name == "draft_resolution":
                draft = dict(block.input)
                result = {"status": "recorded"}
            else:
                result = {"error": f"Unknown tool {block.name}"}
            results.append({"type": "tool_result", "tool_use_id": block.id,
                            "content": json.dumps(result)})
        messages.append({"role": "user", "content": results})

    if draft is None:
        draft = {"ticket_number": ticket_number, "auto_resolve": False,
                 "resolution_text": "Agent produced no draft - route to L2.",
                 "kb_article_used": "none"}
    final, reasons = apply_guardrails(draft, kb_level, priority)

    print(f"\n  -> Confidence: {final['confidence']}  |  Auto-resolve: {final['auto_resolve']}")
    print(f"  -> KB Article: {final.get('kb_article_used')}")
    print("\n  RESOLUTION DRAFT:")
    for line in str(final.get("resolution_text", "")).splitlines():
        print(f"    {line}")
    if not final["auto_resolve"]:
        print(f"\n  WARNING HITL FLAG: human review required ({'; '.join(reasons)})")
    return final


# ── RUN ON SAMPLE TICKETS ─────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"ISDO Resolution Agent  |  model: {MODEL}")

    # (number, summary, details, category, priority) - category/priority as from Lab C3 triage
    test_tickets = [
        ("INC0001001", "VPN not connecting after password change",
         "User reports VPN client fails to connect after AD password was reset. "
         "Error: authentication failed.", "Network", "P2"),
        ("INC0001006", "Password reset request",
         "User locked out of AD account after 5 failed attempts. Needs immediate reset.",
         "Access", "P2"),
        ("INC0001002", "Cannot access ERP system - login error",
         "Multiple users in Finance unable to login to SAP. Error code: DBCON_FAIL. "
         "Started 09:00 today.", "Application", "P1"),
        # Step 5 - no KB article covers this: expect LOW + HITL
        ("TEST-004", "Cisco Webex not launching on Mac M2",
         "Cisco Webex not launching on Mac M2", "Software", "P3"),
    ]

    decisions = [resolve_ticket(*t) for t in test_tickets]

    print(f"\n{'=' * 55}\nSUMMARY\n{'=' * 55}")
    print(f"  {'Ticket':<12}{'Confidence':<12}{'Auto':<7}KB article")
    for d in decisions:
        print(f"  {d['ticket_number']:<12}{d['confidence']:<12}"
              f"{str(d['auto_resolve']):<7}{d.get('kb_article_used')}")