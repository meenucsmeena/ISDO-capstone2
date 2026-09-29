"""
ISDO Lab C2 - Mock Jira Service Management REST API (port 5002)

  GET /rest/agile/1.0/board/requests   all requests (?request_type= ?priority= ?status=)
  GET /rest/api/2/issue/<key>          one request, Jira-style nested 'fields'
  PUT /rest/api/2/issue/<key>          update fields in memory
  GET /health                          service status

Run from the project root:  python mcp_server/jira_shim.py
"""

import csv
from pathlib import Path

from flask import Flask, jsonify, request

PORT = 5002
DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "requests.csv"
FILTERS = ["request_type", "priority", "assignee", "status"]

app = Flask(__name__)


def load_requests() -> dict:
    """Load requests.csv into an in-memory dict keyed by issue key."""
    if not DATA_FILE.exists():
        print(f"WARNING: {DATA_FILE} not found - starting with no requests")
        return {}
    with open(DATA_FILE, newline="", encoding="utf-8") as f:
        return {row["key"]: row for row in csv.DictReader(f)}


REQUESTS = load_requests()


def to_jira(req: dict) -> dict:
    """Convert a flat CSV row into Jira's nested issue format."""
    return {
        "key": req["key"],
        "fields": {
            "summary": req.get("summary"),
            "issuetype": {"name": req.get("request_type")},
            "priority": {"name": req.get("priority")},
            "status": {"name": req.get("status")},
            "assignee": {"displayName": req.get("assignee")},
            "customfield_sla": req.get("sla"),
        },
    }


@app.get("/rest/agile/1.0/board/requests")
def list_requests():
    results = list(REQUESTS.values())
    for field in FILTERS:
        value = request.args.get(field)
        if value:
            results = [r for r in results if r.get(field, "").lower() == value.lower()]
    return jsonify({"issues": results, "total": len(results)})


@app.get("/rest/api/2/issue/<key>")
def get_issue(key):
    if key not in REQUESTS:
        return jsonify({"errorMessages": [f"Issue {key} does not exist"]}), 404
    return jsonify(to_jira(REQUESTS[key]))


@app.put("/rest/api/2/issue/<key>")
def update_issue(key):
    if key not in REQUESTS:
        return jsonify({"errorMessages": [f"Issue {key} does not exist"]}), 404
    body = request.get_json(silent=True) or {}
    fields = body.get("fields", body)  # accept Jira-style {"fields": {...}} or flat
    if not fields:
        return jsonify({"errorMessages": ["No update body provided"]}), 400
    fields.pop("key", None)
    REQUESTS[key].update(fields)
    print(f"[Jira Mock] Updated {key}: {fields}")
    return jsonify({"key": key, "message": "Updated successfully"})


@app.get("/health")
def health():
    return jsonify({"status": "ok", "service": "Jira Mock", "requests_loaded": len(REQUESTS)})


if __name__ == "__main__":
    print(f"Jira Mock API starting on http://localhost:{PORT}")
    print(f"Loaded {len(REQUESTS)} requests from data/requests.csv")
    app.run(port=PORT, debug=False)