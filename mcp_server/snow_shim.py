"""
ISDO Lab C2 - Mock ServiceNow Table API (port 5001)

  GET   /api/now/table/incident            all incidents (?category= ?priority= ?state=)
  GET   /api/now/table/incident/<number>   one incident
  PATCH /api/now/table/incident/<number>   update fields in memory
  GET   /health                            service status

Run from the project root:  python mcp_server/snow_shim.py
"""

import csv
from pathlib import Path

from flask import Flask, jsonify, request

PORT = 5001
DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "incidents.csv"
FILTERS = ["category", "priority", "state", "assignment_group"]

app = Flask(__name__)


def load_incidents() -> dict:
    """Load incidents.csv into an in-memory dict keyed by incident number."""
    if not DATA_FILE.exists():
        print(f"WARNING: {DATA_FILE} not found - starting with no incidents")
        return {}
    with open(DATA_FILE, newline="", encoding="utf-8") as f:
        return {row["number"]: row for row in csv.DictReader(f)}


INCIDENTS = load_incidents()


@app.get("/api/now/table/incident")
def list_incidents():
    results = list(INCIDENTS.values())
    for field in FILTERS:
        value = request.args.get(field)
        if value:
            results = [r for r in results if r.get(field, "").lower() == value.lower()]
    return jsonify({"result": results, "total": len(results)})


@app.get("/api/now/table/incident/<number>")
def get_incident(number):
    incident = INCIDENTS.get(number)
    if incident is None:
        return jsonify({"error": f"Incident {number} not found"}), 404
    return jsonify({"result": incident})


@app.patch("/api/now/table/incident/<number>")
def update_incident(number):
    if number not in INCIDENTS:
        return jsonify({"error": f"Incident {number} not found"}), 404
    updates = request.get_json(silent=True)
    if not updates:
        return jsonify({"error": "Request body must be a non-empty JSON object"}), 400
    updates.pop("number", None)  # the record key cannot be changed
    INCIDENTS[number].update(updates)
    print(f"[ServiceNow Mock] Updated {number}: {updates}")
    return jsonify({"result": INCIDENTS[number], "message": "Updated successfully"})


@app.get("/health")
def health():
    return jsonify({"status": "ok", "service": "ServiceNow Mock",
                    "incidents_loaded": len(INCIDENTS)})


if __name__ == "__main__":
    print(f"ServiceNow Mock API starting on http://localhost:{PORT}")
    print(f"Loaded {len(INCIDENTS)} incidents from data/incidents.csv")
    app.run(port=PORT, debug=False)