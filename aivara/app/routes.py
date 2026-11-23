"""REST API: workflows, runs and tool catalogue."""
import json
import sqlite3

from flask import Blueprint, current_app, jsonify, request

from . import engine
from .db import get_db, row_to_dict
from .security import ValidationError, error, require_auth, validate_workflow

api = Blueprint("api", __name__)


@api.get("/health")
def health():
    """Unauthenticated liveness probe."""
    return jsonify({"status": "ok"})


@api.get("/tools")
@require_auth
def list_tools():
    """List available agent tools."""
    return jsonify({"tools": sorted(engine.TOOLS)})


@api.post("/workflows")
@require_auth
def create_workflow():
    """Validate and store a workflow definition."""
    payload = request.get_json(silent=True)
    try:
        name, steps = validate_workflow(payload, engine.TOOLS, current_app.config["MAX_STEPS"])
        engine.plan(steps)
    except (ValidationError, engine.WorkflowError) as exc:
        return error(str(exc), 400)
    db = get_db()
    try:
        cur = db.execute("INSERT INTO workflows (name, definition) VALUES (?, ?)",
                         (name, json.dumps(steps)))
        db.commit()
    except sqlite3.IntegrityError:
        return error("workflow name already exists", 409)
    return jsonify({"id": cur.lastrowid, "name": name}), 201


@api.get("/workflows")
@require_auth
def list_workflows():
    """List stored workflows."""
    rows = get_db().execute("SELECT * FROM workflows ORDER BY id DESC").fetchall()
    return jsonify([row_to_dict(r) for r in rows])


@api.delete("/workflows/<int:wf_id>")
@require_auth
def delete_workflow(wf_id):
    """Delete a workflow and its runs."""
    db = get_db()
    cur = db.execute("DELETE FROM workflows WHERE id = ?", (wf_id,))
    db.commit()
    return ("", 204) if cur.rowcount else error("workflow not found", 404)


@api.post("/workflows/<int:wf_id>/run")
@require_auth
def run_workflow(wf_id):
    """Execute a workflow and persist the run."""
    db = get_db()
    row = db.execute("SELECT * FROM workflows WHERE id = ?", (wf_id,)).fetchone()
    if row is None:
        return error("workflow not found", 404)
    steps = json.loads(row["definition"])
    status, outputs, log = engine.execute(steps, current_app.config["MAX_RETRIES"])
    cur = db.execute("INSERT INTO runs (workflow_id, status, result, log) VALUES (?,?,?,?)",
                     (wf_id, status, json.dumps(outputs), json.dumps(log)))
    db.commit()
    return jsonify({"run_id": cur.lastrowid, "status": status,
                    "result": outputs, "log": log}), 200 if status == "success" else 422


@api.get("/runs")
@require_auth
def list_runs():
    """List the 50 most recent runs."""
    rows = get_db().execute("SELECT * FROM runs ORDER BY id DESC LIMIT 50").fetchall()
    return jsonify([row_to_dict(r) for r in rows])


@api.app_errorhandler(413)
def too_large(_e):
    """Reject oversized bodies."""
    return error("request body too large", 413)
