"""Authentication, rate limiting and input validation."""
import hmac
import re
import time
from collections import defaultdict, deque
from functools import wraps

from flask import current_app, jsonify, request

NAME_RE = re.compile(r"^[A-Za-z0-9_\- ]{1,60}$")
STEP_ID_RE = re.compile(r"^[a-z][a-z0-9_]{0,29}$")
_hits = defaultdict(deque)


class ValidationError(ValueError):
    """Raised when client input fails validation."""


def error(message, status):
    """Build a uniform JSON error response."""
    return jsonify({"error": message}), status


def require_auth(view):
    """Decorator: enforce API key (constant-time) and rate limit."""

    @wraps(view)
    def wrapper(*args, **kwargs):
        supplied = request.headers.get("X-API-Key", "")
        expected = current_app.config["API_KEY"]
        if not hmac.compare_digest(supplied.encode(), expected.encode()):
            return error("invalid or missing API key", 401)
        client = request.remote_addr or "unknown"
        now = time.monotonic()
        window = _hits[client]
        while window and now - window[0] > 60:
            window.popleft()
        if len(window) >= current_app.config["RATE_LIMIT"]:
            return error("rate limit exceeded", 429)
        window.append(now)
        return view(*args, **kwargs)

    return wrapper


def reset_rate_limits():
    """Clear counters (used by tests)."""
    _hits.clear()


def validate_workflow(payload, tools, max_steps):
    """Validate a workflow definition and return (name, steps).

    Raises ValidationError with a human-readable message on any problem.
    """
    if not isinstance(payload, dict):
        raise ValidationError("body must be a JSON object")
    name = payload.get("name")
    if not isinstance(name, str) or not NAME_RE.match(name):
        raise ValidationError("name must be 1-60 letters, digits, space, _ or -")
    steps = payload.get("steps")
    if not isinstance(steps, list) or not 1 <= len(steps) <= max_steps:
        raise ValidationError(f"steps must be a list of 1-{max_steps} items")
    seen = set()
    for step in steps:
        if not isinstance(step, dict):
            raise ValidationError("each step must be an object")
        sid = step.get("id")
        if not isinstance(sid, str) or not STEP_ID_RE.match(sid):
            raise ValidationError("step id must match [a-z][a-z0-9_]{0,29}")
        if sid in seen:
            raise ValidationError(f"duplicate step id: {sid}")
        seen.add(sid)
        if step.get("tool") not in tools:
            raise ValidationError(f"unknown tool in step {sid}")
        if not isinstance(step.get("params", {}), dict):
            raise ValidationError(f"params of step {sid} must be an object")
        deps = step.get("depends_on", [])
        if not isinstance(deps, list) or not all(isinstance(d, str) for d in deps):
            raise ValidationError(f"depends_on of step {sid} must be a list of ids")
    for step in steps:
        for dep in step.get("depends_on", []):
            if dep not in seen:
                raise ValidationError(f"step {step['id']} depends on unknown step {dep}")
    return name, steps
