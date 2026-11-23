"""Agentic workflow engine: tool registry, DAG planning and execution."""
import ast
import operator
import re
from collections import deque


class WorkflowError(Exception):
    """Raised when a workflow cannot be planned or executed."""


POSITIVE = {"good", "great", "excellent", "happy", "love", "fast", "thanks", "resolved"}
NEGATIVE = {"bad", "slow", "broken", "angry", "hate", "error", "refund", "outage", "down"}
STOP = {"the", "a", "an", "and", "or", "is", "are", "to", "of", "in", "it", "for", "on", "we", "our"}
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg}


def _words(text):
    return re.findall(r"[a-z']+", text.lower())


def summarize(text, sentences=2):
    """Extractive summary: top sentences ranked by word frequency."""
    parts = [s.strip() for s in re.split(r"(?<=[.!?])\s+", str(text)) if s.strip()]
    freq = {}
    for w in _words(str(text)):
        if w not in STOP:
            freq[w] = freq.get(w, 0) + 1
    ranked = sorted(range(len(parts)),
                    key=lambda i: -sum(freq.get(w, 0) for w in _words(parts[i])))
    keep = sorted(ranked[: max(1, int(sentences))])
    return " ".join(parts[i] for i in keep)


def sentiment(text):
    """Lexicon sentiment returning label and score in [-1, 1]."""
    words = _words(str(text))
    score = sum(w in POSITIVE for w in words) - sum(w in NEGATIVE for w in words)
    norm = max(-1.0, min(1.0, score / max(len(words), 1) * 5))
    label = "positive" if norm > 0.1 else "negative" if norm < -0.1 else "neutral"
    return {"label": label, "score": round(norm, 3)}


def extract_entities(text):
    """Extract emails, URLs and numbers using regular expressions."""
    text = str(text)
    return {"emails": re.findall(r"[\w.+-]+@[\w-]+\.[\w.]+", text),
            "urls": re.findall(r"https?://[^\s]+", text),
            "numbers": re.findall(r"\b\d+(?:\.\d+)?\b", text)}


def route_ticket(text):
    """Classify a support ticket into a team and priority."""
    words = set(_words(str(text)))
    team = ("billing" if words & {"refund", "invoice", "charge", "payment"}
            else "engineering" if words & {"error", "bug", "down", "outage", "broken"}
            else "support")
    priority = "high" if words & {"outage", "down", "urgent", "angry"} else "normal"
    return {"team": team, "priority": priority}


def calc(expression):
    """Safely evaluate arithmetic (no eval). Supports + - * / ** and parentheses."""
    def walk(node):
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            right = walk(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise WorkflowError("exponent too large")
            return _OPS[type(node.op)](walk(node.left), right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](walk(node.operand))
        raise WorkflowError("unsupported expression")
    try:
        return walk(ast.parse(str(expression), mode="eval"))
    except ZeroDivisionError as exc:
        raise WorkflowError("division by zero") from exc
    except SyntaxError as exc:
        raise WorkflowError("invalid expression") from exc


def fill_template(template, **values):
    """Substitute {name} placeholders; unknown names raise WorkflowError."""
    try:
        return str(template).format(**values)
    except (KeyError, IndexError, ValueError, AttributeError) as exc:
        raise WorkflowError(f"template error: {exc}") from exc


TOOLS = {
    "summarize": lambda p: summarize(p["text"], p.get("sentences", 2)),
    "sentiment": lambda p: sentiment(p["text"]),
    "extract_entities": lambda p: extract_entities(p["text"]),
    "route_ticket": lambda p: route_ticket(p["text"]),
    "calc": lambda p: calc(p["expression"]),
    "template": lambda p: fill_template(p["template"], **p.get("values", {})),
}


def plan(steps):
    """Return step ids in dependency order (Kahn's algorithm).

    Raises WorkflowError if the graph contains a cycle.
    """
    indegree = {s["id"]: len(s.get("depends_on", [])) for s in steps}
    children = {s["id"]: [] for s in steps}
    for s in steps:
        for dep in s.get("depends_on", []):
            children[dep].append(s["id"])
    queue = deque(i for i, d in indegree.items() if d == 0)
    order = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for child in children[node]:
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    if len(order) != len(steps):
        raise WorkflowError("workflow contains a dependency cycle")
    return order


def _resolve(value, outputs):
    """Replace '$step_id' / '$step_id.key' references with earlier outputs."""
    if isinstance(value, str) and value.startswith("$"):
        ref, _, key = value[1:].partition(".")
        if ref not in outputs:
            raise WorkflowError(f"unresolved reference {value}")
        out = outputs[ref]
        if key:
            if not isinstance(out, dict) or key not in out:
                raise WorkflowError(f"unresolved reference {value}")
            return out[key]
        return out
    if isinstance(value, dict):
        return {k: _resolve(v, outputs) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve(v, outputs) for v in value]
    return value


def execute(steps, retries=3, tools=None):
    """Run a validated workflow; return (status, outputs, log).

    Transient errors (KeyError/TypeError/ValueError) are retried up to
    `retries` times; WorkflowError is deterministic and fails immediately.
    A failure marks all remaining steps as skipped.
    """
    tools = tools or TOOLS
    by_id = {s["id"]: s for s in steps}
    outputs, log = {}, []
    failed = False
    for sid in plan(steps):
        step = by_id[sid]
        if failed:
            log.append({"step": sid, "status": "skipped"})
            continue
        last_error = None
        for attempt in range(1, retries + 1):
            try:
                params = _resolve(step.get("params", {}), outputs)
                outputs[sid] = tools[step["tool"]](params)
                log.append({"step": sid, "status": "ok", "attempts": attempt})
                last_error = None
                break
            except WorkflowError as exc:
                last_error = str(exc)
                break
            except (KeyError, TypeError, ValueError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
        if last_error:
            failed = True
            log.append({"step": sid, "status": "failed", "error": last_error})
    return ("failed" if failed else "success"), outputs, log
