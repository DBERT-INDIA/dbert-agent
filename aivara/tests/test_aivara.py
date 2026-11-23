"""Unit and integration tests. Run: python -m unittest discover -s tests -v"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app import create_app, engine  # noqa: E402
from app.security import reset_rate_limits  # noqa: E402

KEY = {"X-API-Key": "test-key"}


def wf(name="demo", steps=None):
    """Build a workflow payload."""
    return {"name": name, "steps": steps if steps is not None else [
        {"id": "a", "tool": "sentiment", "params": {"text": "great and fast"}},
        {"id": "b", "tool": "template",
         "params": {"template": "mood={m}", "values": {"m": "$a.label"}},
         "depends_on": ["a"]}]}


class EngineTests(unittest.TestCase):
    """Core algorithm tests."""

    def test_calc_precedence(self):
        self.assertEqual(engine.calc("2 + 3 * (4 - 1)"), 11)

    def test_calc_rejects_code(self):
        with self.assertRaises(engine.WorkflowError):
            engine.calc("__import__('os').system('id')")

    def test_calc_division_by_zero(self):
        with self.assertRaises(engine.WorkflowError):
            engine.calc("1/0")

    def test_calc_huge_exponent_blocked(self):
        with self.assertRaises(engine.WorkflowError):
            engine.calc("9**9999")

    def test_sentiment_labels(self):
        self.assertEqual(engine.sentiment("great excellent")["label"], "positive")
        self.assertEqual(engine.sentiment("broken outage")["label"], "negative")
        self.assertEqual(engine.sentiment("")["label"], "neutral")

    def test_entities(self):
        out = engine.extract_entities("mail a@b.co see https://x.io 42")
        self.assertEqual(out["emails"], ["a@b.co"])
        self.assertEqual(out["numbers"], ["42"])

    def test_route_ticket(self):
        self.assertEqual(engine.route_ticket("refund urgent"),
                         {"team": "billing", "priority": "high"})

    def test_summarize_length(self):
        text = "Cats sleep. Cats eat fish. Dogs bark loudly today."
        self.assertEqual(len(engine.summarize(text, 1).split(". ")), 1)

    def test_plan_order(self):
        steps = wf()["steps"][::-1]
        self.assertEqual(engine.plan(steps), ["a", "b"])

    def test_plan_cycle(self):
        steps = [{"id": "a", "tool": "calc", "depends_on": ["b"]},
                 {"id": "b", "tool": "calc", "depends_on": ["a"]}]
        with self.assertRaises(engine.WorkflowError):
            engine.plan(steps)

    def test_execute_success_and_reference(self):
        status, out, _ = engine.execute(wf()["steps"])
        self.assertEqual((status, out["b"]), ("success", "mood=positive"))

    def test_execute_failure_skips_rest(self):
        steps = [{"id": "a", "tool": "calc", "params": {"expression": "1/0"}},
                 {"id": "b", "tool": "calc", "params": {"expression": "1"},
                  "depends_on": ["a"]}]
        status, _, log = engine.execute(steps)
        self.assertEqual(status, "failed")
        self.assertEqual(log[1]["status"], "skipped")

    def test_retry_on_transient_error(self):
        calls = {"n": 0}

        def flaky(_p):
            calls["n"] += 1
            if calls["n"] < 3:
                raise ValueError("temporary")
            return "ok"
        status, _, log = engine.execute([{"id": "a", "tool": "flaky"}],
                                        retries=3, tools={"flaky": flaky})
        self.assertEqual((status, log[0]["attempts"]), ("success", 3))


class ApiTests(unittest.TestCase):
    """HTTP-level tests including security and validation boundaries."""

    def setUp(self):
        reset_rate_limits()
        fd, self.path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self.app = create_app({"DATABASE": self.path, "API_KEY": "test-key",
                               "TESTING": True, "RATE_LIMIT": 1000})
        self.c = self.app.test_client()

    def tearDown(self):
        os.remove(self.path)

    def test_health_open(self):
        self.assertEqual(self.c.get("/api/health").status_code, 200)

    def test_auth_required(self):
        self.assertEqual(self.c.get("/api/workflows").status_code, 401)
        bad = self.c.get("/api/workflows", headers={"X-API-Key": "nope"})
        self.assertEqual(bad.status_code, 401)

    def test_create_list_run_delete(self):
        r = self.c.post("/api/workflows", json=wf(), headers=KEY)
        self.assertEqual(r.status_code, 201)
        wid = r.get_json()["id"]
        run = self.c.post(f"/api/workflows/{wid}/run", headers=KEY)
        self.assertEqual(run.get_json()["result"]["b"], "mood=positive")
        self.assertEqual(len(self.c.get("/api/runs", headers=KEY).get_json()), 1)
        self.assertEqual(self.c.delete(f"/api/workflows/{wid}", headers=KEY).status_code, 204)
        self.assertEqual(self.c.get("/api/runs", headers=KEY).get_json(), [])

    def test_duplicate_name_conflict(self):
        self.c.post("/api/workflows", json=wf(), headers=KEY)
        self.assertEqual(self.c.post("/api/workflows", json=wf(), headers=KEY).status_code, 409)

    def test_validation_boundaries(self):
        bad = [wf(name=""), wf(name="x" * 61), wf(name="a;DROP TABLE"),
               wf(steps=[]), {"name": "n", "steps": "x"},
               wf(steps=[{"id": "A", "tool": "calc"}]),
               wf(steps=[{"id": "a", "tool": "rm"}]),
               wf(steps=[{"id": "a", "tool": "calc", "depends_on": ["zz"]}]),
               wf(steps=[{"id": "s%d" % i, "tool": "calc"} for i in range(21)])]
        for payload in bad:
            r = self.c.post("/api/workflows", json=payload, headers=KEY)
            self.assertEqual(r.status_code, 400, payload)

    def test_cycle_rejected(self):
        steps = [{"id": "a", "tool": "calc", "depends_on": ["b"]},
                 {"id": "b", "tool": "calc", "depends_on": ["a"]}]
        self.assertEqual(self.c.post("/api/workflows", json=wf(steps=steps),
                                     headers=KEY).status_code, 400)

    def test_non_json_body(self):
        r = self.c.post("/api/workflows", data="x", headers=KEY)
        self.assertEqual(r.status_code, 400)

    def test_oversized_body(self):
        r = self.c.post("/api/workflows", data="x" * 70000, headers=KEY,
                        content_type="application/json")
        self.assertEqual(r.status_code, 413)

    def test_failed_run_returns_422(self):
        steps = [{"id": "a", "tool": "calc", "params": {"expression": "1/0"}}]
        wid = self.c.post("/api/workflows", json=wf("bad", steps), headers=KEY).get_json()["id"]
        r = self.c.post(f"/api/workflows/{wid}/run", headers=KEY)
        self.assertEqual((r.status_code, r.get_json()["status"]), (422, "failed"))

    def test_missing_workflow_404(self):
        self.assertEqual(self.c.post("/api/workflows/99/run", headers=KEY).status_code, 404)

    def test_sql_injection_name_is_rejected(self):
        r = self.c.post("/api/workflows", json=wf(name="x'; DROP TABLE runs;--"), headers=KEY)
        self.assertEqual(r.status_code, 400)

    def test_rate_limit(self):
        self.app.config["RATE_LIMIT"] = 2
        codes = [self.c.get("/api/tools", headers=KEY).status_code for _ in range(3)]
        self.assertEqual(codes, [200, 200, 429])

    def test_security_headers(self):
        self.assertEqual(self.c.get("/api/health").headers["X-Frame-Options"], "DENY")

    def test_dashboard_served(self):
        self.assertIn(b"Aivara", self.c.get("/").data)


if __name__ == "__main__":
    unittest.main()
