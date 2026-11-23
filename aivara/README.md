# Aivara — Autonomous AI Operations & Agentic Workflow Suite

A full stack app that chains small AI "tools" into automated workflows (a DAG),
runs them with retries, and records a step-by-step log.

**Stack:** Python 3.10+ / Flask REST API · SQLite · vanilla HTML/JS dashboard · unittest

## Quick start
```bash
pip install -r requirements.txt
export AIVARA_API_KEY="choose-a-long-secret"   # default: dev-key-change-me
python run.py                                  # http://127.0.0.1:5000
python -m unittest discover -s tests -v        # 27 tests
```

## API (header `X-API-Key` required except `/api/health`)
| Method | Path | Purpose |
|---|---|---|
| GET | /api/health | liveness |
| GET | /api/tools | tool catalogue |
| POST | /api/workflows | create `{name, steps[]}` |
| GET | /api/workflows | list |
| DELETE | /api/workflows/<id> | delete (cascades runs) |
| POST | /api/workflows/<id>/run | execute; 200 success, 422 failed |
| GET | /api/runs | last 50 runs |

Step: `{"id","tool","params":{},"depends_on":["other_id"]}`.
Use `$step_id` or `$step_id.key` inside params to pass earlier outputs forward.

Tools: `summarize`, `sentiment`, `extract_entities`, `route_ticket`, `calc`, `template`.

## Example
```bash
curl -X POST localhost:5000/api/workflows -H "X-API-Key: dev-key-change-me" \
 -H "Content-Type: application/json" -d '{"name":"demo","steps":[
 {"id":"a","tool":"sentiment","params":{"text":"great and fast"}},
 {"id":"b","tool":"template","params":{"template":"mood={m}","values":{"m":"$a.label"}},"depends_on":["a"]}]}'
```
