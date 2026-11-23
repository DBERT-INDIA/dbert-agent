# Architecture Report

## Layers (clean separation)
| Module | Responsibility |
|---|---|
| `app/__init__.py` | application factory, security headers |
| `app/routes.py` | HTTP layer only: parse, call, serialise |
| `app/security.py` | API-key auth, rate limit, schema validation |
| `app/engine.py` | pure domain logic: tools, DAG planner, executor (no Flask imports) |
| `app/db.py` | SQLite schema and connection lifecycle |
| `static/index.html` | dashboard client (uses `textContent`, never innerHTML for data) |

## Schema
- `workflows(id, name UNIQUE, definition JSON, created_at)`
- `runs(id, workflow_id FK ON DELETE CASCADE, status CHECK in (success, failed), result, log, created_at)`

## Key decisions
1. **DAG via Kahn's algorithm** – gives execution order and detects cycles at save time.
2. **Safe calculator** – AST walk with an operator whitelist and exponent cap; `eval` is never used.
3. **Retry policy** – transient errors (KeyError/TypeError/ValueError) retry up to 3 times; deterministic `WorkflowError` fails fast; remaining steps are logged as `skipped`.
4. **Parameterised SQL only** – no string-built queries.
5. **Pure engine** – testable without HTTP; tools are injectable (`execute(tools=...)`).

## Security & validation
Constant-time API key compare · per-client rate limit · 64 KB body cap · regex-validated names/ids ·
max 20 steps · tool whitelist · dependency existence check · CSP, nosniff, frame-deny headers.
Production notes: put behind HTTPS, set `AIVARA_API_KEY`, replace the in-memory rate limiter with Redis if running several workers.

## Testing
27 tests: algorithm unit tests (calc safety, sentiment, planner, retry), API integration tests
(auth, CRUD, 400/401/404/409/413/422/429 boundaries, SQL-injection attempt, cycle rejection).
