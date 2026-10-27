# Changelog

All notable changes to this project. Versions follow [semantic versioning](https://semver.org/).

## 1.0.0 — 2026-10-27

### Added
- `pyproject.toml` with an `mrmgov` console command, and pinned `requirements.lock` /
  `requirements-dev.lock` (generated with `uv pip compile --universal`).
- Container image: slim Python 3.12 base, non-root user, health check on `/healthz`.
- CI on every push: ruff, bandit, pip-audit, pytest with a coverage report.
- Settings from `MRMGOV_*` environment variables, checked at startup. `MRMGOV_ENV=production`
  refuses to start without hashed API keys, `MRMGOV_SIGNING_KEY` and `MRMGOV_DATABASE_URL`.
- Roles on API keys: `validator` runs audits (and so issues certificates), `analyst` runs the
  conformal and fairness tools, `auditor` reads issued certificates, `monitor` scrapes metrics,
  `admin` does all of it plus the benchmark. Keys can be stored as `sha256:<hex>`;
  `python -m mrmgov.ops keygen` makes new ones.
- Per-client rate limit, request body cap (chunked bodies now counted instead of refused), security
  headers, CORS only for listed origins.
- JSON logs with request IDs; `/healthz`, `/readyz` and Prometheus `/metrics`
  (`mrmgov_decisions_total` counts certificate dispositions).
- A certificate register in a database: SQLite by default, Postgres through `MRMGOV_DATABASE_URL`,
  numbered SQL migrations (`python -m mrmgov migrate`). Every certificate `/audit` signs is stored
  with who asked for it and the public key that signs it; `GET /certificates` and
  `GET /certificates/{id}` return them.
- Signing-key rotation: `MRMGOV_SIGNING_KEY_PREVIOUS`; `/public-key` lists the previous public keys
  so older certificates stay verifiable.
- `SECURITY.md`, `docs/threat-model.md`, `docs/runbook.md`, `docs/model-card.md`.

### Changed
- **`MRMGOV_SIGNING_KEY` is now base64 of the 32-byte seed**, not 64 hex characters (a hex value is
  refused, not misread). Convert with
  `python -c "import base64,sys; print(base64.b64encode(bytes.fromhex(sys.argv[1])).decode())" <hex>`.
- `MRMGOV_API_KEY` (one shared key) is replaced by `MRMGOV_API_KEYS` with roles. Without keys the
  service refuses to listen on anything but loopback.
- `/health` is replaced by `/healthz` (liveness) and `/readyz` (database reachable). `/bench` needs
  the admin role.

### Unchanged
- Drift, conformal, fairness and certificate code, and every benchmark and real-data number in the
  README. `results/real_data.json` was not re-run: the only change in `realdata.py` turns a lambda
  into a function for the linter, which does not change what it computes.
