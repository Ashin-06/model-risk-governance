# Runbook

How to deploy, configure and operate the checks service. Commands assume the repository root.

## Deploy

```bash
docker build -t model-risk-governance:1.0.0 .
python -m mrmgov.ops keygen          # one API key per client, plus a signing key
docker run -d --name mrmgov -p 8019:8019 -v mrmgov-data:/data \
  -e MRMGOV_KEYS_FILE=/run/secrets/mrmgov-keys.json \
  -e MRMGOV_SIGNING_KEY=... \
  model-risk-governance:1.0.0
curl -fsS http://127.0.0.1:8019/readyz
curl -fsS http://127.0.0.1:8019/public-key     # publish this once through a channel readers trust
```

The image runs in production mode as a non-root user and keeps its SQLite database (the
certificate register) in the `/data` volume. It refuses to start, and logs why, when anything
required is missing. Terminate TLS in front of it.

For more than one replica, use Postgres (`MRMGOV_DATABASE_URL=postgresql://user:pass@host/db`), give
every replica the same signing key, and put a shared rate limiter at the ingress.

Without Docker: `pip install -r requirements.lock && pip install -e .`, then `mrmgov serve`.

## Configure

| Variable | Required in production | Meaning |
|---|---|---|
| `MRMGOV_ENV` | — | `development` (default) or `production` |
| `MRMGOV_API_KEYS` / `MRMGOV_KEYS_FILE` | yes | JSON `{"sha256:<hex>": {"user": "...", "role": "..."}}`. Roles: `validator` (`/audit`, the tools, the register), `analyst` (`/conformal/predict`, `/fairness/repair`), `auditor` (`/certificates`), `monitor` (`/metrics`), `admin` (all, plus `/bench`) |
| `MRMGOV_SIGNING_KEY` | yes | base64 Ed25519 private key seed, exactly 32 bytes (1.0.0 no longer reads the old 64-hex format) |
| `MRMGOV_SIGNING_KEY_PREVIOUS` | — | comma-separated older seeds; their public keys are listed by `/public-key` |
| `MRMGOV_DATABASE_URL` | yes | `sqlite:////data/mrmgov.db` or `postgresql://...` (default in development: in memory) |
| `MRMGOV_HOST`, `MRMGOV_PORT` | — | default `127.0.0.1:8019`; any non-loopback address needs API keys |
| `MRMGOV_RATE_LIMIT` | — | requests per minute per client (default 120) |
| `MRMGOV_MAX_BODY_BYTES` | — | default 16,000,000 |
| `MRMGOV_CORS_ORIGINS` | — | comma-separated origins allowed to call from a browser; none by default |
| `MRMGOV_LOG_LEVEL` | — | default `INFO` |
| `FORWARDED_ALLOW_IPS` | — | addresses of proxies whose `X-Forwarded-For` is trusted (uvicorn) |

## Watch

- `GET /healthz`: the process is up (used by the container health check).
- `GET /readyz`: database reachable; 503 otherwise.
- `GET /metrics` (role `monitor`): `http_requests_total`, `http_request_duration_seconds`,
  `errors_total`, `mrmgov_decisions_total{decision="AUTOMATED_CHECKS_PASSED|CHECKS_PASSED_WITH_CONCEPT_DRIFT|CHECKS_FAILED"}`.
- Logs are one JSON object per line on stdout, each with a `request_id` that is also returned in the
  `X-Request-ID` response header. No request data is logged.

Suggested alerts: `/readyz` failing for 2 minutes; `errors_total{kind="unhandled"}` increasing; 5xx
share above 1% for 10 minutes; `/audit` p95 latency above 5 s (someone sending maximum-size requests
in a loop); a burst of certificates from one user.

## Rotate keys

**API keys.** Add the new hash to the key file, restart, give the client the new key, then remove
the old hash and restart again.

**Signing key.** Set `MRMGOV_SIGNING_KEY_PREVIOUS` to the current seed and `MRMGOV_SIGNING_KEY` to a
new one, restart, and publish the new public key through your trusted channel. New certificates are
signed with the new key; each stored certificate records the public key that signed it. Keep the
old public key published for as long as certificates signed with it matter. Remove `_PREVIOUS` from
the service once you no longer need `/public-key` to list it.

## Back up and restore

SQLite (in the `/data` volume):

```bash
docker exec mrmgov python -c "import sqlite3; s=sqlite3.connect('/data/mrmgov.db'); d=sqlite3.connect('/data/backup.db'); s.backup(d)"
docker cp mrmgov:/data/backup.db ./mrmgov-$(date +%F).db
```

Restore: stop the container, copy the backup to `/data/mrmgov.db` in the volume, start it, check
`/readyz` and `GET /certificates`.

Postgres: `pg_dump` / `pg_restore` as usual. Migrations are applied automatically on start, or with
`python -m mrmgov migrate`.

Export the register to write-once storage regularly: deletions can't be detected from the
register itself.

## Incidents

1. **Leaked API key.** Remove its hash from the key file and restart. List certificates issued by
   that user (`GET /certificates`, `issued_by`) and treat them as unverified. Issue a new key.
2. **Leaked signing key.** Rotate *without* keeping the old key as previous, publish the new public
   key, and tell certificate readers that anything signed with the old key after the suspected leak
   time can't be trusted (there is no trusted timestamp, so "after" is a judgement).
3. **A certificate doesn't verify.** It was edited, or it's checked against the wrong public key.
   Fetch the stored copy (`/certificates/{id}`), which records the signing public key, and compare.
4. **Database unavailable.** `/readyz` returns 503 and `/audit` fails with 500 (the certificate is
   not issued without being stored). Callers retry.
