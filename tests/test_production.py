"""Settings, auth, limits, headers, metrics, health, the certificate register and migrations."""
import base64
import hashlib
import importlib
import json
import os
import subprocess
import sys

import numpy as np
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from fastapi.testclient import TestClient
from sqlalchemy import text

from mrmgov import db, ops
from mrmgov.report import GovernanceCertificate, verify_certificate_signature

S32 = base64.b64encode(b"s" * 32).decode()
KEY = "mrmgov_" + "x" * 40
HASHED = {f"sha256:{hashlib.sha256(KEY.encode()).hexdigest()}": {"user": "meera", "role": "validator"}}
KEYS = {"k-val": {"user": "meera", "role": "validator"}, "k-ana": {"user": "ravi", "role": "analyst"},
        "k-aud": {"user": "asha", "role": "auditor"}, "k-mon": {"user": "prom", "role": "monitor"},
        "k-admin": {"user": "root", "role": "admin"}}
CONF = {"probs_calib": [0.1, 0.9], "y_calib": [0, 1], "groups_calib": ["A", "A"], "probs_test": [0.2], "groups_test": ["A"]}


def audit_body(model_id="PROD-TEST-01"):
    rng = np.random.default_rng(3)
    X_r, X_p = rng.normal(size=(60, 2)), rng.normal(loc=0.4, size=(60, 2))
    return {"model_id": model_id, "X_ref": X_r.tolist(), "y_ref": [int(r[0] > 0) for r in X_r],
            "X_prod": X_p.tolist(), "y_prod": [int(r[0] > 0) for r in X_p],
            "groups_ref": [i % 2 for i in range(60)], "groups_prod": [i % 2 for i in range(60)]}


def prod_env(**over):
    env = {"MRMGOV_ENV": "production", "MRMGOV_API_KEYS": json.dumps(HASHED), "MRMGOV_SIGNING_KEY": S32,
           "MRMGOV_DATABASE_URL": "sqlite://"}
    env.update(over)
    return {k: v for k, v in env.items() if v is not None}


def boot(monkeypatch, **env):
    for k in [k for k in os.environ if k.startswith("MRMGOV_")]:
        monkeypatch.delenv(k)
    for k, v in env.items():
        monkeypatch.setenv(f"MRMGOV_{k}", v)
    import mrmgov.api as api

    return importlib.reload(api)


def test_production_settings_accepted():
    s = ops.load_settings(prod_env())
    assert s.production and not s.demo and s.secret["SIGNING_KEY"] == b"s" * 32 and not s.ephemeral


@pytest.mark.parametrize("change, message", [
    ({"MRMGOV_API_KEYS": None}, "API keys"),
    ({"MRMGOV_SIGNING_KEY": None}, "SIGNING_KEY"),
    ({"MRMGOV_DATABASE_URL": None}, "DATABASE_URL"),
    ({"MRMGOV_API_KEYS": json.dumps({KEY: {"user": "meera", "role": "validator"}})}, "sha256"),
    ({"MRMGOV_API_KEYS": json.dumps({"sha256:" + "0" * 64: {"user": "x", "role": "root"}})}, "roles"),
    ({"MRMGOV_SIGNING_KEY": base64.b64encode(b"short").decode()}, "32 bytes"),
    ({"MRMGOV_SIGNING_KEY": "73" * 32}, "32 bytes"),  # the old 64-hex-character format is refused, not misread
    ({"MRMGOV_SIGNING_KEY_PREVIOUS": base64.b64encode(b"p" * 48).decode()}, "32 bytes"),
    ({"MRMGOV_SIGNING_KEY": "not base64!"}, "base64"),
    ({"MRMGOV_CORS_ORIGINS": "*"}, "CORS"),
    ({"MRMGOV_ENV": "prod"}, "development or production"),
    ({"MRMGOV_DATABASE_URL": "mysql://x"}, "postgresql"),
])
def test_production_refuses_to_start(change, message):
    with pytest.raises(ops.ConfigError, match=message):
        ops.load_settings(prod_env(**change))


def test_no_public_bind_without_keys():
    with pytest.raises(ops.ConfigError, match="without API keys"):
        ops.load_settings({"MRMGOV_HOST": "0.0.0.0"})
    assert ops.load_settings({"MRMGOV_HOST": "0.0.0.0", "MRMGOV_API_KEYS": json.dumps(HASHED)}).host == "0.0.0.0"
    dev = ops.load_settings({})
    assert dev.demo and dev.ephemeral == ("SIGNING_KEY",) and dev.port == 8019


def test_serve_exits_with_a_readable_error():
    env = {k: v for k, v in os.environ.items() if not k.startswith("MRMGOV_")}
    r = subprocess.run([sys.executable, "-m", "mrmgov", "serve", "--host", "0.0.0.0"], env=env,
                       capture_output=True, text=True, timeout=60)
    assert r.returncode != 0 and "configuration error" in r.stderr and "without API keys" in r.stderr


def test_roles_decide_what_a_key_may_do(monkeypatch):
    c = TestClient(boot(monkeypatch, API_KEYS=json.dumps(KEYS)).app)
    assert c.post("/conformal/predict", json=CONF).status_code == 401
    assert c.post("/conformal/predict", json=CONF, headers={"x-api-key": "wrong"}).status_code == 401
    assert c.post("/conformal/predict", json=CONF, headers={"x-api-key": "k-ana"}).status_code == 200
    assert c.post("/audit", json=audit_body(), headers={"x-api-key": "k-ana"}).status_code == 403
    assert c.post("/audit", json=audit_body(), headers={"authorization": "Bearer k-val"}).status_code == 200
    assert c.get("/certificates", headers={"x-api-key": "k-ana"}).status_code == 403
    assert c.get("/certificates", headers={"x-api-key": "k-aud"}).status_code == 200
    assert c.post("/conformal/predict", json=CONF, headers={"x-api-key": "k-aud"}).status_code == 403
    assert c.get("/bench", headers={"x-api-key": "k-val"}).status_code == 403
    assert c.get("/public-key").status_code == 200  # public by design
    assert c.get("/metrics", headers={"x-api-key": "k-val"}).status_code == 403
    m = c.get("/metrics", headers={"x-api-key": "k-mon"})
    assert m.status_code == 200 and 'http_requests_total{method="POST",route="/audit",status="200"}' in m.text
    assert "mrmgov_decisions_total" in m.text and "errors_total" in m.text


def test_rate_limit(monkeypatch):
    c = TestClient(boot(monkeypatch, RATE_LIMIT="3").app)
    codes = [c.post("/conformal/predict", json=CONF).status_code for _ in range(5)]
    assert codes[:3] == [200] * 3 and codes[3:] == [429, 429]
    assert int(c.post("/conformal/predict", json=CONF).headers["retry-after"]) >= 1
    assert c.get("/healthz").status_code == 200  # probes are never limited


def test_body_size_cap(monkeypatch):
    c = TestClient(boot(monkeypatch, MAX_BODY_BYTES="1000").app)
    assert c.post("/conformal/predict", json={**CONF, "probs_test": [0.5] * 300}).status_code == 413

    def chunks():  # no content-length: the cap still holds
        for _ in range(20):
            yield b"a" * 100

    assert c.post("/conformal/predict", content=chunks(), headers={"content-type": "application/json"}).status_code == 413
    assert c.post("/conformal/predict", json=CONF).status_code == 200


def test_headers_and_request_ids(monkeypatch):
    c = TestClient(boot(monkeypatch).app)
    r = c.post("/conformal/predict", json=CONF, headers={"x-request-id": "abc-123"})
    assert r.headers["x-request-id"] == "abc-123"
    assert r.headers["x-content-type-options"] == "nosniff" and r.headers["x-frame-options"] == "DENY"
    assert r.headers["content-security-policy"] == ops.API_CSP and r.headers["cache-control"] == "no-store"
    assert "unsafe-inline" in c.get("/").headers["content-security-policy"]
    assert len(c.get("/healthz", headers={"x-request-id": "bad id\n"}).headers["x-request-id"]) == 32
    assert "access-control-allow-origin" not in c.get("/healthz", headers={"origin": "https://evil.example"}).headers


def test_cors_only_for_listed_origins(monkeypatch):
    c = TestClient(boot(monkeypatch, CORS_ORIGINS="https://console.bank.example").app)
    ok = c.get("/healthz", headers={"origin": "https://console.bank.example"})
    assert ok.headers["access-control-allow-origin"] == "https://console.bank.example"
    assert "access-control-allow-origin" not in c.get("/healthz", headers={"origin": "https://evil.example"}).headers


def test_errors_do_not_leak(monkeypatch):
    api = boot(monkeypatch)

    @api.app.get("/boom")
    def boom():
        raise RuntimeError("secret detail: applicant 4111111111111111")

    c = TestClient(api.app)
    r = c.get("/boom")
    assert r.status_code == 500 and "4111" not in r.text and r.json()["request_id"] == r.headers["x-request-id"]
    bad = c.post("/conformal/predict", json={**CONF, "probs_test": [4111111111111111]})
    assert bad.status_code == 422 and "4111" not in bad.text
    assert c.post("/audit", content=b"{not json", headers={"content-type": "application/json"}).status_code == 422


def test_health_and_readiness(monkeypatch):
    api = boot(monkeypatch)
    c = TestClient(api.app)
    assert c.get("/healthz").json() == {"status": "ok"}
    r = c.get("/readyz")
    assert r.status_code == 200 and r.json()["checks"] == {"database": True}
    api.store.dispose()
    monkeypatch.setattr(api.store, "connect", lambda: (_ for _ in ()).throw(RuntimeError("db down")))
    r = c.get("/readyz")
    assert r.status_code == 503 and r.json()["checks"]["database"] is False and "db down" not in r.text


def test_json_logs_carry_request_ids(capsys):
    ops.setup_logging("INFO")
    token = ops.request_id.set("rid-1")
    ops.log.info("hello", extra={"fields": {"n": 1}})
    ops.request_id.reset(token)
    line = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert line["msg"] == "hello" and line["request_id"] == "rid-1" and line["n"] == 1 and line["level"] == "INFO"


def _url(tmp_path):
    return os.environ.get("TEST_DATABASE_URL") or f"sqlite:///{tmp_path / 'certs.db'}"


def test_certificates_survive_a_restart_and_rotation(monkeypatch, tmp_path):
    url, keys = _url(tmp_path), json.dumps(KEYS)
    api = boot(monkeypatch, DATABASE_URL=url, API_KEYS=keys, SIGNING_KEY=S32)
    c = TestClient(api.app)
    issued = c.post("/audit", json=audit_body("PROD-RESTART"), headers={"x-api-key": "k-val"}).json()["certificate"]
    old_pub = c.get("/public-key").json()["public_key"]

    new = base64.b64encode(b"n" * 32).decode()  # restart with a new key; the old one is kept as previous
    c = TestClient(boot(monkeypatch, DATABASE_URL=url, API_KEYS=keys, SIGNING_KEY=new, SIGNING_KEY_PREVIOUS=S32).app)
    pk = c.get("/public-key").json()
    assert pk["public_key"] != old_pub and pk["previous_public_keys"] == [old_pub]
    stored = c.get(f"/certificates/{issued['certificate_id']}", headers={"x-api-key": "k-aud"}).json()
    assert stored["issued_by"] == "meera" and stored["public_key"] == old_pub and stored["certificate"] == issued
    cert = GovernanceCertificate(**stored["certificate"])
    assert verify_certificate_signature(cert, Ed25519PublicKey.from_public_bytes(bytes.fromhex(old_pub)))
    listed = c.get("/certificates", params={"model_id": "PROD-RESTART"}, headers={"x-api-key": "k-aud"}).json()
    assert issued["certificate_id"] in [x["certificate_id"] for x in listed["certificates"]]
    assert c.get("/certificates/MRM-CERT-NOPE", headers={"x-api-key": "k-aud"}).status_code == 404


def test_migrations_apply_once(tmp_path):
    url = f"sqlite:///{tmp_path / 'm.db'}"
    eng = db.engine(url)
    assert db.migrate(eng) == ["001_init.sql"]
    assert db.migrate(eng) == []
    with eng.connect() as c:
        assert [r[0] for r in c.execute(text("SELECT version FROM schema_migrations"))] == [1]
        tables = {r[0] for r in c.execute(text("SELECT name FROM sqlite_master WHERE type = 'table'"))}
    assert "certificates" in tables
    env = {**{k: v for k, v in os.environ.items() if not k.startswith("MRMGOV_")}, "MRMGOV_DATABASE_URL": url}
    out = subprocess.run([sys.executable, "-m", "mrmgov", "migrate"], env=env, capture_output=True, text=True, timeout=60)
    assert out.returncode == 0 and "up to date" in out.stdout
