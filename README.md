# Model Risk Governance

Checks a model-risk team can run on a credit model, with a signed record of the results:

1. Drift: have the inputs changed, or has the relationship between inputs and outcomes changed?
2. Conformal abstention: turn the model's probability into a set of possible outcomes and send unclear cases to a person.
3. Fairness: disparate impact, equal opportunity and related metrics across groups.
4. Certificate: a hash chain over the three results, signed with Ed25519.

The certificate records automated checks. It is not a model approval: frameworks such as SR 11-7 and RBI's model-risk guidance expect independent validation and sign-off by people with that authority.

## Quick start

```bash
pip install -r requirements-dev.lock && pip install -e .
python -m pytest -q                        # 34 tests
python -m mrmgov bench                     # add --json for the full report
python -m mrmgov realdata --data FILE      # UCI credit default data, a few seconds
python -m mrmgov serve                     # API and console on http://127.0.0.1:8019
```

To run it in a container:

```bash
docker build -t model-risk-governance .
python -m mrmgov.ops keygen                # prints an API key, its hash and a signing key
docker run -d -p 8019:8019 -v mrmgov-data:/data -e MRMGOV_API_KEYS='{"sha256:...": {"user": "ops", "role": "admin"}}' \
  -e MRMGOV_SIGNING_KEY=... model-risk-governance
```

## API

| Endpoint | Purpose |
|---|---|
| `POST /audit` | run all checks on reference and production data and issue a signed certificate |
| `POST /conformal/predict` | prediction sets and decisions for new applicants |
| `POST /fairness/repair` | fairness metrics and the per-group threshold diagnostic |
| `GET /certificates`, `/certificates/{id}` | the certificate register |
| `GET /public-key` | current and previous signing keys |
| `GET /bench` | the benchmark |
| `GET /healthz`, `/readyz`, `/metrics` | health, readiness and Prometheus metrics |

API keys carry a role: `validator` runs audits and issues certificates, `analyst` uses the conformal and fairness tools, `auditor` reads the register, `monitor` reads metrics, `admin` does everything. How each check works is described in [docs/design.md](docs/design.md).

## Results

| Test | Result |
|---|---|
| Generated data with planted drift | the drifted features found exactly on 40 of 40 seeds (covariate) and 38 of 40 (concept) |
| Generated data, conformal coverage at 95% | 94.85–95.09% per group, with 42% of applicants sent to review |
| Real credit data (UCI), random split as a control | no drift reported |
| Real credit data, split by age or credit limit | drift reported in both; coverage falls to 92–94% in production |
| Real credit data, credit-limit split | ROC-AUC drops from 0.746 to 0.666 without the drift check raising it as a problem |
| Real credit data, approval rates by sex at one cut-off | women 85.1%, men 83.2% (no adverse impact) |

Details are in [docs/evaluation.md](docs/evaluation.md).

## Configuration

| Variable | Purpose |
|---|---|
| `MRMGOV_ENV=production` | refuse to start without hashed API keys, the signing key and a database |
| `MRMGOV_SIGNING_KEY` | base64 Ed25519 seed that signs certificates |
| `MRMGOV_SIGNING_KEY_PREVIOUS` | older keys to keep publishing while rotating |
| `MRMGOV_API_KEYS` / `MRMGOV_KEYS_FILE` | hashed API keys with roles |
| `MRMGOV_DATABASE_URL` | `sqlite:///...` or `postgresql://...` for the certificate register |

## Limitations

- Only one real dataset was used, and it has no time order, so its drift windows come from splitting on a feature.
- Drift thresholds are hand-set, and the split between input and relationship drift relies on a linear classifier.
- Conformal guarantees stop holding under drift, which is exactly when they matter.
- Fairness metrics on lending data need outcomes for rejected applicants too. Per-group thresholds are disparate treatment and are only offered as a diagnostic.
- The signing key is an environment variable, with no HSM, revocation list or trusted timestamp.
- The checks have not been reviewed by model-risk or fair-lending specialists, and the code has not had an independent security review.

See [SECURITY.md](SECURITY.md), [docs/threat-model.md](docs/threat-model.md) and [docs/model-card.md](docs/model-card.md).
