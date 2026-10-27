# Model Risk Governance checks (`mrmgov`)

Three checks a model-risk team might run on a credit model, plus a signed record of what they found:

1. **Drift attribution** (`drift.py`): did the inputs move (covariate drift, P(X)) or did the relationship between inputs and outcome move (concept drift, P(Y|X))?
2. **Conformal abstention** (`conformal.py`): turn a model's probability into a set of plausible labels, and send the ambiguous cases to a human.
3. **Group fairness metrics** (`fairness.py`): disparate impact ratio, equal opportunity difference and related metrics, plus a search over per-group thresholds used as a diagnostic.
4. **Certificate** (`report.py`): a hash chain over the three summaries, signed with Ed25519.

The benchmark runs on **generated data**; "Real data" below runs the checks on a public credit dataset.

> The certificate is a record of automated checks, **not a model approval**. SR 11-7 expects independent validation and sign-off by people with that authority. The frameworks the certificate names (SR 11-7, RBI model-risk expectations, ECOA / Reg B, EU AI Act Annex III) are references only. Passing these checks does not show compliance with any of them.

## How it works

**Drift.** For each feature it computes PSI against reference deciles, the two-sample KS statistic with an asymptotic p-value, and the 1-D Wasserstein-1 distance. The KS and W1 values match scipy.

A linear logistic "domain classifier" separates reference rows from production rows, and its odds give importance weights w(x) ≈ dP_prod/dP_ref. The change in Brier score then splits into two parts:
- ΔBrier_covariate = reweighted reference Brier − reference Brier;
- ΔBrier_concept = production Brier − reweighted reference Brier.

The two parts add up exactly by construction. How meaningful the split is depends on the weights, and a linear classifier mostly sees mean shifts.

Each feature also gets two attribution shares. These are heuristics, not Shapley values:
- **Covariate share:** the mean change in prediction when only that feature is quantile-mapped onto its production distribution, plus 0.25·PSI and 0.15·normalised W1.
- **Concept share:** the change in covariance between the standardised feature and the model residual.

A feature is flagged as covariate-drifted if PSI ≥ 0.15, or if KS ≥ 0.14 and normalised W1 ≥ 0.25. It is flagged as concept-drifted if its slope shift is ≥ 0.06. These thresholds are hand-set.

**Conformal.** This is Mondrian split conformal prediction:
- Non-conformity score: s = 1 − p̂ for label 1, and s = p̂ for label 0.
- Each group's quantile is the ⌈(n+1)(1−α)⌉-th smallest calibration score. When that rank exceeds n, the quantile is infinite and the set contains both labels.

Possible prediction sets:
- {0} → `AUTO_APPROVE`
- {1} → `AUTO_DECLINE`
- {0, 1} → `ABSTAIN_HUMAN_UNDERWRITER`

If calibration and test rows in a group are exchangeable, then P(Y ∈ C(X) | G = g) ≥ 1 − α. This holds on average over calibration draws. It is not a promise that any single decision is right: every miss is a singleton set.

**Fairness.** Two parts:
- **Metrics:** DIR, SPD, EOD, AOD, F1, AUC and expected profit, computed at given thresholds.
- **Threshold search:** a grid search over (t_priv, t_unpriv). It maximises profit + 0.35·F1 subject to 0.825 ≤ DIR ≤ 1.25 and |EOD| ≤ 0.03, and falls back to looser bounds if nothing qualifies.

**Using separate cut-offs per protected group is disparate treatment** under ECOA / Regulation B (12 CFR 1002.4(a)), even when it narrows a gap. The 0.80 ratio comes from the EEOC's employment guidelines (29 CFR 1607.4(D)). US lending law has no such safe harbour. The search is useful for showing how far apart two groups' score distributions are. A real remediation would look for a less discriminatory model that applies one rule to everyone.

**Certificate.**
- The chain hashes the header (model, version, timestamp), then the drift, conformal and fairness summaries, then a root over the checklist, the disposition and the fairness hash.
- The root hash is signed with Ed25519 (`cryptography`).
- `verify_certificate_signature(cert, public_key)` recomputes the chain and checks the signature. The public key must come from somewhere you trust, never from the certificate itself.
- Disposition values are `AUTOMATED_CHECKS_PASSED`, `CHECKS_PASSED_WITH_CONCEPT_DRIFT` or `CHECKS_FAILED`.

## Results

`python -m mrmgov bench` (seed 42), all on generated data:

| Check | Result |
|---|---|
| no-drift sample | `NO_DRIFT`, no features flagged |
| covariate drift planted on 2 features (mean +0.95, +0.85) | `COVARIATE_DRIFT`, exactly those 2 flagged |
| concept drift planted on 2 coefficients | `CONCEPT_DRIFT`, exactly those 2 flagged |
| conformal coverage, α = 0.05, 3 groups × 1,200 calibration and 1,200 test rows, mean over 50 splits | 94.85%–95.09% per group (target 95%) |
| conformal abstention, first split | 42.4% of applicants sent to review |
| baseline DIR → DIR with group thresholds | 0.629 → 0.878 |
| EOD baseline → with group thresholds | −0.188 → −0.018 |
| AUC / profit retention | 100.4% / 104.9% |
| certificate signature verifies (edited and re-signed copies are rejected in the tests) | yes |

Over seeds 0–39:
- Covariate drift was identified exactly on all 40 seeds.
- Concept drift was identified exactly on **38 of 40**. On the other 2, a third feature's slope shift (0.063 and 0.069) crossed the 0.06 threshold.
- The conformal mean-coverage check passed on **39 of 40**. The check allows 3 standard errors in each of 3 groups, so an occasional miss is expected.
- Baseline DIR ranged 0.61–0.70.
- With group thresholds, DIR ranged 0.88–1.04 and |EOD| stayed ≤ 0.030.

Read these as consistency checks, not measurements:
- **Drift:** the scenarios plant exactly the kind of shift the detectors look for.
- **Conformal:** the scores are calibrated by construction.
- **Fairness:** the "bias" is a constant logit shift on one group, which a threshold shift undoes exactly, so the search is bound to succeed. The 42% abstention rate is the more informative number. With scores this uncertain, 95% coverage means a human reviews almost half the applicants.

A single 1,200-point split has a coverage standard deviation of about 0.9 percentage points. The earlier check of "≥ 94.5% in every group on one split" failed on 23 of 40 seeds even though the method was working. The benchmark therefore checks the mean over repeated splits.

## Real data

`python -m mrmgov realdata --data "default of credit card clients.xls"` runs the checks on UCI's Default of Credit Card Clients data: 30,000 card holders at a Taiwanese bank in 2005, 22.1% of whom defaulted the next month (CC BY 4.0). Results are in `results/real_data.json`.
- **Windows.** The data has no time column, so reference and production are made by splitting on a feature: holders under 35 against 35 and over (age), and credit limits up to the median against above it (limit). A random half-and-half split is the control, where nothing should drift.
- **Model.** The L2 logistic regression `/audit` fits, on log-scaled amounts. Sex is the group attribute and is kept out of the model. In each split the reference rows are divided 50/25/25 into model fit, conformal calibration and a reference test set. The approval cut-off and the group-threshold search use the calibration rows only.

| Split | Regime | Covariate-drifted features | ROC-AUC ref → prod | Coverage at α = 0.05, ref → prod |
|---|---|---|---|---|
| random (control) | `NO_DRIFT` | none | 0.744 → 0.748 | 95.3% → 95.4% |
| age | `COVARIATE_DRIFT` | AGE, MARRIAGE | 0.740 → 0.738 | 94.9% → **93.8%** |
| limit | `COVARIATE_DRIFT` | 21 of 22 | 0.746 → **0.666** | 95.6% → **92.2%** |

What the real data shows:
- **No false alarm on the control**, and both feature splits are caught as covariate drift. That part is easy: the split feature separates the windows perfectly (domain-classifier AUC 1.00).
- **No concept drift was flagged, and refitting agrees.** A model refitted on half the production rows scores no better on the other half than the reference model (Brier, reference vs refitted: 0.1479 vs 0.1479 for age, 0.1115 vs 0.1121 for limit).
- **The drift check misses a real loss of accuracy.** In the limit split, ROC-AUC falls from 0.746 to 0.666, yet the regime is plain covariate drift and the Brier score improves (0.167 → 0.117), because high-limit holders default less (15.7% vs 28.3%) and are harder to rank. A monitor watching only the regime and Brier would not see it. When the windows don't overlap at all, the density-ratio weights are degenerate, so the covariate/concept split of the Brier change means little.
- **Conformal coverage breaks under drift, as expected.** It holds on the reference test sets (94.9–95.6%) and falls to 93.8% and 92.2% in production, well outside sampling error (about 0.2 points on 14,000 rows).
- **Abstention is heavy on real scores.** At α = 0.05, 57% of applicants go to a human underwriter (64% of men, 52% of women), against 42% on the generated data. At α = 0.10 it is 25%, at α = 0.20 3%.
- **No adverse impact against women at a single cut-off.** Approving when P(repay) ≥ 0.625 (the most profitable cut-off on the calibration rows) approves 85.1% of women and 83.2% of men: DIR 1.02, EOD +0.002. Women in this data default less (20.8% vs 24.2%). The group-threshold search finds almost nothing to change (DIR 1.01).

These come from one run (seed 0); nothing was tuned after seeing them.

## Run it

```bash
pip install -r requirements-dev.lock && pip install -e .
python -m pytest -q                        # 34 tests
python -m mrmgov bench                     # add --json for the full report
python -m mrmgov realdata --data FILE      # UCI credit default data, a few seconds (pip install -e ".[realdata]")
python -m mrmgov serve                     # API + console at http://127.0.0.1:8019
```

Endpoints:
- `GET /public-key` (current and previous signing keys; public)
- `POST /audit`
- `POST /conformal/predict`
- `POST /fairness/repair`
- `GET /certificates`, `GET /certificates/{id}`
- `GET /bench?seed=`
- `GET /healthz`, `GET /readyz`, `GET /metrics`

Configuration (all `MRMGOV_*` environment variables, see `docs/runbook.md`):
- **API keys with roles.** Callers send a key in `X-API-Key` (the console has a field for it) or `Authorization: Bearer`. `validator` runs `/audit` and so issues certificates, `analyst` runs the conformal and fairness tools, `auditor` reads the certificate register, `monitor` reads metrics, `admin` does all of it plus the benchmark. `python -m mrmgov.ops keygen` makes a key and the hash to configure. Without keys the service runs open in development and refuses to listen on anything but loopback.
- **`MRMGOV_SIGNING_KEY`:** base64 of a 32-byte Ed25519 seed (`keygen` prints one). Required in production; in development an unset key is generated per process, so certificates can only be verified against that process's `/public-key`. `MRMGOV_SIGNING_KEY_PREVIOUS` keeps older public keys listed while rotating.
- **Certificate register.** Every certificate `/audit` signs is stored in SQLite or Postgres (`MRMGOV_DATABASE_URL`) with who asked for it and the public key that signs it.
- `MRMGOV_ENV=production` refuses to start without hashed keys, the signing key and a database.

**Docker.**

```bash
docker build -t model-risk-governance:1.0.0 .
docker run -d -p 8019:8019 -v mrmgov-data:/data -e MRMGOV_API_KEYS='{"sha256:...": {"user": "ops", "role": "admin"}}' -e MRMGOV_SIGNING_KEY=... model-risk-governance:1.0.0
```

The image runs in production mode as a non-root user. Run without keys it exits with a configuration error.

`/audit` uses y = 1 to mean default, and it takes a group label for every reference and production row. It works in three steps:
1. It fits a logistic model on a random half of the reference rows.
2. It calibrates the conformal layer on the other half, so calibration is out-of-sample.
3. It passes 1 − score and 1 − y to the fairness metrics, which use the convention "score = P(repay), y = 1 means repaid". You can supply `fairness_*` inputs instead.

Request limits:
- 10,000 rows × 50 features per matrix;
- 50,000 points per list;
- a 16 MB body.

A maximum-size `/audit` takes about 2.3 s.

## Limits

- **One real dataset, no real time order.** The UCI test builds its drift windows by splitting on a feature, which is starker than drift over time, and it has outcomes for every holder (no reject-inference problem).
- **Drift detection is heuristic.** The thresholds are hand-set and the attribution shares aren't Shapley values. The density ratio comes from a linear classifier, so a shift in variance or interactions can be misattributed to concept drift. Two of 40 seeds flagged an extra feature.
- **Conformal coverage is conditional on exchangeability.** Under drift, which is exactly when it matters, the guarantee no longer holds. Groups unseen in calibration fall back to the pooled threshold, which has no per-group guarantee. Coverage is marginal within a group, not per applicant.
- **Fairness metrics need outcomes for everyone.** In lending, repayment is only observed for approved applicants, so EOD on real data needs reject inference. The group-threshold search is disparate treatment and is only for diagnosis (see above).
- **The certificate is only as trustworthy as the key.** The key is an environment variable (rotatable, required in production). There is no HSM, revocation or trusted timestamp, and the issuing time is whatever the server clock says.
- **The service certifies the data it is given.** It can't tell whether the reference and production rows are the real ones; certificates record who asked. Rate limits are per process, and a caller with a key can still keep the CPU busy with maximum-size requests.
- **Not reviewed by a model-risk or fair-lending specialist.** A code audit of this repository does not make it fit for production model governance.

## Production readiness

In place (1.0.0):
- Settings checked at startup; production refuses to start without hashed API keys, the signing key and a database, and never listens publicly without keys.
- API keys with roles, per-client rate limit, request body cap, security headers, CORS only for listed origins, validation errors that don't echo input.
- A certificate register in a database with migrations; signing-key rotation with previous public keys published.
- JSON logs with request IDs, `/healthz`, `/readyz`, Prometheus `/metrics`.
- Pinned dependency locks, a non-root container image with a health check, CI running ruff, bandit, pip-audit and the tests.
- `SECURITY.md`, `docs/threat-model.md`, `docs/runbook.md`, `docs/model-card.md`.

Still needed before real use:
- Review by model-risk and fair-lending specialists, and independent validation of the checks against the bank's own models and data (with real time windows and reject inference).
- An independent security review and penetration test.
- RBI model-risk guidance and DPDP Act 2023 sign-off for the data sent to the service; ECOA / Reg B review wherever US lending rules apply.
- Real integrations: the model inventory, monitoring pipelines and the governance workflow that acts on certificates.
- HSM/KMS for the signing key, a revocation list and trusted timestamps for certificates, and write-once export of the register.
- An on-call team, SLOs and alerting wired to the metrics above.
