# Design

## Drift

For each feature, `drift.py` computes the Population Stability Index against reference deciles, the two-sample KS statistic with an asymptotic p-value, and the Wasserstein-1 distance; KS and W1 match scipy.

A logistic "domain classifier" separates reference rows from production rows, and its odds give importance weights w(x) ≈ dP_prod/dP_ref. The change in Brier score then splits into two parts that add up exactly:
- covariate part = reweighted reference Brier − reference Brier;
- concept part = production Brier − reweighted reference Brier.

How meaningful the split is depends on the weights; a linear classifier mostly sees shifts in the mean.

Each feature also gets two attribution shares, which are heuristics rather than Shapley values: the covariate share is the mean change in prediction when only that feature is mapped onto its production distribution (plus small PSI and W1 terms), and the concept share is the change in covariance between the standardised feature and the model's residual.

A feature is flagged as covariate drift if PSI ≥ 0.15, or if KS ≥ 0.14 and normalised W1 ≥ 0.25, and as concept drift if its slope shifts by 0.06 or more. These thresholds are hand-set.

## Conformal abstention

`conformal.py` uses Mondrian split conformal prediction. The non-conformity score is 1 − p for label 1 and p for label 0, and each group's quantile is the ⌈(n+1)(1−α)⌉-th smallest calibration score. If that rank exceeds n, the set contains both labels.

| Prediction set | Decision |
|---|---|
| {0} | `AUTO_APPROVE` |
| {1} | `AUTO_DECLINE` |
| {0, 1} | `ABSTAIN_HUMAN_UNDERWRITER` |

If calibration and test rows in a group are exchangeable, the true label is in the set with probability at least 1 − α for that group, on average over calibration draws. It is not a promise about any single decision.

## Fairness

`fairness.py` computes the disparate impact ratio (DIR), statistical parity difference, equal opportunity difference (EOD), average odds difference, F1, AUC and expected profit at given thresholds.

It also searches per-group thresholds to maximise profit + 0.35 × F1 subject to 0.825 ≤ DIR ≤ 1.25 and |EOD| ≤ 0.03. Using separate cut-offs for protected groups is disparate treatment under ECOA / Regulation B, even when it narrows a gap, so the search is only a diagnostic of how far apart two groups' scores are. A real fix would look for a less discriminatory model that applies one rule to everyone.

## Certificate

`report.py` hashes a header (model, version, timestamp), then the drift, conformal and fairness summaries, then a root over the checklist, the disposition and the fairness hash, and signs the root with Ed25519. `verify_certificate_signature(cert, public_key)` recomputes the chain and checks the signature; the public key must come from a trusted source, never from the certificate. The disposition is `AUTOMATED_CHECKS_PASSED`, `CHECKS_PASSED_WITH_CONCEPT_DRIFT` or `CHECKS_FAILED`.

## `/audit` conventions

`/audit` uses y = 1 for default and needs a group label for every row. It fits a logistic model on half the reference rows, calibrates the conformal layer on the other half, and passes 1 − score and 1 − y to the fairness metrics, which use "score = P(repay), y = 1 means repaid". Limits are 10,000 rows × 50 features per matrix, 50,000 points per list and a 16 MB body; a maximum-size request takes about 2.3 s.
