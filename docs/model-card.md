# Model card: the `/audit` reference model and the checks around it

This repository's product is a set of checks, not a credit model. The only model it trains is the
one `/audit` fits so the checks have something to examine. This card describes that model and how
the checks behaved on real data, from `results/real_data.json` and the README.

## What it is

- **Reference model.** L2-regularised logistic regression (Newton's method, λ 1.0) on standardised
  features, amounts log-scaled in the real-data run. `/audit` fits it on a random half of the
  reference rows and calibrates the conformal layer on the other half.
- **Drift check.** Per-feature PSI, KS and Wasserstein-1; a linear logistic domain classifier gives
  importance weights that split the Brier change into covariate and concept parts. Hand-set flags:
  PSI ≥ 0.15, or KS ≥ 0.14 and normalised W1 ≥ 0.25 (covariate); slope shift ≥ 0.06 (concept). The
  per-feature attribution shares are heuristics, not Shapley values.
- **Conformal check.** Mondrian split conformal prediction per group; {0} auto-approve, {1}
  auto-decline, {0, 1} send to a human underwriter.
- **Fairness check.** DIR, SPD, EOD, AOD, F1, AUC and expected profit at given cut-offs, plus a
  per-group threshold search used **only as a diagnostic**: separate cut-offs per protected group
  are disparate treatment under ECOA / Regulation B.

## Intended use

Helping a model-risk team see whether a credit model's inputs or behaviour moved, how many
applicants a conformal layer would send to review, and how approval rates differ between groups.
The signed certificate records what the automated checks found. It is **not a model approval**:
SR 11-7 expects independent validation and sign-off by people with that authority, and the
frameworks the certificate names are references only.

## Data

**Generated** (`python -m mrmgov bench`): planted covariate and concept drift, calibrated scores,
and a constant logit shift on one group. These are consistency checks, not measurements; the
scenarios plant exactly what the detectors look for.

**Real data** (`python -m mrmgov realdata`): UCI's Default of Credit Card Clients, 30,000 card
holders at a Taiwanese bank in 2005, 22.1% of whom defaulted the next month (CC BY 4.0); 18,112
women, 11,888 men.
- The data has no time column, so reference and production windows are made by splitting on a
  feature: under 35 vs 35 and over (age), credit limit up to the median vs above (limit). A random
  half-and-half split is the control.
- Sex is the group attribute and is kept out of the model. Reference rows are divided 50/25/25
  into model fit, conformal calibration and a reference test set. The approval cut-off and the
  group-threshold search use the calibration rows only.
- One run (seed 0); nothing was tuned after seeing the results.

## Metrics (real data)

| Split | Regime | Covariate-drifted features | ROC-AUC ref → prod | Brier ref → prod | Coverage at α 0.05, ref → prod | Abstention ref → prod |
|---|---|---|---|---|---|---|
| random (control) | `NO_DRIFT` | none | 0.7437 → 0.7481 | 0.1422 → 0.1441 | 95.30% → 95.44% | 56.8% → 56.5% |
| age | `COVARIATE_DRIFT` | MARRIAGE, AGE | 0.7396 → 0.7375 | 0.1374 → 0.1468 | 94.88% → **93.82%** | 52.1% → 48.7% |
| limit | `COVARIATE_DRIFT` | 21 of 22 | 0.7457 → **0.6658** | 0.1670 → 0.1167 | 95.63% → **92.16%** | 70.7% → 25.5% |

- Domain-classifier AUC: 0.517 (control), 1.00 (age), 1.00 (limit). The split feature separates the
  windows perfectly, so catching these splits is easy.
- Refitting on half the production rows does no better on the other half than the reference
  model (Brier 0.1479 vs 0.1479 for age, 0.1115 vs 0.1121 for limit), which agrees with "no concept
  drift".
- Abstention at the calibration split: 56.8% at α 0.05 (64.0% of men, 51.9% of women), 25.2% at
  α 0.10, 3.4% at α 0.20. On generated data it is 42.4% at α 0.05.
- Fairness at the single most profitable cut-off on the calibration rows (approve when
  P(repay) ≥ 0.625): 85.1% of women and 83.2% of men approved, DIR 1.02, EOD +0.002. Women in this
  data default less (20.8% vs 24.2%). The group-threshold search finds almost nothing to change
  (DIR 1.01).

Generated-data benchmark (seed 42; seeds 0–39 for the ranges): covariate drift identified exactly
on 40/40 seeds, concept drift on 38/40, conformal mean-coverage check passed on 39/40; per-group
coverage 94.85–95.09% (target 95%); DIR 0.629 → 0.878 with group thresholds, EOD −0.188 → −0.018.

## Known limitations and failure modes

- **The drift check misses a real loss of accuracy.** In the limit split ROC-AUC falls from 0.746
  to 0.666, yet the regime is plain covariate drift and the Brier score improves, because
  high-limit holders default less (15.7% vs 28.3%) and are harder to rank. Watch AUC on labelled
  production data, not only the regime and Brier.
- **The covariate/concept split means little when windows don't overlap**: the density-ratio
  weights are degenerate. The linear domain classifier mostly sees mean shifts, so a shift in
  variance or interactions can be misattributed to concept drift.
- **Conformal coverage needs exchangeability**, which drift breaks: coverage fell to 93.8% and
  92.2% in production, well outside sampling error (about 0.2 points on 14,000 rows). Groups unseen
  in calibration fall back to the pooled threshold, which has no per-group guarantee. Coverage is
  marginal within a group, not per applicant.
- **Fairness metrics need outcomes for everyone.** In lending, repayment is observed only for
  approved applicants, so EOD on real data needs reject inference. The UCI data has outcomes for
  every holder, which real portfolios don't.
- **One real dataset, no real time order.** Feature-split windows are starker than drift over time.
- **Thresholds are hand-set**, and the generated benchmark flagged an extra feature on 2 of 40
  seeds.
- **Not reviewed by a model-risk or fair-lending specialist.**
