# Evaluation

## Generated data

`python -m mrmgov bench` (seed 42):

| Check | Result |
|---|---|
| sample with no drift | `NO_DRIFT`, no features flagged |
| covariate drift planted on 2 features | `COVARIATE_DRIFT`, exactly those 2 flagged |
| concept drift planted on 2 coefficients | `CONCEPT_DRIFT`, exactly those 2 flagged |
| conformal coverage, α = 0.05, 3 groups, mean over 50 splits | 94.85–95.09% per group (target 95%) |
| applicants sent to review | 42.4% |
| DIR at one cut-off → with group thresholds | 0.629 → 0.878 |
| EOD at one cut-off → with group thresholds | −0.188 → −0.018 |
| certificate signature verifies; edited and re-signed copies rejected | yes |

Over seeds 0–39, covariate drift was identified exactly every time and concept drift on 38 of 40 (twice a third feature crossed the threshold). The coverage check passed on 39 of 40, as expected for a check allowing 3 standard errors per group.

These are consistency checks: the scenarios plant exactly the shifts the detectors look for, the scores are calibrated by construction, and the "bias" is a constant shift that a threshold change undoes exactly. The 42% review rate is the more telling number: with scores this uncertain, 95% coverage means a person reviews almost half the applicants.

## Real credit data

`python -m mrmgov realdata --data "default of credit card clients.xls"` runs the checks on UCI's Default of Credit Card Clients data (CC BY 4.0): 30,000 card holders at a Taiwanese bank in 2005, 22.1% of whom defaulted the next month. Results go to `results/real_data.json`.

The data has no time column, so reference and production windows are made by splitting on a feature: card holders under 35 against 35 and over, and credit limits up to the median against above it. A random half-and-half split is the control. The model is the logistic regression `/audit` fits; sex is the group attribute and is kept out of the model.

| Split | Result | Covariate-drifted features | ROC-AUC ref → prod | Coverage at α = 0.05, ref → prod |
|---|---|---|---|---|
| random (control) | `NO_DRIFT` | none | 0.744 → 0.748 | 95.3% → 95.4% |
| age | `COVARIATE_DRIFT` | AGE, MARRIAGE | 0.740 → 0.738 | 94.9% → 93.8% |
| credit limit | `COVARIATE_DRIFT` | 21 of 22 | 0.746 → 0.666 | 95.6% → 92.2% |

- No false alarm on the control, and both feature splits are caught as covariate drift.
- No concept drift was flagged, and a refitted model confirms there was none to find.
- The drift check misses a real loss of accuracy: in the credit-limit split, ROC-AUC falls from 0.746 to 0.666 while the Brier score improves, because high-limit holders default less and are harder to rank. A monitor watching only the drift verdict and Brier score would not notice.
- Conformal coverage breaks under drift, as theory says it can: 93.8% and 92.2% in production, well outside sampling error.
- Review is heavy on real scores: at α = 0.05, 57% of applicants go to an underwriter; at α = 0.10, 25%.
- At a single cut-off there is no adverse impact against women: 85.1% of women and 83.2% of men are approved (DIR 1.02). Women in this data default less often.

These come from one run, and nothing was tuned after seeing them.
