"""The checks on real credit data: UCI "Default of Credit Card Clients".

30,000 credit-card holders at a Taiwanese bank, April to September 2005 (Yeh & Lien 2009,
UCI Machine Learning Repository, CC BY 4.0): credit limit, education, marital status, age,
six months of repayment status, bill and payment amounts, and whether the holder defaulted
the following month (22.1%). Sex is the group attribute and is kept out of the model.

There is no time column, so "reference" and "production" windows are made by splitting on a
feature:
- random: a random half each, as a control where nothing should drift;
- age: holders under 35 as reference, 35 and over as production;
- limit: credit limits up to the median (NT$140,000) as reference, above it as production.

In each, the reference rows are split 50/25/25 into model fit, conformal calibration and a
reference test set. The model is the L2 logistic regression `/audit` fits, on log-scaled
amounts. Nothing is tuned on test or production rows: the approval cut-off and the
per-group threshold search use the calibration rows only.

    python -m mrmgov realdata --data "default of credit card clients.xls"
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from .conformal import MondrianConformalPredictor
from .drift import _compute_roc_auc, _fit_logistic_Newton, _sigmoid, analyze_drift
from .fairness import compute_fairness_metrics, repair_group_thresholds

AMOUNTS = ["LIMIT_BAL"] + [f"BILL_AMT{i}" for i in range(1, 7)] + [f"PAY_AMT{i}" for i in range(1, 7)]
FEATURES = ["LIMIT_BAL", "EDUCATION", "MARRIAGE", "AGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"] + AMOUNTS[1:]
TP_PROFIT, FP_COST = 1.0, 1.6  # compute_fairness_metrics' defaults: a repaid loan earns 1, a default loses 1.6


def load(path: str | Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Features (amounts as sign * log1p), default label, sex ("male"/"female"), raw columns."""
    import pandas as pd  # only needed for the real data (.xls needs xlrd)

    df = pd.read_excel(path, header=1) if str(path).endswith(".xls") else pd.read_csv(path)
    raw = {c: df[c].to_numpy(dtype=float) for c in FEATURES + ["SEX"]}
    X = np.column_stack([np.sign(raw[c]) * np.log1p(np.abs(raw[c])) if c in AMOUNTS else raw[c] for c in FEATURES])
    y = df["default payment next month"].to_numpy(dtype=int)
    sex = np.where(raw["SEX"] == 1, "male", "female")
    return X, y, sex, raw


def _fit(X: np.ndarray, y: np.ndarray):
    mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-8
    w, b = _fit_logistic_Newton((X - mu) / sd, y, l2_reg=1.0)
    return lambda Xi: _sigmoid(((np.asarray(Xi, dtype=float) - mu) / sd) @ w + b)


def _coverage(res) -> dict:
    return {"overall_coverage": res.overall_coverage, "abstention": round(res.overall_abstention_rate, 4),
            "by_sex": {r.group_id: {"coverage": round(r.empirical_coverage, 4), "abstention": round(r.abstention_rate, 4),
                                    "n": r.n_eval} for r in res.group_reports}}


def scenario(X, y, sex, ref_mask: np.ndarray, seed: int = 0, alpha: float = 0.05) -> dict:
    rng = np.random.default_rng(seed)
    ref, prod = rng.permutation(np.flatnonzero(ref_mask)), np.flatnonzero(~ref_mask)
    n = len(ref)
    fit, cal, test = ref[: n // 2], ref[n // 2: 3 * n // 4], ref[3 * n // 4:]
    model = _fit(X[fit], y[fit])

    drift = analyze_drift(X[test], y[test], X[prod], y[prod], predict_proba_fn=model, feature_names=FEATURES)

    # What the drift actually costs: refit on half the production rows, compare on the other half.
    half = rng.permutation(prod)
    p_fit, p_eval = half[: len(half) // 2], half[len(half) // 2:]
    refit = _fit(X[p_fit], y[p_fit])

    def brier(m, idx):
        return float(np.mean((y[idx] - m(X[idx])) ** 2))

    conf = MondrianConformalPredictor(alpha=alpha).fit(model(X[cal]), y[cal], sex[cal])
    return {
        "rows": {"fit": len(fit), "calibration": len(cal), "reference_test": len(test), "production": len(prod)},
        "default_rate": {"reference": round(float(y[ref].mean()), 4), "production": round(float(y[prod].mean()), 4)},
        "auc": {"reference_test": round(_compute_roc_auc(y[test], model(X[test])), 4),
                "production": round(_compute_roc_auc(y[prod], model(X[prod])), 4)},
        "drift": {k: drift.to_dict()[k] for k in (
            "regime", "domain_classifier_auc", "max_psi", "brier_ref", "brier_prod", "delta_brier_covariate",
            "delta_brier_concept", "residual_calibration_shift", "covariate_drifted_features", "concept_drifted_features")},
        "refit_on_production": {"brier_reference_model": round(brier(model, p_eval), 5),
                                "brier_production_model": round(brier(refit, p_eval), 5)},
        "conformal": {"alpha": alpha,
                      "reference_test": _coverage(conf.predict(model(X[test]), sex[test], y_test=y[test])),
                      "production": _coverage(conf.predict(model(X[prod]), sex[prod], y_test=y[prod]))},
        "_model": model, "_cal": cal, "_test": test,
    }


def fairness(X, y, sex, sc: dict) -> dict:
    """Sex as the group, female as the protected group. Scores are P(repay); y = 1 means repaid."""
    cal, test, model = sc["_cal"], sc["_test"], sc["_model"]
    s_cal, s_test, r_cal, r_test = 1 - model(X[cal]), 1 - model(X[test]), 1 - y[cal], 1 - y[test]
    grid = np.linspace(0.30, 0.95, 131)
    profit = [(np.sum((s_cal >= t) & (r_cal == 1)) * TP_PROFIT - np.sum((s_cal >= t) & (r_cal == 0)) * FP_COST) for t in grid]
    t = float(grid[int(np.argmax(profit))])
    kw = dict(privileged_group="male", unprivileged_group="female", tp_profit=TP_PROFIT, fp_cost=FP_COST)
    single = compute_fairness_metrics(s_test, r_test, sex[test], threshold_privileged=t, threshold_unprivileged=t, **kw)
    search = repair_group_thresholds(s_cal, r_cal, sex[cal], baseline_threshold=t, **kw)
    th = search.optimal_thresholds
    per_group = compute_fairness_metrics(s_test, r_test, sex[test], threshold_privileged=th["male"],
                                         threshold_unprivileged=th["female"], **kw)
    keep = ("approval_rate_privileged", "approval_rate_unprivileged", "disparate_impact_ratio", "tpr_privileged",
            "tpr_unprivileged", "equal_opportunity_diff", "average_odds_diff", "expected_profit")
    return {"cutoff_from_calibration": round(t, 4),
            "single_cutoff_on_test": {k: single.to_dict()[k] for k in keep},
            "group_thresholds_from_calibration": th, "search_status": search.status,
            "group_thresholds_on_test": {k: per_group.to_dict()[k] for k in keep}}


def experiments(path: str | Path, seed: int = 0, log=print) -> dict:
    X, y, sex, raw = load(path)
    splits = {"random": np.random.default_rng(seed).random(len(y)) < 0.5,
              "age": raw["AGE"] < 35,
              "limit": raw["LIMIT_BAL"] <= np.median(raw["LIMIT_BAL"])}
    res = {"dataset": "UCI Default of Credit Card Clients (30,000 rows, CC BY 4.0)", "rows": len(y),
           "default_rate": round(float(y.mean()), 4), "group_sizes": {g: int(np.sum(sex == g)) for g in ("female", "male")},
           "scenarios": {}}
    for name, mask in splits.items():
        sc = scenario(X, y, sex, mask, seed)
        if name == "random":
            res["fairness"] = fairness(X, y, sex, sc)
            cal, test, model = sc["_cal"], sc["_test"], sc["_model"]
            res["abstention_by_alpha"] = {
                str(a): _coverage(MondrianConformalPredictor(alpha=a).fit(model(X[cal]), y[cal], sex[cal])
                                  .predict(model(X[test]), sex[test], y_test=y[test]))
                for a in (0.05, 0.10, 0.20)}
        res["scenarios"][name] = {k: v for k, v in sc.items() if not k.startswith("_")}
        d = sc["drift"]
        log(f"{name:7s} regime={d['regime']} domain AUC={d['domain_classifier_auc']:.3f} "
            f"AUC ref/prod={sc['auc']['reference_test']}/{sc['auc']['production']} "
            f"coverage ref/prod={sc['conformal']['reference_test']['overall_coverage']}/{sc['conformal']['production']['overall_coverage']}")
    return res


def main(data: str, out: str = "results/real_data.json", seed: int = 0) -> dict:
    res = experiments(data, seed)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(res, indent=2))
    return res
