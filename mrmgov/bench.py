"""
Benchmark on generated data. Nothing here is real credit data.

Each scenario is drawn from a known data-generating process, so these are
consistency checks that the code finds what was planted, not accuracy measurements:
1. Drift: a no-drift sample, a mean shift on two features (covariate drift) and two
   flipped coefficients (concept drift); checks the flagged features match.
2. Conformal: three synthetic score cohorts; checks mean per-group coverage over
   repeated calibration/test splits against the 1 - alpha guarantee.
3. Fairness: a legacy score shifted down for one group; checks the baseline is under
   the 0.80 ratio and the group-threshold search finds thresholds meeting DIR >= 0.82
   and |EOD| <= 0.03. The bias is a constant logit shift, which a threshold shift
   undoes exactly, so the search succeeding is expected by construction.
4. Certificate: hash chain and signature verify.
"""

from __future__ import annotations

from typing import Dict
import numpy as np
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from .drift import analyze_drift, _sigmoid
from .conformal import MondrianConformalPredictor
from .fairness import repair_group_thresholds
from .report import generate_governance_certificate, verify_certificate_signature


FEATURE_NAMES = [
    "debt_to_income",
    "credit_utilization",
    "credit_history_months",
    "delinquency_count",
    "income_stability",
]


def _true_credit_prob(X: np.ndarray, weights: np.ndarray, bias: float = -0.35) -> np.ndarray:
    return _sigmoid(X @ weights + bias)


def run_benchmark_suite(seed: int = 42) -> Dict[str, object]:
    """
    Execute the complete SR 11-7 / RBI Model Governance benchmark suite.
    """
    rng = np.random.default_rng(seed)
    n_samples = 2400
    n_features = len(FEATURE_NAMES)

    # True reference structural weights for P_ref(Y=1 | X)
    w_ref = np.array([1.25, 1.10, -0.85, 0.95, -0.75], dtype=float)
    b_ref = -0.25

    X_ref = rng.normal(loc=0.0, scale=1.0, size=(n_samples, n_features))
    p_ref_true = _true_credit_prob(X_ref, w_ref, b_ref)
    y_ref = rng.binomial(1, p_ref_true, size=n_samples).astype(int)

    def ref_model_proba(X_in: np.ndarray) -> np.ndarray:
        return _true_credit_prob(np.asarray(X_in, dtype=float), w_ref, b_ref)

    # Scenario A: No Drift (Independent sample from identical data-generating process)
    X_clean = rng.normal(loc=0.0, scale=1.0, size=(n_samples, n_features))
    y_clean = rng.binomial(1, _true_credit_prob(X_clean, w_ref, b_ref), size=n_samples).astype(int)
    drift_clean = analyze_drift(
        X_ref=X_ref,
        y_ref=y_ref,
        X_prod=X_clean,
        y_prod=y_clean,
        predict_proba_fn=ref_model_proba,
        feature_names=FEATURE_NAMES,
    )

    # Scenario B: Pure Covariate Drift on ["debt_to_income", "credit_utilization"]
    # Input distribution P_prod(X) shifts on features 0 and 1, while P_prod(Y|X) = P_ref(Y|X)
    X_cov = rng.normal(loc=0.0, scale=1.0, size=(n_samples, n_features))
    X_cov[:, 0] += 0.95  # debt_to_income shift
    X_cov[:, 1] += 0.85  # credit_utilization shift
    y_cov = rng.binomial(1, _true_credit_prob(X_cov, w_ref, b_ref), size=n_samples).astype(int)
    drift_covariate = analyze_drift(
        X_ref=X_ref,
        y_ref=y_ref,
        X_prod=X_cov,
        y_prod=y_cov,
        predict_proba_fn=ref_model_proba,
        feature_names=FEATURE_NAMES,
    )

    # Scenario C: Pure Concept Drift on ["debt_to_income", "delinquency_count"]
    # Marginal P_prod(X) == P_ref(X), but conditional structural relationship P_prod(Y|X) shifts
    X_conc = X_ref.copy() + rng.normal(0.0, 0.02, size=X_ref.shape)
    w_conc = w_ref.copy()
    w_conc[0] = -1.15  # structural shock on debt_to_income relationship
    w_conc[3] = 2.65   # structural surge on delinquency_count penalty
    y_conc = rng.binomial(1, _true_credit_prob(X_conc, w_conc, b_ref + 0.35), size=n_samples).astype(int)
    drift_concept = analyze_drift(
        X_ref=X_ref,
        y_ref=y_ref,
        X_prod=X_conc,
        y_prod=y_conc,
        predict_proba_fn=ref_model_proba,
        feature_names=FEATURE_NAMES,
    )

    injected_cov = {"debt_to_income", "credit_utilization"}
    injected_conc = {"debt_to_income", "delinquency_count"}
    cov_exact_match = set(drift_covariate.covariate_drifted_features) == injected_cov
    conc_exact_match = set(drift_concept.concept_drifted_features) == injected_conc

    # 2. Mondrian Group-Conditional Split Conformal Prediction Benchmark
    # Three synthetic score cohorts with different base rates; scores are calibrated
    # by construction (y ~ Bernoulli(p)), so this tests the conformal layer only
    group_params = {
        "group_A_prime": (-1.65, 2.35),
        "group_B_near_prime": (-0.65, 2.25),
        "group_C_thin_file": (0.35, 2.15),
    }
    n_per_group = 1200
    groups_split = [grp for grp in group_params for _ in range(n_per_group)]

    def draw_split() -> tuple[np.ndarray, np.ndarray]:
        p = np.concatenate([_sigmoid(rng.normal(loc, scale, size=n_per_group))
                            for loc, scale in group_params.values()])
        return p, rng.binomial(1, p).astype(int)

    # One split's coverage has sd ~0.009 per group here, so a single draw can't
    # confirm a 95% guarantee; average over repeated calibration/test splits instead
    n_splits = 50
    alpha = 0.05
    split_coverages: Dict[str, list] = {grp: [] for grp in group_params}
    conformal_res = None
    for _ in range(n_splits):
        probs_cal, y_cal = draw_split()
        probs_test, y_test = draw_split()
        mondrian = MondrianConformalPredictor(alpha=alpha).fit(probs_cal, y_cal, groups_split)
        res = mondrian.predict(probs_test, groups_split, y_test=y_test)
        conformal_res = conformal_res or res  # report the first split in full
        for r in res.group_reports:
            split_coverages[r.group_id].append(r.empirical_coverage)

    mean_group_coverage = {g: float(np.mean(c)) for g, c in split_coverages.items()}
    # Continuous scores: 1 - alpha <= E[coverage] <= 1 - alpha + 1/(n+1)
    coverage_ok = all(
        1 - alpha - 3 * np.std(c) / np.sqrt(n_splits)
        <= mean_group_coverage[g]
        <= 1 - alpha + 1 / (n_per_group + 1) + 3 * np.std(c) / np.sqrt(n_splits)
        for g, c in split_coverages.items()
    )
    min_group_coverage = min(r.empirical_coverage for r in conformal_res.group_reports)

    # 3. ECOA / Fair-Lending Disparate Impact Auditor & Constrained Threshold Repair
    # Simulate creditworthiness scoring where unprivileged group (0) experiences measurement
    # downward bias in legacy credit score despite solid repayment capacity
    n_priv = 1600
    n_unpriv = 1600
    groups_fair = np.array([1] * n_priv + [0] * n_unpriv, dtype=int)

    latent_credit_priv = rng.normal(loc=0.35, scale=1.15, size=n_priv)
    latent_credit_unpriv = rng.normal(loc=0.10, scale=1.15, size=n_unpriv)

    # True repayment probability Y=1 (creditworthy)
    prob_repay_priv = _sigmoid(1.55 * latent_credit_priv + 0.15)
    prob_repay_unpriv = _sigmoid(1.55 * latent_credit_unpriv + 0.15)
    y_fair = np.concatenate(
        [
            rng.binomial(1, prob_repay_priv),
            rng.binomial(1, prob_repay_unpriv),
        ]
    ).astype(int)

    # Biased legacy model score shifts unprivileged group downward by -0.68 logits
    score_priv = _sigmoid(1.45 * latent_credit_priv + 0.25 + rng.normal(0, 0.15, size=n_priv))
    score_unpriv = _sigmoid(
        1.45 * latent_credit_unpriv - 0.43 + rng.normal(0, 0.15, size=n_unpriv)
    )
    scores_fair = np.concatenate([score_priv, score_unpriv])

    fairness_res = repair_group_thresholds(
        scores=scores_fair,
        y_true=y_fair,
        groups=groups_fair,
        privileged_group=1,
        unprivileged_group=0,
        baseline_threshold=0.50,
        min_dir=0.80,
        target_dir_floor=0.83,
        max_abs_eod=0.03,
        tp_profit=1.0,
        fp_cost=1.25,
    )

    # 4. Certificate for the covariate-drift run, signed with a throwaway key
    signing_key = Ed25519PrivateKey.generate()
    certificate = generate_governance_certificate(
        model_id="CREDIT-UNDERWRITER-V4.2",
        model_version="2026.10-SR117",
        drift_result=drift_covariate,
        conformal_result=conformal_res,
        fairness_result=fairness_res,
        signing_key=signing_key,
        issued_at_utc="2026-10-22T09:30:00Z",
    )
    signature_valid = verify_certificate_signature(certificate, signing_key.public_key())

    assertions = {
        "clean_no_drift_verified": drift_clean.regime == "NO_DRIFT"
        and len(drift_clean.drifted_features) == 0,
        "covariate_drift_100pct_identified": cov_exact_match
        and drift_covariate.regime == "COVARIATE_DRIFT",
        "concept_drift_100pct_identified": conc_exact_match
        and drift_concept.regime == "CONCEPT_DRIFT",
        "mondrian_mean_coverage_matches_guarantee": coverage_ok,
        "baseline_dir_below_080": fairness_res.baseline.disparate_impact_ratio < 0.80,
        "repaired_dir_ge_082": fairness_res.repaired.disparate_impact_ratio >= 0.82,
        "repaired_eod_le_003": abs(fairness_res.repaired.equal_opportunity_diff) <= 0.03,
        "utility_preserved_gt_97pct": fairness_res.auc_retention_ratio >= 0.97
        and fairness_res.profit_retention_ratio >= 0.97,
        "certificate_signature_valid": signature_valid,
    }

    return {
        "benchmark_passed": all(assertions.values()),
        "assertions": assertions,
        "drift_scenarios": {
            "no_drift": drift_clean.to_dict(),
            "covariate_drift": drift_covariate.to_dict(),
            "concept_drift": drift_concept.to_dict(),
        },
        "conformal": {
            "n_splits": n_splits,
            "mean_group_coverage": {g: round(c, 6) for g, c in mean_group_coverage.items()},
            "min_group_coverage": round(float(min_group_coverage), 6),
            "result": conformal_res.to_dict(),
        },
        "fairness": fairness_res.to_dict(),
        "certificate": certificate.to_dict(),
    }
