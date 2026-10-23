"""
Comprehensive unit and integration test suite for `mrmgov` (Project 19).
"""

from __future__ import annotations

import copy
import numpy as np
import pytest
from fastapi.testclient import TestClient

from mrmgov.drift import (
    compute_psi,
    compute_ks_stat,
    compute_wasserstein_1d,
    analyze_drift,
)
from mrmgov.conformal import MondrianConformalPredictor
from mrmgov.fairness import compute_fairness_metrics, repair_group_thresholds
from mrmgov.report import generate_governance_certificate, verify_certificate_signature
from mrmgov.bench import run_benchmark_suite
from mrmgov.api import app
from mrmgov.__main__ import main as cli_main


def test_psi_ks_wasserstein_identities_and_monotonicity() -> None:
    rng = np.random.default_rng(101)
    ref = rng.normal(0.0, 1.0, size=1500)
    prod_same = ref.copy()
    prod_moderate = rng.normal(0.45, 1.0, size=1500)
    prod_severe = rng.normal(1.25, 1.15, size=1500)

    assert compute_psi(ref, prod_same) == pytest.approx(0.0, abs=1e-9)
    ks_0, p_0 = compute_ks_stat(ref, prod_same)
    assert ks_0 == pytest.approx(0.0, abs=1e-9)
    assert p_0 == pytest.approx(1.0, abs=1e-6)
    w1_0, w1n_0 = compute_wasserstein_1d(ref, prod_same)
    assert w1_0 == pytest.approx(0.0, abs=1e-9)
    assert w1n_0 == pytest.approx(0.0, abs=1e-9)

    psi_mod = compute_psi(ref, prod_moderate)
    psi_sev = compute_psi(ref, prod_severe)
    assert 0.08 < psi_mod < psi_sev
    assert psi_sev > 0.45

    ks_mod, p_mod = compute_ks_stat(ref, prod_moderate)
    ks_sev, p_sev = compute_ks_stat(ref, prod_severe)
    assert 0.10 < ks_mod < ks_sev
    assert p_sev < p_mod < 0.01

    w1_mod, _ = compute_wasserstein_1d(ref, prod_moderate)
    w1_sev, _ = compute_wasserstein_1d(ref, prod_severe)
    assert 0.30 < w1_mod < w1_sev

    # Production windows are rarely the same size as the reference window
    assert compute_psi(ref, ref[:700]) < 0.05
    assert compute_psi(ref, prod_severe[:700]) > 0.45


def test_covariate_vs_concept_drift_decomposition_and_attribution() -> None:
    bench = run_benchmark_suite(seed=42)
    no_drift = bench["drift_scenarios"]["no_drift"]
    cov_drift = bench["drift_scenarios"]["covariate_drift"]
    conc_drift = bench["drift_scenarios"]["concept_drift"]

    assert no_drift["regime"] == "NO_DRIFT"
    assert no_drift["drifted_features"] == []

    # Exact additive Brier decomposition check: delta_total == delta_cov + delta_concept
    for d in (no_drift, cov_drift, conc_drift):
        assert d["delta_brier_covariate"] + d["delta_brier_concept"] == pytest.approx(
            d["delta_brier_total"], abs=1e-5
        )
        cov_shares = [m["covariate_shapley_share"] for m in d["feature_metrics"]]
        assert sum(cov_shares) == pytest.approx(1.0, abs=1e-4)

    # 100% identification of injected covariate-drifted features
    assert cov_drift["regime"] == "COVARIATE_DRIFT"
    assert set(cov_drift["covariate_drifted_features"]) == {
        "debt_to_income",
        "credit_utilization",
    }
    assert cov_drift["concept_drifted_features"] == []

    # 100% identification of injected concept-drifted features
    assert conc_drift["regime"] == "CONCEPT_DRIFT"
    assert set(conc_drift["concept_drifted_features"]) == {
        "debt_to_income",
        "delinquency_count",
    }
    assert conc_drift["covariate_drifted_features"] == []


def test_mondrian_conformal_finite_sample_coverage_and_abstention() -> None:
    rng = np.random.default_rng(42)
    n_cal, n_test = 2000, 2000
    groups_cal = ["prime"] * n_cal + ["subprime"] * n_cal
    groups_test = ["prime"] * n_test + ["subprime"] * n_test

    p_cal_prime = 1.0 / (1.0 + np.exp(-rng.normal(-1.4, 2.2, size=n_cal)))
    p_cal_sub = 1.0 / (1.0 + np.exp(-rng.normal(0.4, 2.1, size=n_cal)))
    probs_cal = np.concatenate([p_cal_prime, p_cal_sub])
    y_cal = rng.binomial(1, probs_cal).astype(int)

    p_test_prime = 1.0 / (1.0 + np.exp(-rng.normal(-1.4, 2.2, size=n_test)))
    p_test_sub = 1.0 / (1.0 + np.exp(-rng.normal(0.4, 2.1, size=n_test)))
    probs_test = np.concatenate([p_test_prime, p_test_sub])
    y_test = rng.binomial(1, probs_test).astype(int)

    cp = MondrianConformalPredictor(alpha=0.05)
    cp.fit(probs_cal, y_cal, groups_cal)
    res = cp.predict(probs_test, groups_test, y_test=y_test)

    assert res.overall_coverage is not None and res.overall_coverage >= 0.945
    for grp_rep in res.group_reports:
        assert grp_rep.empirical_coverage >= 0.945
        assert grp_rep.is_valid_coverage is True
        expected_q_level = min(1.0, np.ceil((n_cal + 1) * 0.95) / n_cal)
        assert grp_rep.quantile_level == pytest.approx(expected_q_level, abs=1e-6)

    # Verify routing semantics on extreme vs ambiguous probabilities
    probe_res = cp.predict([0.01, 0.99, 0.50], ["prime", "prime", "prime"])
    assert probe_res.prediction_sets[0] == [0]
    assert probe_res.decisions[0] == "AUTO_APPROVE"
    assert probe_res.prediction_sets[1] == [1]
    assert probe_res.decisions[1] == "AUTO_DECLINE"
    assert probe_res.prediction_sets[2] == [0, 1]
    assert probe_res.decisions[2] == "ABSTAIN_HUMAN_UNDERWRITER"


def test_ecoa_fairness_four_fifths_repair_and_profit_preservation() -> None:
    bench = run_benchmark_suite(seed=42)
    fair = bench["fairness"]

    assert fair["baseline"]["disparate_impact_ratio"] < 0.65
    assert fair["baseline"]["four_fifths_compliant"] is False

    assert fair["repaired"]["disparate_impact_ratio"] >= 0.82
    assert abs(fair["repaired"]["equal_opportunity_diff"]) <= 0.03
    assert fair["repaired"]["four_fifths_compliant"] is True
    assert fair["repaired"]["eod_compliant"] is True
    assert fair["repaired"]["overall_compliant"] is True

    assert fair["auc_retention_ratio"] >= 0.97
    assert fair["profit_retention_ratio"] >= 0.97


def test_governance_certificate_hash_chain_and_tamper_detection() -> None:
    rng = np.random.default_rng(19)
    X_ref = rng.normal(size=(300, 3))
    y_ref = (X_ref[:, 0] + rng.normal(0, 0.5, size=300) > 0).astype(int)
    X_prod = rng.normal(size=(300, 3))
    y_prod = (X_prod[:, 0] + rng.normal(0, 0.5, size=300) > 0).astype(int)

    drift_res = analyze_drift(X_ref, y_ref, X_prod, y_prod)
    probs = 1.0 / (1.0 + np.exp(-X_prod[:, 0]))
    groups = [i % 2 for i in range(300)]

    cp = MondrianConformalPredictor(alpha=0.05).fit(probs, y_prod, groups)
    conf_res = cp.predict(probs, groups, y_test=y_prod)
    fair_res = repair_group_thresholds(probs, y_prod, groups)

    cert = generate_governance_certificate(
        model_id="TEST-MODEL-19",
        model_version="1.0.0",
        drift_result=drift_res,
        conformal_result=conf_res,
        fairness_result=fair_res,
    )
    assert verify_certificate_signature(cert) is True

    # Tamper with fairness summary and verify signature check fails
    tampered = copy.deepcopy(cert)
    tampered.fairness_summary["repaired_dir"] = 0.999999
    assert verify_certificate_signature(tampered) is False


def test_fastapi_endpoints_and_cli() -> None:
    client = TestClient(app)

    r_root = client.get("/")
    assert r_root.status_code == 200
    assert "SR 11-7 / RBI Model Risk Governance" in r_root.text

    r_health = client.get("/health")
    assert r_health.status_code == 200
    assert r_health.json()["port"] == 8019

    r_bench = client.get("/bench")
    assert r_bench.status_code == 200
    b_data = r_bench.json()
    assert b_data["benchmark_passed"] is True

    # Test POST /conformal/predict
    r_conf = client.post(
        "/conformal/predict",
        json={
            "alpha": 0.05,
            "probs_calib": [0.05, 0.12, 0.88, 0.94, 0.08, 0.91],
            "y_calib": [0, 0, 1, 1, 0, 1],
            "groups_calib": ["A", "A", "A", "B", "B", "B"],
            "probs_test": [0.04, 0.50, 0.96],
            "groups_test": ["A", "A", "B"],
            "y_test": [0, 1, 1],
        },
    )
    assert r_conf.status_code == 200
    assert r_conf.json()["overall_coverage"] == 1.0

    # Test POST /fairness/repair
    rng = np.random.default_rng(12)
    scores = np.clip(rng.normal(0.55, 0.2, size=200), 0.01, 0.99).tolist()
    y_true = [1 if s > 0.5 else 0 for s in scores]
    groups = [1] * 100 + [0] * 100
    r_fair = client.post(
        "/fairness/repair",
        json={"scores": scores, "y_true": y_true, "groups": groups},
    )
    assert r_fair.status_code == 200
    assert "repaired" in r_fair.json()

    # Test POST /audit
    X_r = rng.normal(size=(120, 3)).tolist()
    y_r = [int(row[0] > 0) for row in X_r]
    X_p = rng.normal(loc=0.5, size=(120, 3)).tolist()
    y_p = [int(row[0] > 0) for row in X_p]
    r_audit = client.post(
        "/audit",
        json={
            "model_id": "API-AUDIT-01",
            "X_ref": X_r,
            "y_ref": y_r,
            "X_prod": X_p,
            "y_prod": y_p,
            "feature_names": ["f1", "f2", "f3"],
        },
    )
    assert r_audit.status_code == 200
    assert r_audit.json()["signature_verified"] is True

    # Test CLI benchmark command
    exit_code = cli_main(["bench"])
    assert exit_code == 0
