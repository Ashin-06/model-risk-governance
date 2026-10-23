"""
Comprehensive unit and integration test suite for `mrmgov` (Project 19).
"""

from __future__ import annotations

import copy
import numpy as np
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from fastapi.testclient import TestClient

from mrmgov.drift import (
    compute_psi,
    compute_ks_stat,
    compute_wasserstein_1d,
    analyze_drift,
)
from mrmgov.conformal import MondrianConformalPredictor
from mrmgov.fairness import repair_group_thresholds
from mrmgov.report import GovernanceCertificate, generate_governance_certificate, verify_certificate_signature
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


def test_bad_inputs_raise_value_error() -> None:
    rng = np.random.default_rng(5)
    X, y = rng.normal(size=(50, 3)), rng.integers(0, 2, size=50)
    bad_drift = [
        (X, y, X[:, :2], y),            # feature counts differ
        (X, y[:40], X, y),              # labels don't match rows
        (X, y, np.full((50, 3), np.nan), y),
        (X, y + 1, X, y),               # labels outside {0, 1}
    ]
    for args in bad_drift:
        with pytest.raises(ValueError):
            analyze_drift(*args)
    with pytest.raises(ValueError):
        analyze_drift(X, y, X, y, feature_names=["a"])

    groups = [i % 2 for i in range(50)]
    with pytest.raises(ValueError):
        repair_group_thresholds(rng.random(50), y[:49], groups)
    with pytest.raises(ValueError):
        MondrianConformalPredictor().fit([np.nan] * 50, y, groups)
    with pytest.raises(ValueError):
        MondrianConformalPredictor().fit(rng.random(50), y + 1, groups)


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
    n_cal, n_test, n_repeats = 2000, 2000, 30
    groups_cal = ["prime"] * n_cal + ["subprime"] * n_cal
    groups_test = ["prime"] * n_test + ["subprime"] * n_test

    def draw(n: int) -> tuple[np.ndarray, np.ndarray]:
        p_prime = 1.0 / (1.0 + np.exp(-rng.normal(-1.4, 2.2, size=n)))
        p_sub = 1.0 / (1.0 + np.exp(-rng.normal(0.4, 2.1, size=n)))
        probs = np.concatenate([p_prime, p_sub])
        return probs, rng.binomial(1, probs).astype(int)

    # One split's coverage is noisy (sd ~0.007 per group here), so check the
    # average over fresh calibration/test splits against the guarantee
    coverages: dict[str, list[float]] = {"prime": [], "subprime": []}
    for _ in range(n_repeats):
        probs_cal, y_cal = draw(n_cal)
        probs_test, y_test = draw(n_test)
        cp = MondrianConformalPredictor(alpha=0.05).fit(probs_cal, y_cal, groups_cal)
        res = cp.predict(probs_test, groups_test, y_test=y_test)
        for grp_rep in res.group_reports:
            coverages[grp_rep.group_id].append(grp_rep.empirical_coverage)
            expected_q_level = np.ceil((n_cal + 1) * 0.95) / n_cal
            assert grp_rep.quantile_level == pytest.approx(expected_q_level, abs=1e-6)

    for covs in coverages.values():
        # Continuous scores: 0.95 <= E[coverage] <= 0.95 + 1/(n+1); 3 se of the mean is ~0.004
        assert 0.946 <= float(np.mean(covs)) <= 0.955

    # Verify routing semantics on extreme vs ambiguous probabilities
    probe_res = cp.predict([0.01, 0.99, 0.50], ["prime", "prime", "prime"])
    assert probe_res.prediction_sets[0] == [0]
    assert probe_res.decisions[0] == "AUTO_APPROVE"
    assert probe_res.prediction_sets[1] == [1]
    assert probe_res.decisions[1] == "AUTO_DECLINE"
    assert probe_res.prediction_sets[2] == [0, 1]
    assert probe_res.decisions[2] == "ABSTAIN_HUMAN_UNDERWRITER"

    # 10 calibration points can't support 95% coverage (ceil(11 * 0.95) = 11 > 10),
    # so every set must contain both labels rather than use the 10th score
    tiny = MondrianConformalPredictor(alpha=0.05).fit(
        np.linspace(0.01, 0.30, 10), [0] * 10, ["g"] * 10
    )
    tiny_res = tiny.predict([0.01, 0.99], ["g", "g"])
    assert tiny_res.prediction_sets == [[0, 1], [0, 1]]


def test_ecoa_fairness_four_fifths_repair_and_profit_preservation() -> None:
    bench = run_benchmark_suite(seed=42)
    fair = bench["fairness"]

    assert fair["baseline"]["disparate_impact_ratio"] < 0.80
    assert fair["baseline"]["dir_meets_four_fifths"] is False

    assert fair["repaired"]["disparate_impact_ratio"] >= 0.82
    assert abs(fair["repaired"]["equal_opportunity_diff"]) <= 0.03
    assert fair["repaired"]["dir_meets_four_fifths"] is True
    assert fair["repaired"]["eod_within_limit"] is True
    assert fair["repaired"]["meets_all_targets"] is True

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

    key = Ed25519PrivateKey.generate()
    pub = key.public_key()
    cert = generate_governance_certificate(
        model_id="TEST-MODEL-19",
        model_version="1.0.0",
        drift_result=drift_res,
        conformal_result=conf_res,
        fairness_result=fair_res,
        signing_key=key,
    )
    assert verify_certificate_signature(cert, pub) is True

    # Edited summary or disposition fails
    tampered = copy.deepcopy(cert)
    tampered.fairness_summary["repaired_dir"] = 0.999999
    assert verify_certificate_signature(tampered, pub) is False
    tampered = copy.deepcopy(cert)
    tampered.overall_disposition = "AUTOMATED_CHECKS_PASSED" if cert.overall_disposition != "AUTOMATED_CHECKS_PASSED" else "CHECKS_FAILED"
    assert verify_certificate_signature(tampered, pub) is False

    # Recomputing every hash doesn't help without the signing key
    forged = generate_governance_certificate(
        model_id="TEST-MODEL-19",
        model_version="1.0.0",
        drift_result=drift_res,
        conformal_result=conf_res,
        fairness_result=fair_res,
        signing_key=Ed25519PrivateKey.generate(),
        issued_at_utc=cert.issued_at_utc,
    )
    assert forged.certificate_root_hash == cert.certificate_root_hash
    assert verify_certificate_signature(forged, pub) is False
    tampered = copy.deepcopy(cert)
    tampered.signature = "zz"
    assert verify_certificate_signature(tampered, pub) is False


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
            "groups_ref": [i % 2 for i in range(120)],
            "groups_prod": [i % 2 for i in range(120)],
            "feature_names": ["f1", "f2", "f3"],
        },
    )
    assert r_audit.status_code == 200
    assert r_audit.json()["drift"]["regime"] != "NO_DRIFT"  # X_prod mean shifted by 0.5
    pub = Ed25519PublicKey.from_public_bytes(bytes.fromhex(client.get("/public-key").json()["public_key"]))
    cert = GovernanceCertificate(**r_audit.json()["certificate"])
    assert verify_certificate_signature(cert, pub) is True

    # Test CLI benchmark command
    exit_code = cli_main(["bench"])
    assert exit_code == 0


def test_api_key_and_input_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    import mrmgov.api as api_mod

    client = TestClient(app)
    conf = {
        "probs_calib": [0.1, 0.9], "y_calib": [0, 1], "groups_calib": ["A", "A"],
        "probs_test": [0.2], "groups_test": ["A"],
    }

    monkeypatch.setattr(api_mod, "API_KEY", "s3cret")
    assert client.post("/conformal/predict", json=conf).status_code == 401
    assert client.post("/conformal/predict", json=conf, headers={"X-API-Key": "nope"}).status_code == 401
    assert client.post("/conformal/predict", json=conf, headers={"X-API-Key": "s3cret"}).status_code == 200
    assert client.get("/bench").status_code == 401
    assert client.get("/health").status_code == 200
    assert client.get("/public-key").status_code == 200
    monkeypatch.setattr(api_mod, "API_KEY", None)

    # Out-of-range probabilities, labels and oversized lists are 422s, not 500s
    assert client.post("/conformal/predict", json={**conf, "probs_test": [1.5]}).status_code == 422
    assert client.post("/conformal/predict", json={**conf, "y_calib": [0, 2]}).status_code == 422
    big = {**conf, "probs_test": [0.5] * (api_mod.MAX_POINTS + 1), "groups_test": ["A"] * (api_mod.MAX_POINTS + 1)}
    assert client.post("/conformal/predict", json=big).status_code == 422
    nan_body = '{"X_ref": [[NaN]], "y_ref": [0], "X_prod": [[0]], "y_prod": [0], "groups_ref": [0], "groups_prod": [0]}'
    r = client.post("/audit", content=nan_body, headers={"Content-Type": "application/json"})
    assert r.status_code == 422
    r = client.post("/fairness/repair", content=b"x" * (api_mod.MAX_BODY_BYTES + 1))
    assert r.status_code == 413

    # Ragged rows and mismatched lengths are 400s with a reason
    rows = [[0.0, 1.0]] * 10
    bad = {"X_ref": rows, "y_ref": [0, 1] * 5, "groups_ref": [0] * 10,
           "X_prod": [[0.0]] * 4, "y_prod": [0, 1, 0, 1], "groups_prod": [0] * 4}
    r = client.post("/audit", json=bad)
    assert r.status_code == 400 and "same number of features" in r.json()["detail"]
    assert client.get("/bench", params={"seed": -1}).status_code == 422
