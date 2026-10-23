"""
SR 11-7 (Federal Reserve / OCC) & RBI Model Risk Governance Audit Certificate Engine.

Produces tamper-evident, SHA-256 hash-chained governance certificates combining:
  - Block 0 (Genesis Header): Model ID, version, regulator framework, timestamp
  - Block 1 (Drift Attribution): Covariate vs Concept drift decomposition
  - Block 2 (Mondrian Conformal): Group-conditional coverage & abstention verification
  - Block 3 (ECOA Fair-Lending): Disparate Impact Four-Fifths & Equal Opportunity repair
  - Root Sign-off: Deterministic SHA-256 Merkle/Chain digest
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from typing import Dict, List, Optional

from .drift import DriftDecompositionResult
from .conformal import ConformalPredictionResult
from .fairness import FairnessRepairResult


def _canonical_sha256(prev_hash: str, payload: Dict[str, object]) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    msg = f"{prev_hash}|{serialized}".encode("utf-8")
    return hashlib.sha256(msg).hexdigest()


@dataclass
class GovernanceCertificate:
    """Hash-chained SR 11-7 / RBI Model Governance Audit Certificate."""

    certificate_id: str
    model_id: str
    model_version: str
    regulatory_frameworks: List[str]
    issued_at_utc: str
    governance_tier: str
    overall_disposition: str  # "APPROVED_FOR_PRODUCTION" | "CONDITIONAL_APPROVAL_WITH_GUARDRAILS" | "QUARANTINED"
    checklist: Dict[str, bool]
    drift_summary: Dict[str, object]
    conformal_summary: Dict[str, object]
    fairness_summary: Dict[str, object]
    hash_chain: Dict[str, str]
    certificate_root_hash: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def generate_governance_certificate(
    model_id: str,
    model_version: str,
    drift_result: DriftDecompositionResult,
    conformal_result: ConformalPredictionResult,
    fairness_result: FairnessRepairResult,
    issued_at_utc: Optional[str] = None,
    governance_tier: str = "TIER_1_MATERIAL_CREDIT_UNDERWRITING",
) -> GovernanceCertificate:
    """
    Generate a deterministic, hash-chained SR 11-7 / RBI Model Governance Certificate.
    """
    if issued_at_utc is None:
        issued_at_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    frameworks = [
        "FED_OCC_SR_11_7",
        "RBI_MRM_2024_GUIDELINES",
        "ECOA_REGULATION_B_FOUR_FIFTHS",
        "EU_AI_ACT_HIGH_RISK_CREDIT",
    ]

    header_payload: Dict[str, object] = {
        "model_id": model_id,
        "model_version": model_version,
        "regulatory_frameworks": frameworks,
        "issued_at_utc": issued_at_utc,
        "governance_tier": governance_tier,
    }

    drift_summary: Dict[str, object] = {
        "regime": drift_result.regime,
        "mean_psi": drift_result.mean_psi,
        "max_psi": drift_result.max_psi,
        "domain_classifier_auc": drift_result.domain_classifier_auc,
        "delta_brier_total": drift_result.delta_brier_total,
        "delta_brier_covariate": drift_result.delta_brier_covariate,
        "delta_brier_concept": drift_result.delta_brier_concept,
        "covariate_drifted_features": drift_result.covariate_drifted_features,
        "concept_drifted_features": drift_result.concept_drifted_features,
        "drifted_features": drift_result.drifted_features,
        "recommended_action": drift_result.recommended_action,
    }

    all_groups_covered = all(g.is_valid_coverage for g in conformal_result.group_reports)
    min_group_cov = min(
        (g.empirical_coverage for g in conformal_result.group_reports),
        default=conformal_result.target_coverage,
    )
    conformal_summary: Dict[str, object] = {
        "alpha": conformal_result.alpha,
        "target_coverage": conformal_result.target_coverage,
        "overall_coverage": conformal_result.overall_coverage,
        "min_group_coverage": round(float(min_group_cov), 6),
        "overall_abstention_rate": conformal_result.overall_abstention_rate,
        "overall_singleton_accuracy": conformal_result.overall_singleton_accuracy,
        "group_thresholds": conformal_result.group_thresholds,
        "all_groups_valid_coverage": all_groups_covered,
    }

    fairness_summary: Dict[str, object] = {
        "baseline_dir": fairness_result.baseline.disparate_impact_ratio,
        "repaired_dir": fairness_result.repaired.disparate_impact_ratio,
        "baseline_eod": fairness_result.baseline.equal_opportunity_diff,
        "repaired_eod": fairness_result.repaired.equal_opportunity_diff,
        "optimal_thresholds": fairness_result.optimal_thresholds,
        "profit_retention_ratio": fairness_result.profit_retention_ratio,
        "f1_retention_ratio": fairness_result.f1_retention_ratio,
        "auc_retention_ratio": fairness_result.auc_retention_ratio,
        "dir_meets_four_fifths": fairness_result.repaired.dir_meets_four_fifths,
        "eod_within_limit": fairness_result.repaired.eod_within_limit,
        "meets_all_targets": fairness_result.repaired.meets_all_targets,
        "status": fairness_result.status,
    }

    genesis_hash = _canonical_sha256("0" * 64, header_payload)
    drift_hash = _canonical_sha256(genesis_hash, drift_summary)
    conformal_hash = _canonical_sha256(drift_hash, conformal_summary)
    fairness_hash = _canonical_sha256(conformal_hash, fairness_summary)

    checklist = {
        "sr11_7_drift_attributed": True,
        "no_unmitigated_concept_drift": drift_result.regime
        not in ("CONCEPT_DRIFT", "COMPOUND_DRIFT"),
        "mondrian_conformal_coverage_verified": bool(all_groups_covered),
        "dir_meets_four_fifths": bool(fairness_result.repaired.dir_meets_four_fifths),
        "eod_within_limit": bool(fairness_result.repaired.eod_within_limit),
        "utility_retention_above_97pct": bool(
            max(
                fairness_result.auc_retention_ratio,
                fairness_result.profit_retention_ratio,
            )
            >= 0.97
        ),
    }

    root_payload: Dict[str, object] = {
        "checklist": checklist,
        "fairness_hash": fairness_hash,
    }
    root_hash = _canonical_sha256(fairness_hash, root_payload)
    cert_id = f"SR117-CERT-{root_hash[:12].upper()}"

    if (
        checklist["no_unmitigated_concept_drift"]
        and checklist["mondrian_conformal_coverage_verified"]
        and checklist["dir_meets_four_fifths"]
    ):
        disposition = "APPROVED_FOR_PRODUCTION"
    elif (
        checklist["mondrian_conformal_coverage_verified"]
        and checklist["dir_meets_four_fifths"]
    ):
        disposition = "CONDITIONAL_APPROVAL_WITH_GUARDRAILS"
    else:
        disposition = "QUARANTINED_PENDING_REMEDIATION"

    hash_chain = {
        "genesis_header_hash": genesis_hash,
        "drift_block_hash": drift_hash,
        "conformal_block_hash": conformal_hash,
        "fairness_block_hash": fairness_hash,
        "certificate_root_hash": root_hash,
    }

    return GovernanceCertificate(
        certificate_id=cert_id,
        model_id=model_id,
        model_version=model_version,
        regulatory_frameworks=frameworks,
        issued_at_utc=issued_at_utc,
        governance_tier=governance_tier,
        overall_disposition=disposition,
        checklist=checklist,
        drift_summary=drift_summary,
        conformal_summary=conformal_summary,
        fairness_summary=fairness_summary,
        hash_chain=hash_chain,
        certificate_root_hash=root_hash,
    )


def verify_certificate_signature(cert: GovernanceCertificate) -> bool:
    """
    Recompute the entire SHA-256 hash chain from certificate contents and verify
    that no block or summary metric has been tampered with.
    """
    header_payload: Dict[str, object] = {
        "model_id": cert.model_id,
        "model_version": cert.model_version,
        "regulatory_frameworks": cert.regulatory_frameworks,
        "issued_at_utc": cert.issued_at_utc,
        "governance_tier": cert.governance_tier,
    }
    expected_genesis = _canonical_sha256("0" * 64, header_payload)
    expected_drift = _canonical_sha256(expected_genesis, cert.drift_summary)
    expected_conformal = _canonical_sha256(expected_drift, cert.conformal_summary)
    expected_fairness = _canonical_sha256(expected_conformal, cert.fairness_summary)
    expected_root = _canonical_sha256(
        expected_fairness,
        {"checklist": cert.checklist, "fairness_hash": expected_fairness},
    )

    return (
        cert.hash_chain.get("genesis_header_hash") == expected_genesis
        and cert.hash_chain.get("drift_block_hash") == expected_drift
        and cert.hash_chain.get("conformal_block_hash") == expected_conformal
        and cert.hash_chain.get("fairness_block_hash") == expected_fairness
        and cert.hash_chain.get("certificate_root_hash") == expected_root
        and cert.certificate_root_hash == expected_root
    )
