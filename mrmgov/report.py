"""
Signed record of the automated model-risk checks.

A certificate chains SHA-256 hashes over:
  - Block 0 (header): model ID, version, frameworks referenced, timestamp
  - Block 1 (drift): covariate vs concept drift summary
  - Block 2 (conformal): group coverage and abstention
  - Block 3 (fairness): disparate impact and equal opportunity summary
  - Root: checklist + disposition + the block hashes
and signs the root hash with Ed25519. Verification needs the signer's public key
from a trusted source; without the signature the hash chain alone proves nothing,
since anyone who edits a value can recompute every hash.

This is not a model approval. SR 11-7 expects independent validation and sign-off
by people with that authority; the disposition only summarises automated checks.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from typing import Dict, List, Optional

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from .drift import DriftDecompositionResult
from .conformal import ConformalPredictionResult
from .fairness import FairnessRepairResult


def _canonical_sha256(prev_hash: str, payload: Dict[str, object]) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    msg = f"{prev_hash}|{serialized}".encode("utf-8")
    return hashlib.sha256(msg).hexdigest()


@dataclass
class GovernanceCertificate:
    """Hash-chained, Ed25519-signed record of automated model-risk checks."""

    certificate_id: str
    model_id: str
    model_version: str
    frameworks_referenced: List[str]
    issued_at_utc: str
    governance_tier: str
    overall_disposition: str  # "AUTOMATED_CHECKS_PASSED" | "CHECKS_PASSED_WITH_CONCEPT_DRIFT" | "CHECKS_FAILED"
    checklist: Dict[str, bool]
    drift_summary: Dict[str, object]
    conformal_summary: Dict[str, object]
    fairness_summary: Dict[str, object]
    hash_chain: Dict[str, str]
    certificate_root_hash: str
    signature: str  # Ed25519 over certificate_root_hash, hex

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def generate_governance_certificate(
    model_id: str,
    model_version: str,
    drift_result: DriftDecompositionResult,
    conformal_result: ConformalPredictionResult,
    fairness_result: FairnessRepairResult,
    signing_key: Ed25519PrivateKey,
    issued_at_utc: Optional[str] = None,
    governance_tier: str = "TIER_1_MATERIAL_CREDIT_UNDERWRITING",
) -> GovernanceCertificate:
    """
    Generate a hash-chained certificate and sign its root hash with `signing_key`.
    """
    if issued_at_utc is None:
        issued_at_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Informational: the checks below are inputs to these processes, not evidence of compliance
    frameworks = [
        "SR 11-7 (Fed/OCC model risk guidance)",
        "RBI model risk expectations for credit models",
        "ECOA / Regulation B",
        "EU AI Act Annex III (credit scoring)",
    ]

    header_payload: Dict[str, object] = {
        "model_id": model_id,
        "model_version": model_version,
        "frameworks_referenced": frameworks,
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
        "drift_analysis_run": True,
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

    if (
        checklist["no_unmitigated_concept_drift"]
        and checklist["mondrian_conformal_coverage_verified"]
        and checklist["dir_meets_four_fifths"]
    ):
        disposition = "AUTOMATED_CHECKS_PASSED"
    elif (
        checklist["mondrian_conformal_coverage_verified"]
        and checklist["dir_meets_four_fifths"]
    ):
        disposition = "CHECKS_PASSED_WITH_CONCEPT_DRIFT"
    else:
        disposition = "CHECKS_FAILED"

    root_hash = _canonical_sha256(fairness_hash, _root_payload(checklist, disposition, fairness_hash))
    cert_id = f"MRM-CERT-{root_hash[:12].upper()}"

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
        frameworks_referenced=frameworks,
        issued_at_utc=issued_at_utc,
        governance_tier=governance_tier,
        overall_disposition=disposition,
        checklist=checklist,
        drift_summary=drift_summary,
        conformal_summary=conformal_summary,
        fairness_summary=fairness_summary,
        hash_chain=hash_chain,
        certificate_root_hash=root_hash,
        signature=signing_key.sign(root_hash.encode("ascii")).hex(),
    )


def _root_payload(checklist: Dict[str, bool], disposition: str, fairness_hash: str) -> Dict[str, object]:
    return {"checklist": checklist, "disposition": disposition, "fairness_hash": fairness_hash}


def verify_certificate_signature(cert: GovernanceCertificate, public_key: Ed25519PublicKey) -> bool:
    """
    Recompute the hash chain from the certificate contents, then check the Ed25519
    signature on the root hash against `public_key`, which must come from a trusted
    source (never from the certificate or whoever sent it).
    """
    header_payload: Dict[str, object] = {
        "model_id": cert.model_id,
        "model_version": cert.model_version,
        "frameworks_referenced": cert.frameworks_referenced,
        "issued_at_utc": cert.issued_at_utc,
        "governance_tier": cert.governance_tier,
    }
    expected_genesis = _canonical_sha256("0" * 64, header_payload)
    expected_drift = _canonical_sha256(expected_genesis, cert.drift_summary)
    expected_conformal = _canonical_sha256(expected_drift, cert.conformal_summary)
    expected_fairness = _canonical_sha256(expected_conformal, cert.fairness_summary)
    expected_root = _canonical_sha256(
        expected_fairness,
        _root_payload(cert.checklist, cert.overall_disposition, expected_fairness),
    )

    chain_ok = (
        cert.hash_chain.get("genesis_header_hash") == expected_genesis
        and cert.hash_chain.get("drift_block_hash") == expected_drift
        and cert.hash_chain.get("conformal_block_hash") == expected_conformal
        and cert.hash_chain.get("fairness_block_hash") == expected_fairness
        and cert.hash_chain.get("certificate_root_hash") == expected_root
        and cert.certificate_root_hash == expected_root
        and cert.certificate_id == f"MRM-CERT-{expected_root[:12].upper()}"
    )
    if not chain_ok:
        return False
    try:
        public_key.verify(bytes.fromhex(cert.signature), expected_root.encode("ascii"))
    except (InvalidSignature, ValueError):
        return False
    return True
