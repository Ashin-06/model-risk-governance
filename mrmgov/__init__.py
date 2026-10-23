"""
mrmgov: SR 11-7 / RBI Model Risk Management (MRM) Governance Engine.

Provides:
1. Covariate vs Concept Drift Decomposition & Per-Feature Attribution (PSI, KS, Wasserstein-1, heuristic shares).
2. Mondrian (Group-Conditional) Split Conformal Prediction Abstention Layer.
3. ECOA / Fair-Lending Disparate Impact Auditor & Constrained Threshold Repair (Four-Fifths Rule).
4. SR 11-7 / RBI Hash-Chained Model Governance Audit Certificate & Benchmark Suite.
"""

from .drift import (
    FeatureDriftMetric,
    DriftDecompositionResult,
    compute_psi,
    compute_ks_stat,
    compute_wasserstein_1d,
    analyze_drift,
)
from .conformal import (
    MondrianConformalPredictor,
    ConformalPredictionResult,
    GroupCoverageReport,
)
from .fairness import (
    FairnessMetrics,
    FairnessRepairResult,
    compute_fairness_metrics,
    repair_group_thresholds,
)
from .report import (
    GovernanceCertificate,
    generate_governance_certificate,
    verify_certificate_signature,
)
from .bench import run_benchmark_suite

__version__ = "1.0.0"

__all__ = [
    "FeatureDriftMetric",
    "DriftDecompositionResult",
    "compute_psi",
    "compute_ks_stat",
    "compute_wasserstein_1d",
    "analyze_drift",
    "MondrianConformalPredictor",
    "ConformalPredictionResult",
    "GroupCoverageReport",
    "FairnessMetrics",
    "FairnessRepairResult",
    "compute_fairness_metrics",
    "repair_group_thresholds",
    "GovernanceCertificate",
    "generate_governance_certificate",
    "verify_certificate_signature",
    "run_benchmark_suite",
]
