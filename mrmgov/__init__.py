"""
mrmgov: model risk checks for credit models (reference implementation, generated data only).

Provides:
1. Covariate vs Concept Drift Decomposition & Per-Feature Attribution (PSI, KS, Wasserstein-1, heuristic shares).
2. Mondrian (Group-Conditional) Split Conformal Prediction Abstention Layer.
3. Group fairness metrics (DIR, EOD, AOD) and a diagnostic per-group threshold search.
4. Hash-chained, Ed25519-signed record of the checks, and a benchmark suite.
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
