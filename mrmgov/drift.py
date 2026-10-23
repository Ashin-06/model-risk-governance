"""
Covariate vs Concept Drift Attribution Engine (SR 11-7 / RBI MRM Section 4).

Implements:
1. Per-feature Population Stability Index (PSI), Two-Sample Kolmogorov-Smirnov (KS)
   statistic & p-value, and 1D Wasserstein-1 (Earth Mover's) distance.
2. Importance-Weighted Brier Score & Residual Shift Decomposition:
   Separates pure input distribution shift P_prod(X) != P_ref(X) (Covariate Drift)
   from conditional target relationship shift P_prod(Y|X) != P_ref(Y|X) (Concept Drift).
3. Perturbation Shapley Feature Attribution for both Covariate and Concept Drift.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class FeatureDriftMetric:
    """Per-feature drift diagnostics across covariate and concept dimensions."""

    feature_name: str
    psi: float
    ks_stat: float
    ks_pvalue: float
    wasserstein_1d: float
    wasserstein_norm: float
    covariate_shapley_share: float
    concept_slope_shift: float
    concept_shapley_share: float
    status: str  # "STABLE" | "MODERATE" | "CRITICAL"
    is_covariate_drifted: bool
    is_concept_drifted: bool
    is_drifted: bool
    drift_mode: str  # "NONE" | "COVARIATE" | "CONCEPT" | "BOTH"

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class DriftDecompositionResult:
    """Complete Covariate vs Concept Drift decomposition report."""

    regime: str  # "NO_DRIFT" | "COVARIATE_DRIFT" | "CONCEPT_DRIFT" | "COMPOUND_DRIFT"
    mean_psi: float
    max_psi: float
    mean_ks: float
    domain_classifier_auc: float
    brier_ref: float
    brier_prod: float
    brier_reweighted_ref: float
    delta_brier_total: float
    delta_brier_covariate: float
    delta_brier_concept: float
    prediction_mean_shift: float
    residual_calibration_shift: float
    covariate_drift_score: float
    concept_drift_score: float
    covariate_drifted_features: List[str]
    concept_drifted_features: List[str]
    drifted_features: List[str]
    feature_metrics: List[FeatureDriftMetric]
    recommended_action: str

    def to_dict(self) -> Dict[str, object]:
        d = asdict(self)
        d["feature_metrics"] = [m.to_dict() for m in self.feature_metrics]
        return d


def compute_psi(
    ref: np.ndarray,
    prod: np.ndarray,
    n_bins: int = 10,
    eps: float = 1e-4,
) -> float:
    """
    Compute Population Stability Index (PSI) between reference and production 1D arrays.

    Uses reference decile boundaries with Laplace floor `eps` for zero-bin stability:
        PSI = sum_b (p_prod,b - p_ref,b) * ln(p_prod,b / p_ref,b)
    """
    ref_arr = np.asarray(ref, dtype=float).ravel()
    prod_arr = np.asarray(prod, dtype=float).ravel()
    if ref_arr.size == 0 or prod_arr.size == 0:
        raise ValueError("Reference and production arrays must be non-empty.")

    if np.allclose(ref_arr, prod_arr):
        return 0.0

    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(ref_arr, quantiles)
    edges = np.unique(edges)
    if edges.size < 2:
        # Constant reference feature
        return 0.0 if np.allclose(prod_arr, ref_arr[0]) else float(np.log(1.0 / eps))

    edges[0] = -np.inf
    edges[-1] = np.inf

    ref_counts, _ = np.histogram(ref_arr, bins=edges)
    prod_counts, _ = np.histogram(prod_arr, bins=edges)

    p_ref = np.maximum(ref_counts / float(ref_arr.size), eps)
    p_prod = np.maximum(prod_counts / float(prod_arr.size), eps)
    p_ref /= np.sum(p_ref)
    p_prod /= np.sum(p_prod)

    psi_val = float(np.sum((p_prod - p_ref) * np.log(p_prod / p_ref)))
    return max(0.0, psi_val)


def compute_ks_stat(ref: np.ndarray, prod: np.ndarray) -> Tuple[float, float]:
    """
    Compute exact two-sample Kolmogorov-Smirnov statistic D_KS and asymptotic p-value.
    """
    ref_arr = np.sort(np.asarray(ref, dtype=float).ravel())
    prod_arr = np.sort(np.asarray(prod, dtype=float).ravel())
    n1 = ref_arr.size
    n2 = prod_arr.size
    if n1 == 0 or n2 == 0:
        raise ValueError("Arrays must be non-empty.")

    combined = np.concatenate([ref_arr, prod_arr])
    cdf_ref = np.searchsorted(ref_arr, combined, side="right") / float(n1)
    cdf_prod = np.searchsorted(prod_arr, combined, side="right") / float(n2)
    d_ks = float(np.max(np.abs(cdf_ref - cdf_prod)))

    # Stephens (1970) effective sample size correction for two-sided Kolmogorov distribution
    n_eff = (n1 * n2) / float(n1 + n2)
    sqrt_neff = np.sqrt(n_eff)
    lam = (sqrt_neff + 0.12 + 0.11 / sqrt_neff) * d_ks
    if lam <= 1e-8:
        return d_ks, 1.0

    ks_sum = 0.0
    for k in range(1, 100):
        term = ((-1) ** (k - 1)) * np.exp(-2.0 * (k**2) * (lam**2))
        ks_sum += term
        if abs(term) < 1e-12:
            break
    p_val = float(np.clip(2.0 * ks_sum, 0.0, 1.0))
    return d_ks, p_val


def compute_wasserstein_1d(ref: np.ndarray, prod: np.ndarray) -> Tuple[float, float]:
    """
    Compute exact 1D Wasserstein-1 (Earth Mover's) distance and scale-normalized W_1.

    Returns:
        (w1_raw, w1_normalized_by_ref_std)
    """
    ref_arr = np.sort(np.asarray(ref, dtype=float).ravel())
    prod_arr = np.sort(np.asarray(prod, dtype=float).ravel())
    n1, n2 = ref_arr.size, prod_arr.size
    if n1 == 0 or n2 == 0:
        raise ValueError("Arrays must be non-empty.")

    all_vals = np.concatenate([ref_arr, prod_arr])
    all_vals.sort(kind="mergesort")
    deltas = np.diff(all_vals)
    if deltas.size == 0:
        return 0.0, 0.0

    cdf_ref = np.searchsorted(ref_arr, all_vals[:-1], side="right") / float(n1)
    cdf_prod = np.searchsorted(prod_arr, all_vals[:-1], side="right") / float(n2)
    w1 = float(np.sum(np.abs(cdf_ref - cdf_prod) * deltas))
    std_ref = float(np.std(ref_arr))
    w1_norm = w1 / max(std_ref, 1e-8)
    return w1, w1_norm


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35.0, 35.0)))


def _fit_logistic_Newton(
    X: np.ndarray, y: np.ndarray, l2_reg: float = 1.0, max_iter: int = 25
) -> Tuple[np.ndarray, float]:
    """Fit L2-regularized logistic regression via IRLS / Newton-Raphson."""
    n, d = X.shape
    X_aug = np.column_stack([np.ones(n), X])
    w = np.zeros(d + 1, dtype=float)
    reg_mat = np.eye(d + 1) * l2_reg
    reg_mat[0, 0] = 1e-6  # Do not regularize intercept

    for _ in range(max_iter):
        p = _sigmoid(X_aug @ w)
        grad = X_aug.T @ (p - y) + reg_mat @ w
        weights = np.maximum(p * (1.0 - p), 1e-6)
        hess = (X_aug.T * weights) @ X_aug + reg_mat
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hess, grad, rcond=None)[0]
        w -= step
        if float(np.max(np.abs(step))) < 1e-6:
            break

    return w[1:], float(w[0])


def _compute_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Exact Mann-Whitney U / Wilcoxon rank-sum ROC-AUC calculation."""
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_score = np.asarray(y_score, dtype=float).ravel()
    pos = y_true == 1
    neg = y_true == 0
    n_pos = int(np.sum(pos))
    n_neg = int(np.sum(neg))
    if n_pos == 0 or n_neg == 0:
        return 0.5

    order = np.argsort(y_score, kind="mergesort")
    sorted_scores = y_score[order]
    ranks = np.empty_like(sorted_scores, dtype=float)

    # Average ranks for ties
    n = sorted_scores.size
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1

    sum_pos_ranks = float(np.sum(ranks[pos]))
    auc = (sum_pos_ranks - n_pos * (n_pos + 1.0) / 2.0) / float(n_pos * n_neg)
    return float(np.clip(auc, 0.0, 1.0))


def analyze_drift(
    X_ref: np.ndarray,
    y_ref: np.ndarray,
    X_prod: np.ndarray,
    y_prod: np.ndarray,
    predict_proba_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    feature_names: Optional[List[str]] = None,
    psi_threshold: float = 0.15,
    concept_slope_threshold: float = 0.06,
) -> DriftDecompositionResult:
    """
    Perform full Covariate vs Concept Drift Attribution and Decomposition.

    Separates:
    - Covariate shift P_prod(X) != P_ref(X) via PSI, KS, Wasserstein-1, domain
      discriminator AUC, importance-weighted Brier shift, and interventional Shapley.
    - Concept shift P_prod(Y|X) != P_ref(Y|X) via importance-weighted residual Brier
      decomposition and per-feature partial residual slope shift.
    """
    X_ref = np.asarray(X_ref, dtype=float)
    X_prod = np.asarray(X_prod, dtype=float)
    y_ref = np.asarray(y_ref, dtype=float).ravel()
    y_prod = np.asarray(y_prod, dtype=float).ravel()

    if X_ref.ndim == 1:
        X_ref = X_ref[:, None]
    if X_prod.ndim == 1:
        X_prod = X_prod[:, None]

    n_ref, n_features = X_ref.shape
    n_prod = X_prod.shape[0]
    if feature_names is None:
        feature_names = [f"feature_{j}" for j in range(n_features)]

    # Standardize using reference statistics
    mu_ref = np.mean(X_ref, axis=0)
    sd_ref = np.maximum(np.std(X_ref, axis=0), 1e-8)
    Z_ref = (X_ref - mu_ref) / sd_ref
    Z_prod = (X_prod - mu_ref) / sd_ref

    # Fit reference model if predict_proba_fn not supplied
    if predict_proba_fn is None:
        coef_ref, intercept_ref = _fit_logistic_Newton(Z_ref, y_ref, l2_reg=1.0)

        def _default_model(X_in: np.ndarray) -> np.ndarray:
            Z_in = (np.asarray(X_in, dtype=float) - mu_ref) / sd_ref
            return _sigmoid(Z_in @ coef_ref + intercept_ref)

        predict_proba_fn = _default_model
    else:
        coef_ref, _ = _fit_logistic_Newton(Z_ref, y_ref, l2_reg=1.0)

    p_ref = np.asarray(predict_proba_fn(X_ref), dtype=float).ravel()
    p_prod = np.asarray(predict_proba_fn(X_prod), dtype=float).ravel()

    # 1. Domain classifier for density ratio w(x) = dP_prod(x) / dP_ref(x)
    Z_dom = np.vstack([Z_ref, Z_prod])
    y_dom = np.concatenate([np.zeros(n_ref), np.ones(n_prod)])
    w_dom, b_dom = _fit_logistic_Newton(Z_dom, y_dom, l2_reg=2.0)
    q_dom = _sigmoid(Z_dom @ w_dom + b_dom)
    domain_auc = _compute_roc_auc(y_dom, q_dom)
    # Symmetrize AUC around 0.5 so random noise never looks<0.5
    domain_auc = max(domain_auc, 1.0 - domain_auc)

    q_ref = np.clip(q_dom[:n_ref], 0.02, 0.98)
    prior_ratio = float(n_ref) / float(n_prod)
    imp_weights = np.clip((q_ref / (1.0 - q_ref)) * prior_ratio, 0.1, 15.0)
    imp_weights /= np.mean(imp_weights)

    # 2. Additive Importance-Weighted Brier Score Decomposition
    brier_ref = float(np.mean((y_ref - p_ref) ** 2))
    brier_prod = float(np.mean((y_prod - p_prod) ** 2))
    brier_reweighted_ref = float(np.mean(imp_weights * ((y_ref - p_ref) ** 2)))

    delta_brier_total = brier_prod - brier_ref
    delta_brier_covariate = brier_reweighted_ref - brier_ref
    delta_brier_concept = brier_prod - brier_reweighted_ref

    # Prediction shift vs Residual calibration shift
    prediction_mean_shift = float(abs(np.mean(p_prod) - np.mean(p_ref)))
    res_ref = y_ref - p_ref
    res_prod = y_prod - p_prod
    residual_calibration_shift = float(abs(np.mean(res_prod) - np.mean(res_ref)))

    # 3. Per-feature PSI, KS, Wasserstein, Perturbation Shapley, and Concept Slope Shift
    psi_list: List[float] = []
    ks_list: List[Tuple[float, float]] = []
    w1_list: List[Tuple[float, float]] = []
    cov_raw_contrib = np.zeros(n_features, dtype=float)
    concept_slope_shifts = np.zeros(n_features, dtype=float)

    for j in range(n_features):
        x_r = X_ref[:, j]
        x_p = X_prod[:, j]

        psi_j = compute_psi(x_r, x_p)
        ks_j, ksp_j = compute_ks_stat(x_r, x_p)
        w1_j, w1_norm_j = compute_wasserstein_1d(x_r, x_p)

        psi_list.append(psi_j)
        ks_list.append((ks_j, ksp_j))
        w1_list.append((w1_j, w1_norm_j))

        # Interventional quantile transport of feature j from prod onto ref
        rank_idx = np.argsort(np.argsort(x_r))
        matched_prod_vals = np.quantile(x_p, (rank_idx + 0.5) / float(n_ref))
        X_perturbed = X_ref.copy()
        X_perturbed[:, j] = matched_prod_vals
        p_perturbed = np.asarray(predict_proba_fn(X_perturbed), dtype=float).ravel()

        pred_shift_j = float(np.mean(np.abs(p_perturbed - p_ref)))
        cov_raw_contrib[j] = pred_shift_j + 0.25 * psi_j + 0.15 * w1_norm_j

        # Partial relationship shift between standardized feature j and model residual (y - p_hat)
        z_r_j = (x_r - np.mean(x_r)) / max(float(np.std(x_r)), 1e-8)
        z_p_j = (x_p - np.mean(x_p)) / max(float(np.std(x_p)), 1e-8)
        slope_ref_j = float(np.mean(z_r_j * (res_ref - np.mean(res_ref))))
        slope_prod_j = float(np.mean(z_p_j * (res_prod - np.mean(res_prod))))
        concept_slope_shifts[j] = abs(slope_prod_j - slope_ref_j)

    cov_total = float(np.sum(cov_raw_contrib))
    cov_shapley = (
        cov_raw_contrib / cov_total
        if cov_total > 1e-12
        else np.full(n_features, 1.0 / n_features)
    )

    concept_total = float(np.sum(concept_slope_shifts))
    concept_shapley = (
        concept_slope_shifts / concept_total
        if concept_total > 1e-12
        else np.full(n_features, 1.0 / n_features)
    )

    feature_metrics: List[FeatureDriftMetric] = []
    covariate_drifted_features: List[str] = []
    concept_drifted_features: List[str] = []
    drifted_features: List[str] = []

    for j in range(n_features):
        fname = feature_names[j]
        psi_j = psi_list[j]
        ks_j, ksp_j = ks_list[j]
        w1_j, w1_norm_j = w1_list[j]
        c_slope_j = float(concept_slope_shifts[j])

        is_cov = bool(psi_j >= psi_threshold or (ks_j >= 0.14 and w1_norm_j >= 0.25))
        is_conc = bool(c_slope_j >= concept_slope_threshold)
        is_drifted = is_cov or is_conc

        if is_cov:
            covariate_drifted_features.append(fname)
        if is_conc:
            concept_drifted_features.append(fname)
        if is_drifted:
            drifted_features.append(fname)

        if is_cov and is_conc:
            drift_mode = "BOTH"
        elif is_cov:
            drift_mode = "COVARIATE"
        elif is_conc:
            drift_mode = "CONCEPT"
        else:
            drift_mode = "NONE"

        if psi_j >= 0.25 or c_slope_j >= 2.0 * concept_slope_threshold:
            status = "CRITICAL"
        elif is_drifted or psi_j >= 0.10:
            status = "MODERATE"
        else:
            status = "STABLE"

        feature_metrics.append(
            FeatureDriftMetric(
                feature_name=fname,
                psi=round(psi_j, 6),
                ks_stat=round(ks_j, 6),
                ks_pvalue=round(ksp_j, 6),
                wasserstein_1d=round(w1_j, 6),
                wasserstein_norm=round(w1_norm_j, 6),
                covariate_shapley_share=round(float(cov_shapley[j]), 6),
                concept_slope_shift=round(c_slope_j, 6),
                concept_shapley_share=round(float(concept_shapley[j]), 6),
                status=status,
                is_covariate_drifted=is_cov,
                is_concept_drifted=is_conc,
                is_drifted=is_drifted,
                drift_mode=drift_mode,
            )
        )

    mean_psi = float(np.mean(psi_list))
    max_psi = float(np.max(psi_list))
    mean_ks = float(np.mean([k[0] for k in ks_list]))

    covariate_drift_score = float(
        0.5 * max_psi + 0.5 * max(0.0, (domain_auc - 0.5) * 2.0)
    )
    concept_drift_score = float(
        0.5 * residual_calibration_shift
        + 0.5 * float(np.max(concept_slope_shifts))
        + max(0.0, delta_brier_concept)
    )

    has_cov = len(covariate_drifted_features) > 0 or max_psi >= psi_threshold
    has_conc = (
        len(concept_drifted_features) > 0
        or residual_calibration_shift >= 0.05
        or delta_brier_concept >= 0.025
    )

    if has_cov and has_conc:
        regime = "COMPOUND_DRIFT"
        recommended_action = (
            "FULL_RETRAINING_REQUIRED: Both P(X) and P(Y|X) have shifted. "
            "Quarantine automated decisions via Mondrian Conformal Abstention and re-estimate parameters."
        )
    elif has_cov:
        regime = "COVARIATE_DRIFT"
        recommended_action = (
            "IMPORTANCE_REWEIGHT_AND_RECALIBRATE: Input population P(X) shifted while "
            "conditional credit relationship P(Y|X) remains intact. Apply density-ratio "
            "reweighting and recalibrate Mondrian conformal quantiles."
        )
    elif has_conc:
        regime = "CONCEPT_DRIFT"
        recommended_action = (
            "MODEL_REDEVELOPMENT_SR11_7: Structural relationship P(Y|X) broke down despite "
            "stable marginal feature distributions P(X). Trigger Tier-1 Model Risk escalation."
        )
    else:
        regime = "NO_DRIFT"
        recommended_action = (
            "CONTINUE_PRODUCTION_MONITORING: Feature distributions and conditional "
            "calibration are within SR 11-7 / RBI tolerances."
        )

    return DriftDecompositionResult(
        regime=regime,
        mean_psi=round(mean_psi, 6),
        max_psi=round(max_psi, 6),
        mean_ks=round(mean_ks, 6),
        domain_classifier_auc=round(domain_auc, 6),
        brier_ref=round(brier_ref, 6),
        brier_prod=round(brier_prod, 6),
        brier_reweighted_ref=round(brier_reweighted_ref, 6),
        delta_brier_total=round(delta_brier_total, 6),
        delta_brier_covariate=round(delta_brier_covariate, 6),
        delta_brier_concept=round(delta_brier_concept, 6),
        prediction_mean_shift=round(prediction_mean_shift, 6),
        residual_calibration_shift=round(residual_calibration_shift, 6),
        covariate_drift_score=round(covariate_drift_score, 6),
        concept_drift_score=round(concept_drift_score, 6),
        covariate_drifted_features=covariate_drifted_features,
        concept_drifted_features=concept_drifted_features,
        drifted_features=drifted_features,
        feature_metrics=feature_metrics,
        recommended_action=recommended_action,
    )
