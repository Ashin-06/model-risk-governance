"""
ECOA / Fair-Lending Disparate Impact Auditor & Constrained Threshold Repair Engine.

Implements:
1. Disparate Impact Ratio (DIR = P(Y_hat=1 | G=unpriv) / P(Y_hat=1 | G=priv)),
   Statistical Parity Difference (SPD), Equal Opportunity Difference (EOD),
   and Average Odds Difference (AOD).
2. Vectorized Constrained Group-Threshold Optimization:
   Finds optimal group thresholds (t_priv*, t_unpriv*) that maximize expected
   bank lending profit / F1 subject to:
     - ECOA Four-Fifths Rule: DIR >= min_dir (default 0.80, target >= 0.82)
     - Equal Opportunity Parity: |EOD| <= max_abs_eod (default 0.03)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Optional, Sequence, Tuple, Union
import numpy as np


@dataclass
class FairnessMetrics:
    """Comprehensive ECOA / Fair-Lending audit metrics."""

    threshold_privileged: float
    threshold_unprivileged: float
    approval_rate_privileged: float
    approval_rate_unprivileged: float
    disparate_impact_ratio: float
    statistical_parity_diff: float
    tpr_privileged: float
    tpr_unprivileged: float
    fpr_privileged: float
    fpr_unprivileged: float
    equal_opportunity_diff: float
    average_odds_diff: float
    f1_score: float
    accuracy: float
    expected_profit: float
    roc_auc: float
    four_fifths_compliant: bool
    eod_compliant: bool
    overall_compliant: bool

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class FairnessRepairResult:
    """Before/after comparison of Constrained Group-Threshold Fair-Lending Repair."""

    baseline: FairnessMetrics
    repaired: FairnessMetrics
    optimal_thresholds: Dict[str, float]
    dir_improvement: float
    eod_reduction: float
    profit_retention_ratio: float
    f1_retention_ratio: float
    auc_retention_ratio: float
    repaired_predictions: list[int]
    repaired_scores: list[float]
    status: str

    def to_dict(self) -> Dict[str, object]:
        d = asdict(self)
        d["baseline"] = self.baseline.to_dict()
        d["repaired"] = self.repaired.to_dict()
        return d


def _roc_auc_binary(y_true: np.ndarray, y_score: np.ndarray) -> float:
    pos = y_true == 1
    neg = y_true == 0
    n_pos = int(np.sum(pos))
    n_neg = int(np.sum(neg))
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(y_score, kind="mergesort")
    sorted_s = y_score[order]
    ranks = np.empty_like(sorted_s, dtype=float)
    n = sorted_s.size
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    auc = (float(np.sum(ranks[pos])) - n_pos * (n_pos + 1.0) / 2.0) / float(
        n_pos * n_neg
    )
    return float(np.clip(auc, 0.0, 1.0))


def compute_fairness_metrics(
    scores: Sequence[float],
    y_true: Sequence[int],
    groups: Sequence[Union[int, str]],
    privileged_group: Union[int, str] = 1,
    unprivileged_group: Union[int, str] = 0,
    threshold_privileged: float = 0.50,
    threshold_unprivileged: float = 0.50,
    tp_profit: float = 1.0,
    fp_cost: float = 1.6,
    min_dir: float = 0.80,
    max_abs_eod: float = 0.03,
) -> FairnessMetrics:
    """
    Compute ECOA Disparate Impact Ratio, Equal Opportunity Difference, Average Odds
    Difference, F1, ROC-AUC, and Expected Bank Profit.

    Convention:
      - `scores` in [0, 1]: probability of creditworthiness / loan repayment (Y = 1).
      - Approval decision Y_hat = 1 iff score >= group_threshold.
    """
    s = np.asarray(scores, dtype=float).ravel()
    y = np.asarray(y_true, dtype=int).ravel()
    g = np.asarray([str(v) for v in groups])
    priv_str = str(privileged_group)
    unpriv_str = str(unprivileged_group)
    if not (s.size == y.size == g.size):
        raise ValueError("scores, y_true and groups must have the same length.")
    if not np.all(np.isfinite(s)) or not np.all(np.isin(y, (0, 1))):
        raise ValueError("scores must be finite and y_true must be 0 or 1.")

    priv_mask = g == priv_str
    unpriv_mask = g == unpriv_str
    if not np.any(priv_mask) or not np.any(unpriv_mask):
        raise ValueError(
            f"Both privileged ({priv_str}) and unprivileged ({unpriv_str}) groups must be present."
        )

    y_hat = np.zeros_like(y, dtype=int)
    y_hat[priv_mask] = (s[priv_mask] >= threshold_privileged).astype(int)
    y_hat[unpriv_mask] = (s[unpriv_mask] >= threshold_unprivileged).astype(int)

    app_priv = float(np.mean(y_hat[priv_mask]))
    app_unpriv = float(np.mean(y_hat[unpriv_mask]))

    dir_val = app_unpriv / max(app_priv, 1e-8) if app_priv > 0 else 1.0
    spd_val = app_unpriv - app_priv

    # Group-conditional TPR and FPR
    priv_pos = priv_mask & (y == 1)
    priv_neg = priv_mask & (y == 0)
    unpriv_pos = unpriv_mask & (y == 1)
    unpriv_neg = unpriv_mask & (y == 0)

    tpr_priv = float(np.mean(y_hat[priv_pos])) if np.any(priv_pos) else 0.0
    tpr_unpriv = float(np.mean(y_hat[unpriv_pos])) if np.any(unpriv_pos) else 0.0
    fpr_priv = float(np.mean(y_hat[priv_neg])) if np.any(priv_neg) else 0.0
    fpr_unpriv = float(np.mean(y_hat[unpriv_neg])) if np.any(unpriv_neg) else 0.0

    eod_val = tpr_unpriv - tpr_priv
    aod_val = 0.5 * ((tpr_unpriv - tpr_priv) + (fpr_unpriv - fpr_priv))

    tp = int(np.sum((y_hat == 1) & (y == 1)))
    fp = int(np.sum((y_hat == 1) & (y == 0)))
    fn = int(np.sum((y_hat == 0) & (y == 1)))
    denom_f1 = 2 * tp + fp + fn
    f1_val = (2.0 * tp / float(denom_f1)) if denom_f1 > 0 else 0.0
    acc_val = float(np.mean(y_hat == y))
    profit_val = float(tp * tp_profit - fp * fp_cost) / float(y.size)

    # Effective threshold-aligned score for ROC-AUC evaluation
    s_aligned = s.copy()
    s_aligned[priv_mask] = s[priv_mask] - (threshold_privileged - 0.5)
    s_aligned[unpriv_mask] = s[unpriv_mask] - (threshold_unprivileged - 0.5)
    auc_val = _roc_auc_binary(y, s_aligned)

    four_fifths_ok = bool(dir_val >= min_dir - 1e-9)
    eod_ok = bool(abs(eod_val) <= max_abs_eod + 1e-9)

    return FairnessMetrics(
        threshold_privileged=round(float(threshold_privileged), 6),
        threshold_unprivileged=round(float(threshold_unprivileged), 6),
        approval_rate_privileged=round(app_priv, 6),
        approval_rate_unprivileged=round(app_unpriv, 6),
        disparate_impact_ratio=round(dir_val, 6),
        statistical_parity_diff=round(spd_val, 6),
        tpr_privileged=round(tpr_priv, 6),
        tpr_unprivileged=round(tpr_unpriv, 6),
        fpr_privileged=round(fpr_priv, 6),
        fpr_unprivileged=round(fpr_unpriv, 6),
        equal_opportunity_diff=round(eod_val, 6),
        average_odds_diff=round(aod_val, 6),
        f1_score=round(f1_val, 6),
        accuracy=round(acc_val, 6),
        expected_profit=round(profit_val, 6),
        roc_auc=round(auc_val, 6),
        four_fifths_compliant=four_fifths_ok,
        eod_compliant=eod_ok,
        overall_compliant=bool(four_fifths_ok and eod_ok),
    )


def repair_group_thresholds(
    scores: Sequence[float],
    y_true: Sequence[int],
    groups: Sequence[Union[int, str]],
    privileged_group: Union[int, str] = 1,
    unprivileged_group: Union[int, str] = 0,
    baseline_threshold: float = 0.50,
    min_dir: float = 0.80,
    target_dir_floor: float = 0.825,
    max_abs_eod: float = 0.03,
    tp_profit: float = 1.0,
    fp_cost: float = 1.6,
    grid_size: int = 240,
) -> FairnessRepairResult:
    """
    Solve constrained group-threshold optimization to find (t_priv*, t_unpriv*)
    maximizing expected bank profit + F1 utility subject to:
      1. ECOA Four-Fifths Rule: DIR >= max(min_dir, target_dir_floor) and DIR <= 1.25
      2. Equal Opportunity Parity: |EOD| <= max_abs_eod
    """
    s = np.asarray(scores, dtype=float).ravel()
    y = np.asarray(y_true, dtype=int).ravel()
    g = np.asarray([str(v) for v in groups])
    priv_str = str(privileged_group)
    unpriv_str = str(unprivileged_group)

    baseline = compute_fairness_metrics(
        scores=s,
        y_true=y,
        groups=g,
        privileged_group=priv_str,
        unprivileged_group=unpriv_str,
        threshold_privileged=baseline_threshold,
        threshold_unprivileged=baseline_threshold,
        tp_profit=tp_profit,
        fp_cost=fp_cost,
        min_dir=min_dir,
        max_abs_eod=max_abs_eod,
    )

    priv_mask = g == priv_str
    unpriv_mask = g == unpriv_str

    s_priv = s[priv_mask]
    y_priv = y[priv_mask]
    s_unpriv = s[unpriv_mask]
    y_unpriv = y[unpriv_mask]

    n_total = float(s.size)
    n_priv = float(s_priv.size)
    n_unpriv = float(s_unpriv.size)
    n_priv_pos = max(float(np.sum(y_priv == 1)), 1.0)
    n_unpriv_pos = max(float(np.sum(y_unpriv == 1)), 1.0)

    # Candidate threshold grids covering [0.15, 0.85] + group score quantiles
    q_grid = np.linspace(0.05, 0.95, grid_size // 2)
    u_grid = np.linspace(0.15, 0.85, grid_size // 2)
    t_priv_candidates = np.unique(
        np.clip(np.concatenate([[baseline_threshold], np.quantile(s_priv, q_grid), u_grid]), 0.05, 0.95)
    )
    t_unpriv_candidates = np.unique(
        np.clip(np.concatenate([[baseline_threshold], np.quantile(s_unpriv, q_grid), u_grid]), 0.05, 0.95)
    )

    # Vectorized 1D statistics per group across its candidate thresholds
    # Shape: (M_priv, N_priv)
    pred_priv = s_priv[None, :] >= t_priv_candidates[:, None]
    app_priv_vec = np.mean(pred_priv, axis=1)
    tp_priv_vec = np.sum(pred_priv & (y_priv[None, :] == 1), axis=1).astype(float)
    fp_priv_vec = np.sum(pred_priv & (y_priv[None, :] == 0), axis=1).astype(float)
    fn_priv_vec = np.sum((~pred_priv) & (y_priv[None, :] == 1), axis=1).astype(float)
    tpr_priv_vec = tp_priv_vec / n_priv_pos

    # Shape: (M_unpriv, N_unpriv)
    pred_unpriv = s_unpriv[None, :] >= t_unpriv_candidates[:, None]
    app_unpriv_vec = np.mean(pred_unpriv, axis=1)
    tp_unpriv_vec = np.sum(pred_unpriv & (y_unpriv[None, :] == 1), axis=1).astype(float)
    fp_unpriv_vec = np.sum(pred_unpriv & (y_unpriv[None, :] == 0), axis=1).astype(float)
    fn_unpriv_vec = np.sum((~pred_unpriv) & (y_unpriv[None, :] == 1), axis=1).astype(float)
    tpr_unpriv_vec = tp_unpriv_vec / n_unpriv_pos

    # 2D broadcast across (i_priv, j_unpriv)
    dir_mat = app_unpriv_vec[None, :] / np.maximum(app_priv_vec[:, None], 1e-8)
    eod_mat = tpr_unpriv_vec[None, :] - tpr_priv_vec[:, None]

    tp_mat = tp_priv_vec[:, None] + tp_unpriv_vec[None, :]
    fp_mat = fp_priv_vec[:, None] + fp_unpriv_vec[None, :]
    fn_mat = fn_priv_vec[:, None] + fn_unpriv_vec[None, :]

    profit_mat = (tp_mat * tp_profit - fp_mat * fp_cost) / n_total
    f1_mat = (2.0 * tp_mat) / np.maximum(2.0 * tp_mat + fp_mat + fn_mat, 1.0)

    # Composite objective: Expected Bank Profit + 0.35 * F1 - small threshold-Regularization
    reg_mat = 0.02 * (
        (t_priv_candidates[:, None] - baseline_threshold) ** 2
        + (t_unpriv_candidates[None, :] - baseline_threshold) ** 2
    )
    obj_mat = profit_mat + 0.35 * f1_mat - reg_mat

    effective_dir_floor = max(min_dir, target_dir_floor)
    feasible_strict = (
        (dir_mat >= effective_dir_floor)
        & (dir_mat <= 1.25)
        & (np.abs(eod_mat) <= max_abs_eod)
        & (app_priv_vec[:, None] >= 0.10)
    )

    if np.any(feasible_strict):
        masked_obj = np.where(feasible_strict, obj_mat, -np.inf)
        best_i, best_j = np.unravel_index(int(np.argmax(masked_obj)), masked_obj.shape)
    else:
        # Relax to exact min_dir if strict floor was too tight
        feasible_legal = (
            (dir_mat >= min_dir)
            & (dir_mat <= 1.30)
            & (np.abs(eod_mat) <= max_abs_eod)
            & (app_priv_vec[:, None] >= 0.08)
        )
        if np.any(feasible_legal):
            masked_obj = np.where(feasible_legal, obj_mat, -np.inf)
            best_i, best_j = np.unravel_index(int(np.argmax(masked_obj)), masked_obj.shape)
        else:
            # Penalty fallback minimizing constraint violation
            dir_Shortfall = np.maximum(0.0, effective_dir_floor - dir_mat)
            eod_Excess = np.maximum(0.0, np.abs(eod_mat) - max_abs_eod)
            penalized = obj_mat - 25.0 * (dir_Shortfall + eod_Excess)
            best_i, best_j = np.unravel_index(int(np.argmax(penalized)), penalized.shape)

    t_priv_opt = float(t_priv_candidates[best_i])
    t_unpriv_opt = float(t_unpriv_candidates[best_j])

    repaired = compute_fairness_metrics(
        scores=s,
        y_true=y,
        groups=g,
        privileged_group=priv_str,
        unprivileged_group=unpriv_str,
        threshold_privileged=t_priv_opt,
        threshold_unprivileged=t_unpriv_opt,
        tp_profit=tp_profit,
        fp_cost=fp_cost,
        min_dir=min_dir,
        max_abs_eod=max_abs_eod,
    )

    repaired_preds = np.zeros_like(y, dtype=int)
    repaired_preds[priv_mask] = (s[priv_mask] >= t_priv_opt).astype(int)
    repaired_preds[unpriv_mask] = (s[unpriv_mask] >= t_unpriv_opt).astype(int)

    repaired_scores_arr = s.copy()
    repaired_scores_arr[priv_mask] = np.clip(s[priv_mask] - (t_priv_opt - 0.50), 0.0, 1.0)
    repaired_scores_arr[unpriv_mask] = np.clip(
        s[unpriv_mask] - (t_unpriv_opt - 0.50), 0.0, 1.0
    )

    profit_ret = (
        repaired.expected_profit / baseline.expected_profit
        if baseline.expected_profit > 1e-8
        else 1.0
    )
    f1_ret = repaired.f1_score / baseline.f1_score if baseline.f1_score > 1e-8 else 1.0
    auc_ret = repaired.roc_auc / baseline.roc_auc if baseline.roc_auc > 1e-8 else 1.0

    status = (
        "ECOA_COMPLIANT_REPAIRED"
        if repaired.overall_compliant
        else "PARTIAL_REPAIR_REVIEW_REQUIRED"
    )

    return FairnessRepairResult(
        baseline=baseline,
        repaired=repaired,
        optimal_thresholds={
            priv_str: round(t_priv_opt, 6),
            unpriv_str: round(t_unpriv_opt, 6),
        },
        dir_improvement=round(
            repaired.disparate_impact_ratio - baseline.disparate_impact_ratio, 6
        ),
        eod_reduction=round(
            abs(baseline.equal_opportunity_diff) - abs(repaired.equal_opportunity_diff), 6
        ),
        profit_retention_ratio=round(float(profit_ret), 6),
        f1_retention_ratio=round(float(f1_ret), 6),
        auc_retention_ratio=round(float(auc_ret), 6),
        repaired_predictions=repaired_preds.tolist(),
        repaired_scores=np.round(repaired_scores_arr, 6).tolist(),
        status=status,
    )
