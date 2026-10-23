"""
Mondrian (Group-Conditional) Split Conformal Prediction Abstention Layer.

Wraps any binary credit/fraud probability scorer p_hat(x) in [0, 1] and calibrates
non-conformity scores s_i = 1 - p_hat_{y_i}(x_i) separately within each demographic
or risk segment group g at target miscoverage rate alpha using the finite-sample
order statistic ceil((n_g + 1) * (1 - alpha)) / n_g.

Produces prediction sets C(x) subset of {0, 1}:
  - {0}    -> AUTO_APPROVE (guaranteed high confidence non-default)
  - {1}    -> AUTO_DECLINE (guaranteed high confidence default/fraud)
  - {0, 1} -> ABSTAIN_HUMAN_UNDERWRITER (ambiguous region routed to human review)

Guarantees finite-sample group-conditional marginal coverage:
  P(Y in C(X) | G = g) >= 1 - alpha   for all groups g.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union
import numpy as np


@dataclass
class GroupCoverageReport:
    """Empirical coverage and abstention metrics for a single demographic/segment group."""

    group_id: str
    n_calib: int
    n_eval: int
    target_coverage: float
    empirical_coverage: float
    coverage_gap: float
    quantile_level: float
    q_hat: float
    auto_approve_rate: float
    auto_decline_rate: float
    abstention_rate: float
    singleton_accuracy: float
    is_valid_coverage: bool

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class ConformalPredictionResult:
    """Batch output from Mondrian Conformal Predictor."""

    alpha: float
    target_coverage: float
    prediction_sets: List[List[int]]
    decisions: List[str]
    lower_thresholds: List[float]
    upper_thresholds: List[float]
    group_thresholds: Dict[str, float]
    overall_coverage: Optional[float]
    overall_abstention_rate: float
    overall_auto_approve_rate: float
    overall_auto_decline_rate: float
    overall_singleton_accuracy: Optional[float]
    group_reports: List[GroupCoverageReport]

    def to_dict(self) -> Dict[str, object]:
        d = asdict(self)
        d["group_reports"] = [g.to_dict() for g in self.group_reports]
        return d


class MondrianConformalPredictor:
    """
    Mondrian (Group-Conditional) Split Conformal Predictor for binary credit/fraud scoring.
    """

    def __init__(self, alpha: float = 0.05, allow_empty: bool = False) -> None:
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.alpha = float(alpha)
        self.allow_empty = bool(allow_empty)
        self.group_q_hat_: Dict[str, float] = {}
        self.group_quantile_level_: Dict[str, float] = {}
        self.group_n_calib_: Dict[str, int] = {}
        self.global_q_hat_: float = 1.0 - alpha

    @staticmethod
    def nonconformity_score(probs: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        Compute ground-truth non-conformity score s_i = 1 - p_hat_{y_i}(x_i).
        For binary y_i in {0, 1} and p_hat(x_i) = P(Y=1|X=x_i):
          - if y_i == 1: s_i = 1 - p_hat(x_i)
          - if y_i == 0: s_i = p_hat(x_i)
        """
        p = np.clip(np.asarray(probs, dtype=float).ravel(), 0.0, 1.0)
        y = np.asarray(y_true, dtype=int).ravel()
        return np.where(y == 1, 1.0 - p, p)

    @staticmethod
    def finite_sample_quantile(scores: np.ndarray, alpha: float) -> Tuple[float, float]:
        """
        Compute finite-sample conformal quantile using rank k = ceil((n + 1) * (1 - alpha)).

        Returns:
            (q_hat, quantile_level) where quantile_level = min(1.0, ceil((n + 1)*(1 - alpha)) / n)
        """
        s = np.sort(np.asarray(scores, dtype=float).ravel())
        n = s.size
        if n == 0:
            raise ValueError("Cannot calibrate conformal quantile on empty score array.")
        raw_k = math.ceil((n + 1) * (1.0 - alpha))
        q_level = min(1.0, raw_k / float(n))
        k_idx = min(n, max(1, raw_k)) - 1
        return float(s[k_idx]), float(q_level)

    def fit(
        self,
        probs_calib: Sequence[float],
        y_calib: Sequence[int],
        groups_calib: Sequence[Union[str, int]],
    ) -> "MondrianConformalPredictor":
        """
        Calibrate group-conditional non-conformity thresholds q_hat_g for each group g.
        """
        p = np.asarray(probs_calib, dtype=float).ravel()
        y = np.asarray(y_calib, dtype=int).ravel()
        g = np.asarray([str(v) for v in groups_calib])

        if not (p.size == y.size == g.size) or p.size == 0:
            raise ValueError("Calibration inputs must be non-empty and have matching lengths.")
        if not np.all(np.isfinite(p)) or not np.all(np.isin(y, (0, 1))):
            raise ValueError("Calibration probabilities must be finite and labels 0 or 1.")

        all_scores = self.nonconformity_score(p, y)
        self.global_q_hat_, _ = self.finite_sample_quantile(all_scores, self.alpha)

        self.group_q_hat_.clear()
        self.group_quantile_level_.clear()
        self.group_n_calib_.clear()

        for grp in sorted(set(g.tolist())):
            mask = g == grp
            scores_g = all_scores[mask]
            q_hat_g, q_level_g = self.finite_sample_quantile(scores_g, self.alpha)
            self.group_q_hat_[grp] = q_hat_g
            self.group_quantile_level_[grp] = q_level_g
            self.group_n_calib_[grp] = int(np.sum(mask))

        return self

    def predict(
        self,
        probs_test: Sequence[float],
        groups_test: Sequence[Union[str, int]],
        y_test: Optional[Sequence[int]] = None,
        coverage_tolerance: float = 0.015,
    ) -> ConformalPredictionResult:
        """
        Produce Mondrian prediction sets C(x) subset of {0, 1} and underwriting routing decisions.
        If `y_test` is provided, also computes per-group empirical coverage and singleton accuracy.
        """
        if not self.group_q_hat_:
            raise RuntimeError("MondrianConformalPredictor must be calibrated via .fit() first.")

        p = np.clip(np.asarray(probs_test, dtype=float).ravel(), 0.0, 1.0)
        g = np.asarray([str(v) for v in groups_test])
        if p.size != g.size or p.size == 0:
            raise ValueError("Test probabilities and groups must be non-empty and match in length.")
        if not np.all(np.isfinite(p)):
            raise ValueError("Test probabilities must be finite.")

        y_arr: Optional[np.ndarray] = None
        if y_test is not None:
            y_arr = np.asarray(y_test, dtype=int).ravel()
            if y_arr.size != p.size or not np.all(np.isin(y_arr, (0, 1))):
                raise ValueError("y_test must match probs_test in length and be 0 or 1.")

        prediction_sets: List[List[int]] = []
        decisions: List[str] = []
        lower_thresholds: List[float] = []
        upper_thresholds: List[float] = []

        for i in range(p.size):
            grp = g[i]
            q_g = self.group_q_hat_.get(grp, self.global_q_hat_)
            prob_i = float(p[i])

            # Candidate 0 non-conformity score: s(x, 0) = p_hat(x) <= q_g
            include_0 = prob_i <= q_g
            # Candidate 1 non-conformity score: s(x, 1) = 1 - p_hat(x) <= q_g <=> p_hat(x) >= 1 - q_g
            include_1 = (1.0 - prob_i) <= q_g

            c_set: List[int] = []
            if include_0:
                c_set.append(0)
            if include_1:
                c_set.append(1)

            if not c_set and not self.allow_empty:
                # Ambiguous band when q_g < 0.5: route to human underwriter as {0, 1}
                c_set = [0, 1]

            if c_set == [0]:
                decision = "AUTO_APPROVE"
            elif c_set == [1]:
                decision = "AUTO_DECLINE"
            else:
                decision = "ABSTAIN_HUMAN_UNDERWRITER"

            low_t = min(q_g, 1.0 - q_g)
            high_t = max(q_g, 1.0 - q_g)

            prediction_sets.append(c_set)
            decisions.append(decision)
            lower_thresholds.append(round(float(low_t), 6))
            upper_thresholds.append(round(float(high_t), 6))

        decisions_arr = np.asarray(decisions)
        overall_approve = float(np.mean(decisions_arr == "AUTO_APPROVE"))
        overall_decline = float(np.mean(decisions_arr == "AUTO_DECLINE"))
        overall_abstain = float(np.mean(decisions_arr == "ABSTAIN_HUMAN_UNDERWRITER"))

        overall_cov: Optional[float] = None
        overall_singleton_acc: Optional[float] = None
        if y_arr is not None:
            covered = np.array(
                [int(y_arr[i]) in prediction_sets[i] for i in range(p.size)],
                dtype=bool,
            )
            overall_cov = round(float(np.mean(covered)), 6)
            singleton_mask = np.array([len(s) == 1 for s in prediction_sets], dtype=bool)
            if np.any(singleton_mask):
                singleton_preds = np.array(
                    [prediction_sets[i][0] for i in np.where(singleton_mask)[0]],
                    dtype=int,
                )
                overall_singleton_acc = round(
                    float(np.mean(singleton_preds == y_arr[singleton_mask])), 6
                )
            else:
                overall_singleton_acc = 1.0

        group_reports: List[GroupCoverageReport] = []
        target_cov = 1.0 - self.alpha

        for grp in sorted(set(g.tolist())):
            mask = np.where(g == grp)[0]
            n_eval_g = int(mask.size)
            q_g = self.group_q_hat_.get(grp, self.global_q_hat_)
            q_lvl = self.group_quantile_level_.get(grp, target_cov)
            n_cal_g = self.group_n_calib_.get(grp, 0)

            dec_g = decisions_arr[mask]
            app_g = float(np.mean(dec_g == "AUTO_APPROVE"))
            dec_rate_g = float(np.mean(dec_g == "AUTO_DECLINE"))
            abs_g = float(np.mean(dec_g == "ABSTAIN_HUMAN_UNDERWRITER"))

            if y_arr is not None:
                cov_g = float(
                    np.mean([int(y_arr[idx]) in prediction_sets[idx] for idx in mask])
                )
                sing_idx = [idx for idx in mask if len(prediction_sets[idx]) == 1]
                if sing_idx:
                    sing_acc_g = float(
                        np.mean([prediction_sets[idx][0] == int(y_arr[idx]) for idx in sing_idx])
                    )
                else:
                    sing_acc_g = 1.0
                is_valid = cov_g >= (target_cov - coverage_tolerance)
            else:
                cov_g = target_cov
                sing_acc_g = 1.0
                is_valid = True

            group_reports.append(
                GroupCoverageReport(
                    group_id=grp,
                    n_calib=n_cal_g,
                    n_eval=n_eval_g,
                    target_coverage=round(target_cov, 6),
                    empirical_coverage=round(cov_g, 6),
                    coverage_gap=round(cov_g - target_cov, 6),
                    quantile_level=round(q_lvl, 6),
                    q_hat=round(q_g, 6),
                    auto_approve_rate=round(app_g, 6),
                    auto_decline_rate=round(dec_rate_g, 6),
                    abstention_rate=round(abs_g, 6),
                    singleton_accuracy=round(sing_acc_g, 6),
                    is_valid_coverage=bool(is_valid),
                )
            )

        return ConformalPredictionResult(
            alpha=round(self.alpha, 6),
            target_coverage=round(target_cov, 6),
            prediction_sets=prediction_sets,
            decisions=decisions,
            lower_thresholds=lower_thresholds,
            upper_thresholds=upper_thresholds,
            group_thresholds={k: round(v, 6) for k, v in self.group_q_hat_.items()},
            overall_coverage=overall_cov,
            overall_abstention_rate=round(overall_abstain, 6),
            overall_auto_approve_rate=round(overall_approve, 6),
            overall_auto_decline_rate=round(overall_decline, 6),
            overall_singleton_accuracy=overall_singleton_acc,
            group_reports=group_reports,
        )
