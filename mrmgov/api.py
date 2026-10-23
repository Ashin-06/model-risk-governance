"""
FastAPI REST Engine for SR 11-7 / RBI Model Risk Governance (Port 8019).

Endpoints:
  - GET  /                  : Interactive SR 11-7 MRM Web Console (HTML)
  - GET  /health            : Service status & regulatory framework metadata
  - POST /audit             : Full Covariate/Concept Drift + Conformal + Fairness + Certificate audit
  - POST /conformal/predict : Mondrian Group-Conditional Split Conformal Prediction & Abstention
  - POST /fairness/repair   : ECOA Disparate Impact Auditor & Constrained Threshold Repair
  - GET  /bench             : Full reproducible SR 11-7 / RBI benchmark suite
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .drift import analyze_drift
from .conformal import MondrianConformalPredictor
from .fairness import repair_group_thresholds
from .report import generate_governance_certificate, verify_certificate_signature
from .bench import run_benchmark_suite


app = FastAPI(
    title="SR 11-7 / RBI Model Risk Governance Engine (mrmgov)",
    description=(
        "Covariate vs Concept Drift Attribution, Mondrian Conformal Prediction "
        "Abstention & ECOA Four-Fifths Fair-Lending Threshold Repair."
    ),
    version="1.0.0",
)

CONSOLE_PATH = Path(__file__).resolve().parent.parent / "web" / "console.html"


class AuditRequest(BaseModel):
    model_id: str = Field(default="CREDIT-UNDERWRITER-V4.2")
    model_version: str = Field(default="2026.10-SR117")
    X_ref: List[List[float]]
    y_ref: List[int]
    X_prod: List[List[float]]
    y_prod: List[int]
    feature_names: Optional[List[str]] = None
    psi_threshold: float = Field(default=0.15, gt=0.0)
    concept_slope_threshold: float = Field(default=0.06, gt=0.0)
    alpha: float = Field(default=0.05, gt=0.0, lt=1.0)
    groups_prod: Optional[List[Union[str, int]]] = None
    fairness_scores: Optional[List[float]] = None
    fairness_y_true: Optional[List[int]] = None
    fairness_groups: Optional[List[Union[str, int]]] = None


class ConformalRequest(BaseModel):
    alpha: float = Field(default=0.05, gt=0.0, lt=1.0)
    probs_calib: List[float]
    y_calib: List[int]
    groups_calib: List[Union[str, int]]
    probs_test: List[float]
    groups_test: List[Union[str, int]]
    y_test: Optional[List[int]] = None


class FairnessRepairRequest(BaseModel):
    scores: List[float]
    y_true: List[int]
    groups: List[Union[str, int]]
    privileged_group: Union[str, int] = 1
    unprivileged_group: Union[str, int] = 0
    baseline_threshold: float = Field(default=0.50, gt=0.0, lt=1.0)
    min_dir: float = Field(default=0.80, gt=0.0)
    target_dir_floor: float = Field(default=0.825, gt=0.0)
    max_abs_eod: float = Field(default=0.03, gt=0.0)
    tp_profit: float = Field(default=1.0, gt=0.0)
    fp_cost: float = Field(default=1.25, gt=0.0)


@app.get("/", response_class=HTMLResponse)
def get_console() -> HTMLResponse:
    """Serve the interactive SR 11-7 / RBI Model Risk Governance Web Console."""
    if CONSOLE_PATH.exists():
        return HTMLResponse(content=CONSOLE_PATH.read_text(encoding="utf-8"))
    return HTMLResponse(
        content="<h1>mrmgov SR 11-7 Model Risk Governance Engine</h1><p>Console ready.</p>"
    )


@app.get("/health")
def health_check() -> Dict[str, object]:
    """Service health and regulatory metadata."""
    return {
        "status": "ok",
        "package": "mrmgov",
        "port": 8019,
        "frameworks": ["FED_OCC_SR_11_7", "RBI_MRM_2024", "ECOA_REG_B_FOUR_FIFTHS"],
    }


@app.post("/audit")
def post_audit(req: AuditRequest) -> Dict[str, object]:
    """
    Execute a complete SR 11-7 / RBI Model Governance Audit:
    1. Covariate vs Concept Drift Attribution
    2. Mondrian Conformal Prediction Abstention
    3. ECOA Four-Fifths Fair-Lending Repair
    4. Cryptographic SHA-256 Hash-Chained Audit Certificate
    """
    try:
        X_ref = np.asarray(req.X_ref, dtype=float)
        y_ref = np.asarray(req.y_ref, dtype=int)
        X_prod = np.asarray(req.X_prod, dtype=float)
        y_prod = np.asarray(req.y_prod, dtype=int)

        drift_res = analyze_drift(
            X_ref=X_ref,
            y_ref=y_ref,
            X_prod=X_prod,
            y_prod=y_prod,
            feature_names=req.feature_names,
            psi_threshold=req.psi_threshold,
            concept_slope_threshold=req.concept_slope_threshold,
        )

        # Derive reference model scores for conformal & fairness if not explicitly supplied
        mu = np.mean(X_ref, axis=0)
        sd = np.maximum(np.std(X_ref, axis=0), 1e-8)
        from .drift import _fit_logistic_Newton, _sigmoid

        w, b = _fit_logistic_Newton((X_ref - mu) / sd, y_ref, l2_reg=1.0)
        p_ref = _sigmoid(((X_ref - mu) / sd) @ w + b)
        p_prod = _sigmoid(((X_prod - mu) / sd) @ w + b)

        n_ref = len(y_ref)
        n_prod = len(y_prod)
        groups_ref = [str(i % 2) for i in range(n_ref)]
        groups_prod = (
            [str(g) for g in req.groups_prod]
            if req.groups_prod is not None
            else [str(i % 2) for i in range(n_prod)]
        )

        mondrian = MondrianConformalPredictor(alpha=req.alpha)
        mondrian.fit(p_ref.tolist(), y_ref.tolist(), groups_ref)
        conformal_res = mondrian.predict(
            p_prod.tolist(), groups_prod, y_test=y_prod.tolist()
        )

        f_scores = (
            req.fairness_scores if req.fairness_scores is not None else p_prod.tolist()
        )
        f_y = req.fairness_y_true if req.fairness_y_true is not None else y_prod.tolist()
        f_groups = (
            req.fairness_groups if req.fairness_groups is not None else groups_prod
        )
        unique_g = sorted(set(str(g) for g in f_groups))
        priv_g = unique_g[-1] if len(unique_g) >= 2 else "1"
        unpriv_g = unique_g[0] if len(unique_g) >= 2 else "0"

        fairness_res = repair_group_thresholds(
            scores=f_scores,
            y_true=f_y,
            groups=f_groups,
            privileged_group=priv_g,
            unprivileged_group=unpriv_g,
        )

        cert = generate_governance_certificate(
            model_id=req.model_id,
            model_version=req.model_version,
            drift_result=drift_res,
            conformal_result=conformal_res,
            fairness_result=fairness_res,
        )

        return {
            "drift": drift_res.to_dict(),
            "conformal": conformal_res.to_dict(),
            "fairness": fairness_res.to_dict(),
            "certificate": cert.to_dict(),
            "signature_verified": verify_certificate_signature(cert),
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/conformal/predict")
def post_conformal_predict(req: ConformalRequest) -> Dict[str, object]:
    """Calibrate Mondrian Conformal Predictor and generate group-conditional prediction sets."""
    try:
        predictor = MondrianConformalPredictor(alpha=req.alpha)
        predictor.fit(req.probs_calib, req.y_calib, req.groups_calib)
        res = predictor.predict(req.probs_test, req.groups_test, y_test=req.y_test)
        return res.to_dict()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/fairness/repair")
def post_fairness_repair(req: FairnessRepairRequest) -> Dict[str, object]:
    """Audit Disparate Impact Ratio and solve constrained group-threshold repair."""
    try:
        res = repair_group_thresholds(
            scores=req.scores,
            y_true=req.y_true,
            groups=req.groups,
            privileged_group=req.privileged_group,
            unprivileged_group=req.unprivileged_group,
            baseline_threshold=req.baseline_threshold,
            min_dir=req.min_dir,
            target_dir_floor=req.target_dir_floor,
            max_abs_eod=req.max_abs_eod,
            tp_profit=req.tp_profit,
            fp_cost=req.fp_cost,
        )
        return res.to_dict()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/bench")
def get_benchmark(seed: int = 42) -> Dict[str, object]:
    """Run and return the reproducible SR 11-7 / RBI Model Governance benchmark suite."""
    return run_benchmark_suite(seed=seed)
