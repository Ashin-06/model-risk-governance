"""
FastAPI service for the model-risk checks (port 8019).

Endpoints:
  - GET  /                  : web console (HTML)
  - GET  /public-key        : Ed25519 keys that sign certificates (current and previous)
  - POST /audit             : drift + conformal + fairness + signed certificate (permission: audit)
  - POST /conformal/predict : Mondrian split conformal prediction sets            (check)
  - POST /fairness/repair   : disparate impact metrics and group-threshold search (check)
  - GET  /certificates      : certificates issued so far; /certificates/{id} one  (certificates)
  - GET  /bench             : benchmark on generated data                       (bench)
  - GET  /healthz, /readyz, /metrics (see ops.py)

Settings come from MRMGOV_* environment variables (ops.py). The signing key is
MRMGOV_SIGNING_KEY (base64 Ed25519 seed, required in production); every certificate
issued is kept in the database (register.py).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Dict, List, Optional, Union
import numpy as np
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from . import db, ops
from .drift import _fit_logistic_Newton, _sigmoid, analyze_drift
from .conformal import MondrianConformalPredictor
from .fairness import repair_group_thresholds
from .report import generate_governance_certificate
from .bench import run_benchmark_suite
from .register import Register

settings = ops.load_settings()
ops.setup_logging(settings.log_level)
store = db.connect(settings.database_url)
register = Register(store)

app = FastAPI(
    title="Model Risk Governance checks (mrmgov)",
    description=(
        "Covariate vs concept drift attribution, Mondrian conformal abstention and "
        "group fairness metrics, with an Ed25519-signed record of the results."
    ),
    version="1.0.0",
)
ops.install(app, settings, {"database": lambda: store.connect().close()})

CONSOLE_PATH = Path(__file__).resolve().parent.parent / "web" / "console.html"
MAX_ROWS = 10_000      # per X matrix
MAX_FEATURES = 50
MAX_POINTS = 50_000    # per 1-D list

# In development without MRMGOV_SIGNING_KEY the key is generated per process (ops.py warns),
# so certificates from an earlier run can't be checked against GET /public-key.
SIGNING_KEY = Ed25519PrivateKey.from_private_bytes(settings.secret["SIGNING_KEY"])


def _public_hex(seed: bytes) -> str:
    return Ed25519PrivateKey.from_private_bytes(seed).public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex()


PUBLIC_KEY_HEX = _public_hex(settings.secret["SIGNING_KEY"])
PREVIOUS_PUBLIC_KEYS = [_public_hex(k) for k in settings.previous["SIGNING_KEY"]]

Num = Annotated[float, Field(allow_inf_nan=False)]
Prob = Annotated[float, Field(ge=0.0, le=1.0)]
Label = Annotated[int, Field(ge=0, le=1)]
Group = Union[Annotated[str, Field(max_length=64)], int]
Row = Annotated[List[Num], Field(min_length=1, max_length=MAX_FEATURES)]


class AuditRequest(BaseModel):
    """Labels: y = 1 means default. Groups label each row (segment or protected attribute)."""

    model_id: str = Field(default="CREDIT-UNDERWRITER-V4.2", max_length=64)
    model_version: str = Field(default="2026.10", max_length=64)
    X_ref: List[Row] = Field(..., min_length=10, max_length=MAX_ROWS)
    y_ref: List[Label] = Field(..., max_length=MAX_ROWS)
    groups_ref: List[Group] = Field(..., max_length=MAX_ROWS)
    X_prod: List[Row] = Field(..., min_length=2, max_length=MAX_ROWS)
    y_prod: List[Label] = Field(..., max_length=MAX_ROWS)
    groups_prod: List[Group] = Field(..., max_length=MAX_ROWS)
    feature_names: Optional[List[Annotated[str, Field(max_length=64)]]] = Field(default=None, max_length=MAX_FEATURES)
    psi_threshold: float = Field(default=0.15, gt=0.0, le=10.0)
    concept_slope_threshold: float = Field(default=0.06, gt=0.0, le=10.0)
    alpha: float = Field(default=0.05, gt=0.0, lt=1.0)
    # Fairness convention: score = P(repay), y = 1 means repaid. Defaults: 1 - model score,
    # 1 - y_prod and groups_prod.
    fairness_scores: Optional[List[Prob]] = Field(default=None, max_length=MAX_ROWS)
    fairness_y_true: Optional[List[Label]] = Field(default=None, max_length=MAX_ROWS)
    fairness_groups: Optional[List[Group]] = Field(default=None, max_length=MAX_ROWS)
    privileged_group: Group = "1"
    unprivileged_group: Group = "0"


class ConformalRequest(BaseModel):
    alpha: float = Field(default=0.05, gt=0.0, lt=1.0)
    probs_calib: List[Prob] = Field(..., max_length=MAX_POINTS)
    y_calib: List[Label] = Field(..., max_length=MAX_POINTS)
    groups_calib: List[Group] = Field(..., max_length=MAX_POINTS)
    probs_test: List[Prob] = Field(..., max_length=MAX_POINTS)
    groups_test: List[Group] = Field(..., max_length=MAX_POINTS)
    y_test: Optional[List[Label]] = Field(default=None, max_length=MAX_POINTS)


class FairnessRepairRequest(BaseModel):
    scores: List[Prob] = Field(..., max_length=MAX_POINTS)
    y_true: List[Label] = Field(..., max_length=MAX_POINTS)
    groups: List[Group] = Field(..., max_length=MAX_POINTS)
    privileged_group: Group = 1
    unprivileged_group: Group = 0
    baseline_threshold: float = Field(default=0.50, gt=0.0, lt=1.0)
    min_dir: float = Field(default=0.80, gt=0.0, le=1.0)
    target_dir_floor: float = Field(default=0.825, gt=0.0, le=1.25)
    max_abs_eod: float = Field(default=0.03, gt=0.0, le=1.0)
    tp_profit: float = Field(default=1.0, gt=0.0, le=1e6)
    fp_cost: float = Field(default=1.25, gt=0.0, le=1e6)


@app.get("/", response_class=HTMLResponse)
def get_console() -> HTMLResponse:
    """Serve the web console."""
    if CONSOLE_PATH.exists():
        return HTMLResponse(content=CONSOLE_PATH.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>mrmgov</h1><p>web/console.html not found.</p>")


@app.get("/public-key")
def get_public_key() -> Dict[str, object]:
    """Ed25519 keys that sign certificates; fetch them once and pin them, don't trust them per-response."""
    return {"algorithm": "Ed25519", "public_key": PUBLIC_KEY_HEX, "previous_public_keys": PREVIOUS_PUBLIC_KEYS}


@app.post("/audit")
def post_audit(req: AuditRequest, ident: dict = Depends(ops.require("audit"))) -> Dict[str, object]:
    """
    Drift attribution, Mondrian conformal sets, group fairness metrics and a signed
    certificate. A logistic model is fitted on a random half of the reference rows and
    the conformal layer is calibrated on the other half, so calibration is out-of-sample.
    """
    try:
        X_ref = np.asarray(req.X_ref, dtype=float)
        y_ref = np.asarray(req.y_ref, dtype=int)
        X_prod = np.asarray(req.X_prod, dtype=float)
        y_prod = np.asarray(req.y_prod, dtype=int)
        groups_ref = np.asarray([str(g) for g in req.groups_ref])
        groups_prod = [str(g) for g in req.groups_prod]
        if groups_ref.size != y_ref.size:
            raise ValueError("groups_ref must have one entry per reference row.")

        mu = np.mean(X_ref, axis=0)
        sd = np.maximum(np.std(X_ref, axis=0), 1e-8)
        idx = np.random.default_rng(0).permutation(y_ref.size)
        fit_idx, cal_idx = idx[: y_ref.size // 2], idx[y_ref.size // 2 :]
        w, b = _fit_logistic_Newton((X_ref[fit_idx] - mu) / sd, y_ref[fit_idx], l2_reg=1.0)

        def model(X_in: np.ndarray) -> np.ndarray:
            return _sigmoid(((np.asarray(X_in, dtype=float) - mu) / sd) @ w + b)

        drift_res = analyze_drift(
            X_ref=X_ref,
            y_ref=y_ref,
            X_prod=X_prod,
            y_prod=y_prod,
            predict_proba_fn=model,
            feature_names=req.feature_names,
            psi_threshold=req.psi_threshold,
            concept_slope_threshold=req.concept_slope_threshold,
        )

        p_prod = model(X_prod)
        mondrian = MondrianConformalPredictor(alpha=req.alpha)
        mondrian.fit(model(X_ref[cal_idx]), y_ref[cal_idx], groups_ref[cal_idx])
        conformal_res = mondrian.predict(p_prod, groups_prod, y_test=y_prod)

        fairness_res = repair_group_thresholds(
            scores=req.fairness_scores if req.fairness_scores is not None else 1.0 - p_prod,
            y_true=req.fairness_y_true if req.fairness_y_true is not None else 1 - y_prod,
            groups=req.fairness_groups if req.fairness_groups is not None else groups_prod,
            privileged_group=req.privileged_group,
            unprivileged_group=req.unprivileged_group,
        )

        cert = generate_governance_certificate(
            model_id=req.model_id,
            model_version=req.model_version,
            drift_result=drift_res,
            conformal_result=conformal_res,
            fairness_result=fairness_res,
            signing_key=SIGNING_KEY,
        )
        register.add(cert, ident["user"] or "demo", PUBLIC_KEY_HEX)
        ops.DECISIONS.labels(decision=cert.overall_disposition).inc()

        return {
            "drift": drift_res.to_dict(),
            "conformal": conformal_res.to_dict(),
            "fairness": fairness_res.to_dict(),
            "certificate": cert.to_dict(),
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/conformal/predict")
def post_conformal_predict(req: ConformalRequest, ident: dict = Depends(ops.require("check"))) -> Dict[str, object]:
    """Calibrate the Mondrian conformal predictor and return group-conditional prediction sets."""
    try:
        predictor = MondrianConformalPredictor(alpha=req.alpha)
        predictor.fit(req.probs_calib, req.y_calib, req.groups_calib)
        res = predictor.predict(req.probs_test, req.groups_test, y_test=req.y_test)
        return res.to_dict()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/fairness/repair")
def post_fairness_repair(req: FairnessRepairRequest, ident: dict = Depends(ops.require("check"))) -> Dict[str, object]:
    """Disparate impact metrics and the group-threshold search (a diagnostic; see fairness.py)."""
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
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


_bench = lru_cache(maxsize=8)(run_benchmark_suite)


@app.get("/bench")
def get_benchmark(seed: int = Query(default=42, ge=0, le=2**32 - 1),
                  ident: dict = Depends(ops.require("bench"))) -> Dict[str, object]:
    """Run (or return the cached) benchmark on generated data."""
    return _bench(seed)


@app.get("/certificates")
def list_certificates(model_id: Optional[str] = Query(default=None, max_length=64),
                      limit: int = Query(default=50, ge=1, le=500),
                      ident: dict = Depends(ops.require("certificates"))) -> Dict[str, object]:
    """Certificates issued so far, newest first (summary fields only)."""
    return {"certificates": register.recent(model_id, limit)}


@app.get("/certificates/{certificate_id}")
def get_certificate(certificate_id: str, ident: dict = Depends(ops.require("certificates"))) -> Dict[str, object]:
    """One stored certificate, with who asked for it and the public key that signs it."""
    found = register.get(certificate_id[:64])
    if found is None:
        raise HTTPException(status_code=404, detail="no such certificate")
    return found
