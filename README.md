# Project 19 — SR 11-7 / RBI Model Risk Governance Engine (`mrmgov`)

An institutional-grade **SR 11-7 (Federal Reserve / OCC)** and **RBI Model Risk Management (MRM)** governance engine for credit underwriting and fraud scoring models.

---

## Key Capabilities & Mathematical Architecture

### 1. Covariate vs Concept Drift Attribution (`mrmgov/drift.py`)
Separates benign or recalibratable input population shifts $P_{\text{prod}}(X) \ne P_{\text{ref}}(X)$ (**Covariate Drift**) from structural relationship breakdown $P_{\text{prod}}(Y \mid X) \ne P_{\text{ref}}(Y \mid X)$ (**Concept Drift**):
- **Per-Feature Distributional Metrics**:
  - **Population Stability Index (PSI)**:
    $$\text{PSI}_j = \sum_{b=1}^{B} \left(p_{\text{prod}, b} - p_{\text{ref}, b}\right) \ln\left(\frac{p_{\text{prod}, b}}{p_{\text{ref}, b}}\right)$$
  - **Two-Sample Kolmogorov-Smirnov ($D_{\text{KS}}$)** statistic and Stephens finite-sample corrected $p$-value:
    $$D_{\text{KS}, j} = \sup_x \left|F_{\text{prod}, j}(x) - F_{\text{ref}, j}(x)\right|$$
  - **1D Wasserstein-1 (Earth Mover's) Distance ($W_1$)**:
    $$W_1(P_{\text{ref}, j}, P_{\text{prod}, j}) = \int_{-\infty}^{\infty} \left|F_{\text{ref}, j}(x) - F_{\text{prod}, j}(x)\right| dx$$
- **Exact Additive Importance-Weighted Brier Decomposition**:
  Using a regularized domain discriminator $\hat{q}(x) = P(D = \text{prod} \mid X = x)$ to estimate density ratios $w(x) = \frac{dP_{\text{prod}}(x)}{dP_{\text{ref}}(x)}$:
  $$\Delta \mathcal{L}_{\text{total}} = \underbrace{\left(\mathcal{L}_{\text{ref}}^{(w)} - \mathcal{L}_{\text{ref}}\right)}_{\Delta \mathcal{L}_{\text{covariate}}} + \underbrace{\left(\mathcal{L}_{\text{prod}} - \mathcal{L}_{\text{ref}}^{(w)}\right)}_{\Delta \mathcal{L}_{\text{concept}}}$$
- **Perturbation Shapley Attribution**:
  Quantifies each feature's exact share of prediction shift under quantile-transported interventions (`covariate_shapley_share`) and partial residual slope shifts (`concept_slope_shift`).

### 2. Mondrian (Group-Conditional) Split Conformal Prediction Abstention (`mrmgov/conformal.py`)
Wraps any binary credit/fraud probability scorer $\hat{p}(x) \in [0, 1]$ and calibrates non-conformity scores $s_i = 1 - \hat{p}_{y_i}(x_i)$ separately within each demographic/segment group $g$ at miscoverage rate $\alpha = 0.05$ using the finite-sample quantile:
$$\hat{q}_g = \text{Quantile}\left(\{s_i : G_i = g\},\; \frac{\lceil (n_g + 1)(1 - \alpha) \rceil}{n_g}\right)$$
Produces prediction sets $\mathcal{C}(x) \subseteq \{0, 1\}$ with guaranteed finite-sample group coverage $P(Y \in \mathcal{C}(X) \mid G = g) \ge 1 - \alpha$:
- $\mathcal{C}(x) = \{0\} \implies$ **`AUTO_APPROVE`** (guaranteed high-confidence low risk)
- $\mathcal{C}(x) = \{1\} \implies$ **`AUTO_DECLINE`** (guaranteed high-confidence default/fraud risk)
- $\mathcal{C}(x) = \{0, 1\} \implies$ **`ABSTAIN_HUMAN_UNDERWRITER`** (ambiguous region routed to manual underwriting)

### 3. ECOA Fair-Lending Disparate Impact Auditor & Constrained Threshold Repair (`mrmgov/fairness.py`)
Audits and repairs group-conditional approval thresholds $(t_{\text{priv}}^*, t_{\text{unpriv}}^*)$ to maximize expected bank lending profit and $F_1$ subject to:
- **ECOA / EEOC Four-Fifths Rule**:
  $$\text{DIR} = \frac{P(\hat{Y}=1 \mid G=\text{unprivileged})}{P(\hat{Y}=1 \mid G=\text{privileged})} \ge 0.80$$
- **Equal Opportunity Parity**:
  $$|\text{EOD}| = \left|\text{TPR}_{\text{unprivileged}} - \text{TPR}_{\text{privileged}}\right| \le 0.03$$

### 4. Tamper-Evident SHA-256 Hash-Chained Audit Certificate (`mrmgov/report.py`)
Produces a deterministic cryptographic audit trail chaining `genesis_header_hash -> drift_block_hash -> conformal_block_hash -> fairness_block_hash -> certificate_root_hash`.

---

## Quickstart & Verification

```bash
# Run full pytest suite
python -m pytest -q

# Run SR 11-7 / RBI Model Governance benchmark suite
python -m mrmgov bench

# Start FastAPI server & Web Console on port 8019
python -m mrmgov serve --port 8019
```
