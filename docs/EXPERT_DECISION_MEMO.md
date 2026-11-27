# Expert decision memo — DEV synthetic research run

## Scientific estimand
Mean 12-month HbA1c difference between static always-treated and never-treated regimes **within the synthetic structural data-generating world**.

## Conflicting estimates (example DEV run; verify new execution)
- Conventional adjusted outcome regression: roughly -0.62 HbA1c percentage points.
- Stabilized IPTW MSM: roughly -1.10 points.
- Monitoring-weighted longitudinal g-computation: roughly -0.25 points.
- Unweighted longitudinal g-computation: roughly -0.29 points.
- Synthetic causal truth: roughly -1.31 points.

## Professional decision
Do **not** nominate a medically valid winning model. The simulator intentionally creates unmeasured severity affecting treatment and visit frequency, so no conventional estimator meets exchangeability. Weights have nontrivial tails and truncation introduces bias/variance trade-offs; g-computation depends on transition and monitoring model specification. The known simulation truth permits methodological audits, not validation of outcomes in actual patients.

## Findings to show in recording
1. Propensity weight maximum before truncation vs effective sample size.
2. Change in effect from unweighted to observation-weighted g-computation.
3. Distinction between held-out outcome MAE and synthetic individual effect MAE.
4. Why the adjusted methods disagree and what is unknowable from observational features alone.

## Gate
Quality and integration artifacts can pass for a **research sandbox**, but no model is authorized for patient-facing clinical treatment advice. Future advancement would require stronger observed confounders, identification justification, calibration in independent real cohorts, and governance review.
