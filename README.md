# Longitudinal Type 2 Diabetes Digital Twin — Research Sandbox

**Synthetic-only, educational/research prototype. Not clinical decision support.**

This repository constructs a synthetic twelve-month diabetes cohort with time-varying treatment, changing HbA1c, informative observation patterns and hidden synthetic counterfactual truth. It compares conventional association, stabilized inverse-probability weighted marginal structural regression, observation-weighted longitudinal g-computation and unweighted g-computation. A Streamlit dashboard presents longitudinal EDA, estimator disagreement, monitoring sensitivity, held-out prediction checks, patient-specific scenarios and the simulation-world truth audit.

## Start from scratch

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src
pytest -q
python -m dtwin.cli run --env dev
python -m dtwin.cli check --env dev
streamlit run src/dtwin/dashboard.py
```

Windows PowerShell: `.venv\Scripts\Activate.ps1`, `$env:PYTHONPATH="src"`.

### Three local containerized application environments

```bash
docker compose up --build -d
docker compose ps
# http://localhost:8501  DEV
# http://localhost:8502  TEST
# http://localhost:8503  PROD
# docker compose logs -f dev
# docker compose down
```

These are separate local container processes, **not cloud servers or production-secure infrastructure**. Change ports if in use. Docker needs enough available RAM and CPU to fit all three pipelines.

### GitHub Actions manual selector

Create an empty GitHub repository and push this folder. In GitHub: **Actions → Run selected research environment → Run workflow → select dev/test/prod → Run workflow**. Artifacts can be downloaded from the completed run. GitHub Actions runners do not host persistent Streamlit services.

```bash
git init
git branch -M main
git add .
git commit -m "Initial diabetes digital twin research benchmark"
git remote add origin https://github.com/YOUR_USERNAME/diabetes-digital-twin.git
git push -u origin main
```

## Expert decision exercise

1. Examine estimator disagreement versus the known synthetic truth; disagreement cannot be resolved by an arbitrary numeric cutoff.
2. Diagnose time-varying confounding and the influence of latent unmeasured severity.
3. Compare observation-weighted with unweighted longitudinal g-computation to examine informative monitoring.
4. Inspect positivity and propensity weight effective sample size; understand bias–variance tradeoffs in truncation.
5. Compare prediction accuracy against counterfactual accuracy; a good HbA1c predictor need not recover treatment effects.
6. Record a defensible conclusion that acknowledges non-identifiability and treats model outputs as scenario estimates.

## Output files

`artifacts/<env>/bronze/`, `silver/`, `gold/` plus `method_comparison.csv`, `diagnostics.json`, `counterfactual_evaluation.csv`, `twin_trajectories.csv`, `holdout_predictions.csv`, `quality.json`. Synthetic counterfactual truth is isolated from features and used exclusively for evaluation.

## Modeling limitations

Synthetic DGP imposes a treatment benefit; that is not real evidence of clinical benefit. The DGP intentionally includes latent unmeasured severity affecting treatment and monitoring. Sequential exchangeability therefore fails and weighting/g-computation are not guaranteed unbiased. Observation-based models have misspecification and measurement-selection risks, and static never/always treatment policies may lack support. Predictions across interventions are model assumptions, not clinically validated individual causal effects. Patient-level uncertainty bands are deliberately labeled illustrative and not calibrated counterfactual intervals.

See `docs/RECORDING_SCRIPT.md` and the separate Word guide for a screen-recording sequence.
