"""Synthetic structural data generator. Never use these outputs for clinical guidance."""
import numpy as np
import pandas as pd


def generate(n=450, seed=41):
    rng = np.random.default_rng(seed)
    age = rng.integers(35, 79, size=n)
    sex = rng.integers(0, 2, size=n)
    bmi = np.clip(rng.normal(31, 5, n), 20, 48)
    duration = rng.integers(1, 16, size=n)
    baseline = np.clip(rng.normal(8.4 + .022*(bmi-31)+.022*(duration-7), .85, n), 6.4, 12.5)
    latent = rng.normal(0, 1, n)  # unavailable to analyst: deliberate unmeasured-confounding stressor
    patient = pd.DataFrame({'patient_id':np.arange(1,n+1), 'age':age,'sex':sex,'bmi':bmi,
                            'diabetes_years':duration,'baseline_hba1c':baseline})
    records=[]; truth=[]
    for i in range(n):
        shocks=rng.normal(0,.11,12)
        # Shared shocks across regimes give paired potential outcomes in the synthetic world.
        trajectories={}
        for regime in (0,1):
            h=float(baseline[i]); vals=[h]
            for month in range(1,13):
                effect = -.11*regime*(1+.11*latent[i])
                h=np.clip(h+.085+.009*(bmi[i]-31)+.008*(duration[i]-7)+.015*latent[i]+effect+shocks[month-1],5.2,14)
                vals.append(h)
            trajectories[regime]=vals
        truth.append({'patient_id':i+1,'true_y12_never':trajectories[0][-1],
                      'true_y12_always':trajectories[1][-1],
                      'true_individual_effect':trajectories[1][-1]-trajectories[0][-1]})
        h=float(baseline[i]); prev_tx=0; prev_seen=h
        for month in range(1,13):
            # Decisions respond to disease severity and past treatment; U is unobserved.
            logit=-1.0+1.05*(h-8.0)+.55*prev_tx+.35*latent[i]+.03*(bmi[i]-31)
            propensity=1/(1+np.exp(-np.clip(logit,-20,20)))
            treatment=int(rng.random()<propensity)
            h=np.clip(h+.085+.009*(bmi[i]-31)+.008*(duration[i]-7)+.015*latent[i]
                      -.11*treatment*(1+.11*latent[i])+shocks[month-1],5.2,14)
            # Worse health and prior visibility produce more frequent monitoring.
            visit_prob=1/(1+np.exp(-(-.85+.72*(h-8)+.45*prev_tx+.2*latent[i])))
            observed=bool(rng.random()<visit_prob or month==12)
            measured=np.clip(h+rng.normal(0,.10),5.1,14.5) if observed else np.nan
            records.append({'patient_id':i+1,'month':month,'treatment':treatment,
                            'visit':int(observed),'hba1c_observed':measured,
                            'true_hba1c_hidden':h,'observed_visit_prob_hidden':visit_prob,
                            'treatment_prob_hidden':propensity})
            prev_tx=treatment
            if observed:prev_seen=measured
    panel=pd.DataFrame(records)
    potential=pd.DataFrame(truth)
    return patient,panel,potential
