import json
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
from .generate import generate


def load_config(env):
    if env not in ('dev','test','prod'): raise ValueError('Environment must be dev/test/prod')
    return yaml.safe_load((Path(__file__).resolve().parents[2]/'configs'/f'{env}.yaml').read_text())

def write_json(path, data):
    Path(path).write_text(json.dumps(data,indent=2,default=lambda v:float(v) if isinstance(v,np.floating) else int(v)))

def bronze_silver_gold(cfg):
    dest=Path(cfg['artifact_dir']); dest.mkdir(parents=True,exist_ok=True)
    bronze=dest/'bronze';silver=dest/'silver';gold=dest/'gold'
    for d in [bronze,silver,gold]:d.mkdir(exist_ok=True)
    patients,panel,potential=generate(cfg['n_patients'],cfg['seed'])
    patients.to_csv(bronze/'patients.csv',index=False)
    panel.drop(columns=['true_hba1c_hidden','observed_visit_prob_hidden','treatment_prob_hidden']).to_csv(bronze/'visits.csv',index=False)
    # Hidden data are saved only as separate simulator truth, never included in analytical features.
    potential.to_csv(dest/'synthetic_counterfactual_truth_PRIVATE_TO_EVALUATION.csv',index=False)
    source=pd.read_csv(bronze/'visits.csv');p=pd.read_csv(bronze/'patients.csv')
    failures=[]
    if p.patient_id.duplicated().any():failures.append('duplicate patients')
    if source.duplicated(['patient_id','month']).any():failures.append('duplicate patient-month')
    if not source.patient_id.isin(p.patient_id).all():failures.append('orphan visit')
    if not source.month.between(1,12).all():failures.append('invalid months')
    if not set(source.treatment.unique()).issubset({0,1}):failures.append('invalid treatment')
    if not (source.groupby('patient_id').month.nunique()==12).all():failures.append('incomplete timelines')
    if (source.loc[source.visit==0,'hba1c_observed'].notna()).any():failures.append('unvisited recorded measurement')
    if (source.loc[source.visit==1,'hba1c_observed'].isna()).any():failures.append('visit missing measurement')
    report={'patients':len(p),'patient_months':len(source),'lab_missing_fraction':round(float(source.hba1c_observed.isna().mean()),4),'failures':failures,'passed':not failures}
    write_json(dest/'quality.json',report)
    if failures:raise ValueError('Failed quality gates: '+str(failures))
    source.to_csv(silver/'visits_validated.csv',index=False)
    p.to_csv(silver/'patients_validated.csv',index=False)
    data=source.merge(p,on='patient_id',validate='many_to_one').sort_values(['patient_id','month'])
    # Feature snapshot for month m is from observed data strictly BEFORE month m.
    data['hba1c_history']=data.groupby('patient_id').hba1c_observed.transform(lambda x:x.shift(1).ffill())
    data['hba1c_history']=data.hba1c_history.fillna(data.baseline_hba1c)
    data['previous_treatment']=data.groupby('patient_id').treatment.shift(1).fillna(0).astype(int)
    data['previous_visit']=data.groupby('patient_id').visit.shift(1).fillna(1).astype(int)
    data['observed_cumulative_treatment']=data.groupby('patient_id').treatment.cumsum()
    data['baseline_risk']=data.baseline_hba1c+0.02*(data.bmi-31)
    data.drop(columns=['observed_cumulative_treatment']).to_csv(gold/'panel.csv',index=False)
    return dest,data.drop(columns=['observed_cumulative_treatment']),potential,report
