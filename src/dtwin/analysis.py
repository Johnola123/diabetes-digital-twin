"""Educational longitudinal causal-analysis benchmark, not causal identification proof."""
import json
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from .pipeline import write_json

BASE=['age','sex','bmi','diabetes_years','baseline_hba1c']
HISTORY=BASE+['month','hba1c_history','previous_treatment','previous_visit']
OUTCOME=HISTORY+['treatment']

def propensity_prob(frame, predictors, target, train_mask=None):
    model=LogisticRegression(max_iter=650,solver='lbfgs')
    model.fit(frame.loc[train_mask,predictors] if train_mask is not None else frame[predictors],
              frame.loc[train_mask,target] if train_mask is not None else frame[target])
    return np.clip(model.predict_proba(frame[predictors])[:,1],.025,.975)

def analyze(dest,panel,potential,seed=41,max_weight=15):
    rng=np.random.default_rng(seed)
    pids=panel.patient_id.unique()
    train_ids,test_ids=train_test_split(pids,test_size=.25,random_state=seed)
    train=panel.patient_id.isin(train_ids)
    test=panel.patient_id.isin(test_ids)
    # Baseline descriptive summary deliberately does not infer treatment effectiveness.
    baseline=panel[panel.month==1].copy()
    baseline['started_treatment']=baseline.treatment
    grp=baseline.groupby('started_treatment')[['baseline_hba1c','age','bmi']].mean().round(3)
    grp.to_csv(dest/'baseline_group_summary.csv')
    pval=float(ttest_ind(baseline.loc[baseline.treatment==1,'baseline_hba1c'],
                         baseline.loc[baseline.treatment==0,'baseline_hba1c'],equal_var=False).pvalue)
    # Predict treatment using history. Marginal numerator conditions on time and previous treatment.
    tx_den=propensity_prob(panel,HISTORY,'treatment',train)
    tx_num=propensity_prob(panel,['month','previous_treatment'],'treatment',train)
    likelihood=lambda prob, observed:np.where(observed==1,prob,1-prob)
    treatment_ratios=likelihood(tx_num,panel.treatment.to_numpy())/likelihood(tx_den,panel.treatment.to_numpy())
    # Observation-process model: observed visits are not missing completely at random.
    visit_prob=propensity_prob(panel,HISTORY+['treatment'],'visit',train)
    visit_num=propensity_prob(panel,['month'],'visit',train)
    observed_ratios=likelihood(visit_num,panel.visit.to_numpy())/likelihood(visit_prob,panel.visit.to_numpy())
    work=panel.copy()
    work['tx_ratio']=np.clip(treatment_ratios,.15,6)
    work['visit_ratio']=np.clip(observed_ratios,.15,6)
    work['cum_tx_weight']=work.groupby('patient_id').tx_ratio.cumprod()
    work['cum_visit_weight']=work.groupby('patient_id').visit_ratio.cumprod()
    work['cum_treatment']=work.groupby('patient_id').treatment.cumsum()
    final=work[work.month==12].copy()
    final['weight_raw']=final.cum_tx_weight*final.cum_visit_weight
    final['weight_truncated']=final.weight_raw.clip(lower=.05,upper=max_weight)
    final['y12']=final.hba1c_observed
    # Marginal structural regression using stabilized, truncated combined weights.
    import statsmodels.api as sm
    Xcols=['cum_treatment']+BASE
    X=sm.add_constant(final[Xcols],has_constant='add')
    weighted=sm.WLS(final.y12,X,weights=final.weight_truncated).fit(cov_type='HC3')
    crude=sm.OLS(final.y12,X).fit(cov_type='HC3')
    msmeffect=float(12*weighted.params['cum_treatment'])
    crude_effect=float(12*crude.params['cum_treatment'])
    # Longitudinal g-computation. We fit transition on observed visits, then simulate
    # static regimes recursively. Its assumptions differ from MSM weighting.
    observed=panel[panel.visit==1].copy()
    obtrain=observed.patient_id.isin(train_ids)
    vtrain=panel[train].copy()
    obs_train=observed[obtrain].copy()
    p_obs=visit_prob[panel.visit.to_numpy()==1]
    obs_train_weights=1/np.clip(p_obs[obtrain.to_numpy()],.08,1)
    model=HistGradientBoostingRegressor(max_iter=85,max_leaf_nodes=16,l2_regularization=4,
                                        learning_rate=.06,random_state=seed)
    model.fit(obs_train[OUTCOME],obs_train.hba1c_observed,sample_weight=obs_train_weights)
    naive=HistGradientBoostingRegressor(max_iter=85,max_leaf_nodes=16,l2_regularization=4,
                                        learning_rate=.06,random_state=seed+1)
    naive.fit(obs_train[OUTCOME],obs_train.hba1c_observed)
    hold=observed[observed.patient_id.isin(test_ids)].copy()
    predicted=model.predict(hold[OUTCOME])
    naive_pred=naive.predict(hold[OUTCOME])
    mae=float(mean_absolute_error(hold.hba1c_observed,predicted))
    naive_mae=float(mean_absolute_error(hold.hba1c_observed,naive_pred))
    # Empirical interval widths from held-out observed visits; this is descriptive
    # and NOT a guarantee of calibrated counterfactual intervals.
    residuals=np.abs(hold.hba1c_observed.to_numpy()-predicted)
    interval_half=float(np.quantile(residuals,.90))
    eval_patients=panel[panel.patient_id.isin(test_ids)].sort_values(['patient_id','month'])
    base=eval_patients[eval_patients.month==1][['patient_id']+BASE].reset_index(drop=True)
    def simulate(fitted,regime):
        state=base.copy();state['hba1c_history']=state.baseline_hba1c
        state['previous_treatment']=0;state['previous_visit']=1
        trace=[]
        for month in range(1,13):
            state['month']=month;state['treatment']=regime
            y=fitted.predict(state[OUTCOME]);y=np.clip(y,5.1,14.0)
            trace.append(y.copy());state['hba1c_history']=y
            state['previous_treatment']=regime
            state['previous_visit']=1  # monitoring scenario fixed for comparability
        return np.column_stack(trace)
    sim0=simulate(model,0);sim1=simulate(model,1)
    sim0naive=simulate(naive,0);sim1naive=simulate(naive,1)
    traces=[]
    for i,row in base.iterrows():
        for month in range(12):
            traces.append({'patient_id':int(row.patient_id),'month':month+1,
                           'predicted_never':float(sim0[i,month]),'predicted_always':float(sim1[i,month]),
                           'scenario_difference':float(sim1[i,month]-sim0[i,month])})
    pd.DataFrame(traces).to_csv(dest/'twin_trajectories.csv',index=False)
    comparison=base[['patient_id','baseline_hba1c','age','sex','bmi']].copy()
    comparison['predicted_never']=sim0[:,-1]
    comparison['predicted_always']=sim1[:,-1]
    comparison['estimated_effect']=comparison.predicted_always-comparison.predicted_never
    comparison['estimated_effect_lower_illustrative']=comparison.estimated_effect-2*interval_half
    comparison['estimated_effect_upper_illustrative']=comparison.estimated_effect+2*interval_half
    truth=potential[potential.patient_id.isin(test_ids)]
    comparison=comparison.merge(truth,on='patient_id',validate='one_to_one')
    comparison['effect_error']=comparison.estimated_effect-comparison.true_individual_effect
    comparison.to_csv(dest/'counterfactual_evaluation.csv',index=False)
    # Weighted-vs-unweighted gformula comparison acts as monitoring-bias sensitivity.
    truth_ate=float(potential.true_individual_effect.mean())
    methods=pd.DataFrame([
        {'method':'Conventional regression (association)','effect':crude_effect,'absolute_error_vs_truth':abs(crude_effect-truth_ate)},
        {'method':'Stabilized IPTW MSM (assumption-dependent)','effect':msmeffect,'absolute_error_vs_truth':abs(msmeffect-truth_ate)},
        {'method':'Monitoring-weighted longitudinal g-computation','effect':float((sim1-sim0)[:,-1].mean()),'absolute_error_vs_truth':abs(float((sim1-sim0)[:,-1].mean())-truth_ate)},
        {'method':'Unweighted longitudinal g-computation','effect':float((sim1naive-sim0naive)[:,-1].mean()),'absolute_error_vs_truth':abs(float((sim1naive-sim0naive)[:,-1].mean())-truth_ate)},
        {'method':'Known synthetic data-generating truth','effect':truth_ate,'absolute_error_vs_truth':0.0},
    ])
    methods.to_csv(dest/'method_comparison.csv',index=False)
    ess=float(final.weight_truncated.sum()**2/(final.weight_truncated.pow(2).sum()))
    diagnostics={'baseline_treatment_group_hba1c_p_value':pval,'visit_rate':float(panel.visit.mean()),
       'raw_weight_max':float(final.weight_raw.max()),'truncated_weight_max':float(final.weight_truncated.max()),
       'weight_effective_sample_size':ess,'total_final_patients':int(len(final)),
       'holdout_observed_measurement_mae':mae,'holdout_naive_mae':naive_mae,
       'mean_true_synthetic_effect':truth_ate,'mean_twin_estimated_effect':float(comparison.estimated_effect.mean()),
       'individual_counterfactual_effect_mae':float(np.abs(comparison.effect_error).mean()),
       'illustrative_observed_outcome_interval_half_width':interval_half,
       'warnings':['Synthetic world only; no real-patient clinical validity',
                   'Unmeasured latent severity deliberately violates exchangeability',
                   'Static never/always policies may have limited positivity under observed treatment',
                   'Illustrative interval is based on observed-outcome residuals, NOT counterfactual coverage',
                   'No estimator is automatically selected by a numeric threshold']}
    write_json(dest/'diagnostics.json',diagnostics)
    pd.DataFrame({'patient_id':hold.patient_id,'month':hold.month,'actual_hba1c':hold.hba1c_observed,
                  'weighted_prediction':predicted,'unweighted_prediction':naive_pred}).to_csv(dest/'holdout_predictions.csv',index=False)
    return diagnostics,methods
