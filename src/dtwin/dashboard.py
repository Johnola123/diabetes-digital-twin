import json
import os
from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title='Diabetes Digital Twin',layout='wide')
st.title('Longitudinal Diabetes Digital Twin · Research Sandbox')
st.warning('Synthetic patients only. Model-based estimates are NOT validated clinical treatment effects or medical advice.')
env_options=['dev','test','prod']
active=os.getenv('APP_ENV','dev')
env=st.sidebar.selectbox('Environment',env_options,index=env_options.index(active) if active in env_options else 0)
dest=Path('artifacts')/env
if not (dest/'run_manifest.json').exists():
    st.error(f'No artifacts for {env}. Run: python -m dtwin.cli run --env {env}');st.stop()
read=lambda n:pd.read_csv(dest/n)
diagnostics=json.loads((dest/'diagnostics.json').read_text())
panel=read('gold/panel.csv');methods=read('method_comparison.csv');predictions=read('counterfactual_evaluation.csv');traces=read('twin_trajectories.csv')
tabs=st.tabs(['01 · Data quality & EDA','02 · Confounding & monitoring','03 · Model comparison','04 · Patient digital twin','05 · Validation & deployment'])
with tabs[0]:
    st.subheader('Pre-model: longitudinal cohort')
    a,b,c=st.columns(3);a.metric('Patients',panel.patient_id.nunique());b.metric('Patient-months',len(panel));c.metric('Missing HbA1c fraction',f'{panel.hba1c_observed.isna().mean():.1%}')
    st.plotly_chart(px.line(panel.groupby('month',as_index=False).agg(mean_hba1c=('hba1c_observed','mean')),
                             x='month',y='mean_hba1c',markers=True,title='Observed mean HbA1c by month (selection-affected)'),use_container_width=True)
    st.plotly_chart(px.histogram(panel[panel.month==1],x='baseline_hba1c',color='treatment',barmode='overlay',title='Baseline HbA1c by initial treatment decision'),use_container_width=True)
    st.dataframe(read('baseline_group_summary.csv'),use_container_width=True)
with tabs[1]:
    st.subheader('Competing estimators: method disagreement is a research finding')
    st.plotly_chart(px.bar(methods,x='method',y='effect',title='12-month HbA1c contrast, always vs never treated'),use_container_width=True)
    st.caption('Estimates are not automatically causal. Known truth exists only because the dataset is simulated.')
    a,b,c=st.columns(3);a.metric('Visit frequency',f'{diagnostics["visit_rate"]:.1%}');b.metric('Weight ESS',f'{diagnostics["weight_effective_sample_size"]:.0f}');c.metric('Weight maximum before truncation',f'{diagnostics["raw_weight_max"]:.1f}')
    st.dataframe(methods,use_container_width=True)
    st.markdown('**Expert decision:** compare assumptions, positivity, weight instability, residual confounding and monitoring sensitivity; do not choose solely by the closest synthetic truth.')
with tabs[2]:
    st.subheader('Post-model: holdout predictive checks')
    h=read('holdout_predictions.csv')
    c1,c2=st.columns(2);c1.metric('Monitoring-weighted observed outcome MAE',f'{diagnostics["holdout_observed_measurement_mae"]:.3f}');c2.metric('Unweighted observed outcome MAE',f'{diagnostics["holdout_naive_mae"]:.3f}')
    fig=px.scatter(h,x='actual_hba1c',y='weighted_prediction',title='Observed vs predicted HbA1c (held-out patients)');st.plotly_chart(fig,use_container_width=True)
    st.caption('Good prediction of measured HbA1c does not establish counterfactual treatment-effect accuracy.')
with tabs[3]:
    st.subheader('Virtual patient treatment scenarios')
    patient=int(st.selectbox('Synthetic patient ID',predictions.patient_id.tolist()))
    sub=traces[traces.patient_id==patient]
    plot=sub.melt(id_vars='month',value_vars=['predicted_never','predicted_always'],var_name='scenario',value_name='predicted_hba1c')
    st.plotly_chart(px.line(plot,x='month',y='predicted_hba1c',color='scenario',markers=True,title='12-month simulated HbA1c pathways'),use_container_width=True)
    row=predictions[predictions.patient_id==patient].iloc[0]
    a,b,c=st.columns(3);a.metric('Never treated',f'{row.predicted_never:.2f}');b.metric('Always treated',f'{row.predicted_always:.2f}');c.metric('Scenario difference',f'{row.estimated_effect:+.2f}')
    st.caption(f'Synthetic truth for evaluation: {row.true_individual_effect:+.2f}. Illustrative uncertainty band is NOT a calibrated individual causal interval.')
with tabs[4]:
    st.subheader('Evaluation with hidden synthetic counterfactual truth')
    st.metric('Individual treatment-effect MAE vs synthetic truth',f'{diagnostics["individual_counterfactual_effect_mae"]:.3f}')
    st.plotly_chart(px.scatter(predictions,x='true_individual_effect',y='estimated_effect',hover_data=['patient_id'],title='True vs estimated individual effect (simulation-world audit)'),use_container_width=True)
    for note in diagnostics['warnings']:st.write('•',note)
    st.markdown('**Promotion gate:** quality checks + tests + manual review of estimator disagreement. GitHub Actions jobs are ephemeral, not always-on application servers.')
