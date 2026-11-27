from dtwin.generate import generate
from dtwin.pipeline import load_config

def test_shapes_and_binary():
    p,t,truth=generate(20,11)
    assert len(p)==20 and len(t)==240 and len(truth)==20
    assert set(t.treatment.unique()).issubset({0,1})

def test_potential_outcomes_consistent():
    _,_,truth=generate(40,12)
    assert (truth.true_y12_always<truth.true_y12_never).all()
    assert (truth.true_individual_effect<0).all()

def test_repeatable():
    p,t,q=generate(10,19)
    p2,t2,q2=generate(10,19)
    assert p.equals(p2) and t.equals(t2) and q.equals(q2)

def test_environment_configs():
    for env in ['dev','test','prod']:
        c=load_config(env)
        assert c['environment']==env and c['n_patients']>0

def test_synthetic_measurement_missingness():
    _,t,_=generate(30,15)
    assert t.loc[t.visit==0,'hba1c_observed'].isna().all()
    assert t.loc[t.visit==1,'hba1c_observed'].notna().all()
