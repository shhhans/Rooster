"""Independent source snapshot and numerical convergence checks (A1/2/5/6/7/63)."""
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from scipy.integrate import solve_ivp
from app.sleep_model import circadian
from app.sleep_model.contracts import configuration
from app.sleep_model.normalize import local_instant


def upstream_rhs():
    root=Path('app/sleep_model/sources/circadian')
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8-sig'))
    raw=(root/'models.py.txt').read_bytes()
    assert hashlib.sha256(raw).hexdigest()==manifest['sha256']
    tree=ast.parse(raw.decode())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='derv' and
              any(isinstance(d,ast.Call) and any(isinstance(a,ast.Name) and a.id=='Forger99' for a in d.args) for d in n.decorator_list))
    node.decorator_list=[];node.returns=None
    for a in node.args.args:a.annotation=None
    module=ast.Module(body=[node],type_ignores=[])
    namespace={'np':np}
    exec(compile(ast.fix_missing_locations(module),'pinned-upstream-Forger99','exec'),namespace)
    return lambda t,y,light:namespace['derv'](SimpleNamespace(**circadian.PARAMETERS),t,y,light)


def test_pinned_upstream_rhs_and_seven_day_trajectory():
    upstream=upstream_rhs()
    state=np.array([.2,-.4,.3])
    for light in (0.,50.,250.,1000.):
        np.testing.assert_allclose(circadian.rhs(0,state,light),upstream(0,state,light),atol=1e-12,rtol=0)
    cfg=configuration()
    state=np.array([-.0843259,-1.09607546,.45584306])
    actual,_=circadian.integrate(state,168.,250.,cfg)
    expected=solve_ivp(lambda t,y:upstream(t,y,250.),(0,168.),state,
                       method='DOP853',max_step=1/60,rtol=1e-8,atol=1e-10).y[:,-1]
    np.testing.assert_allclose(actual,expected,atol=1e-5,rtol=0)


def test_half_step_seven_day_state_converges():
    state=[-.0843259,-1.09607546,.45584306]
    a,_=circadian.integrate(state,168.,50.,configuration())
    b,_=circadian.integrate(state,168.,50.,configuration({'max_step_minutes':.5}))
    np.testing.assert_allclose(a,b,atol=1e-5,rtol=0)


def test_variable_light_fifteen_minute_cap_matches_one_minute_reference():
    """Changing solver cap is not changing the one-minute output or forced-input boundaries."""
    a=np.array([-.0843259,-1.09607546,.45584306]);b=a.copy()
    for day in range(7):
        for hours,light in ((7.,0.),(1.,50.),(8.,1000.),(4.,250.),(4.,50.)):
            a,_=circadian.integrate(a,hours,light,configuration({'max_step_minutes':15.}))
            b,_=circadian.integrate(b,hours,light,configuration({'max_step_minutes':1.}))
    np.testing.assert_allclose(a,b,atol=1e-5,rtol=0)


def test_dst_gap_and_fold_cannot_be_silently_guessed():
    with pytest.raises(ValueError,match='nonexistent'):
        local_instant('2027-03-14','02:30','America/New_York')
    with pytest.raises(ValueError,match='ambiguous'):
        local_instant('2027-11-07','01:30','America/New_York')
    a=local_instant('2027-11-07','01:30','America/New_York',0)
    b=local_instant('2027-11-07','01:30','America/New_York',1)
    assert (b-a).total_seconds()==3600
