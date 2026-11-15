"""Independent review counterexamples: resource ambiguity and evaluation-window metrics."""
from datetime import timedelta
import pytest
from app.sleep_model.contracts import configuration
from app.sleep_model.light import light_at
from app.sleep_model.metrics import measure
from tests.test_sleep_model_v2 import NOW,iso


def point(at,burden=.5):
    return {'time_utc':iso(at),'S':.5,'C':.5,'inertia':0.,'burden':burden,
            'asleep':False,'light_source':'assumed','x':.5,'xc':.5,'n':.5}


def test_a7_ambiguous_environment_resources_are_not_an_observed_location():
    home={'id':'home','kind':'environment','location':'home','timezone':'UTC',
          'start_utc':iso(NOW),'end_utc':iso(NOW+timedelta(hours=1))}
    plane=dict(home,id='plane',location='aircraft')
    profile={'home_timezone':'UTC'}
    a=light_at(NOW,False,[home,plane],profile)
    b=light_at(NOW,False,[plane,home],profile)
    assert a==b==(100.,'assumed_location_unknown')
    explicit=dict(plane,id='actual-location',kind='location')
    assert light_at(NOW,False,[home,plane,explicit],profile)==(50.,'assumed_aircraft')


def test_a44_fragmentation_outside_evaluation_window_does_not_leak_in():
    start,end=NOW,NOW+timedelta(hours=2)
    old=[{'id':'p1','kind':'sleep_actual','episode_id':'old','start_utc':iso(NOW-timedelta(hours=10)),
          'end_utc':iso(NOW-timedelta(hours=8))},
         {'id':'p2','kind':'sleep_actual','episode_id':'old','start_utc':iso(NOW-timedelta(hours=7)),
          'end_utc':iso(NOW-timedelta(hours=5))}]
    trace=[point(start),point(end)]
    snap={'now_utc':iso(end),'events':old}
    metrics,_=measure(trace,snap,[],start,end,configuration(),{})
    assert metrics['fragmentation_interruptions'] is None
    assert metrics['waso_hours'] is None


def test_a41_threshold_crossing_area_uses_clipped_linear_integral():
    end=NOW+timedelta(hours=1)
    trace=[point(NOW,.5),point(end,.7)]
    metrics,_=measure(trace,{'events':[]},[],NOW,end,configuration(),{})
    assert metrics['above_threshold_hours']==pytest.approx(.5)
    assert metrics['excess_area']==pytest.approx(.025)


def test_a63_explicit_preferences_are_hashed_archived_and_empty_override_clears(tmp_path):
    from app.sleep_model import evaluate
    from app.sleep_model.archive import archive_evaluation
    from tests.test_sleep_model_v2 import synthetic_snapshot
    import json
    snap=synthetic_snapshot(duty=False)
    snap['preferences']=[{'id':'sleep8','kind':'continuous_sleep','target':8,
        'scope_start_utc':iso(NOW),'scope_end_utc':iso(NOW+timedelta(hours=2))}]
    a=evaluate(snap,scenarios=False,end_utc=iso(NOW+timedelta(hours=2)))
    b=evaluate(snap,scenarios=False,end_utc=iso(NOW+timedelta(hours=2)),preferences=[])
    assert a['utility']<b['utility']
    assert a['provenance']['input_hash']!=b['provenance']['input_hash']
    archive_evaluation(tmp_path,snap,[],b)
    stored=json.loads((tmp_path/'model_inputs.json').read_text(encoding='utf-8'))
    replay=evaluate(stored['snapshot'],stored['opportunities'],preferences=stored['preferences'],
        mode=stored['mode'],start_utc=stored['start_utc'],end_utc=stored['end_utc'],config=stored['config'],scenarios=False)
    assert replay['utility']==b['utility']
    assert replay['provenance']['input_hash']==b['provenance']['input_hash']
