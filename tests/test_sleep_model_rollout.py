"""Dynamic corrections, independent scenarios, and replay/rescore invariants."""
from copy import deepcopy
from datetime import timedelta
import socket
from unittest.mock import patch
import pytest
from app.sleep_model import evaluate
from app.sleep_model.contracts import configuration
from app.sleep_model.initialization import build_state
from app.sleep_model.normalize import dt,normalize
from app.planning.state import PlanningState
from tests.test_sleep_model_v2 import synthetic_snapshot,iso,opportunity,NOW


def test_a25_early_waking_revises_state_without_reset_and_repeated_event_is_idempotent():
    before=synthetic_snapshot(duty=False)
    machine=PlanningState(before)
    action={'id':'correction','action':'upsert','at_utc':before['now_utc'],
            'events':[dict(before['events'][2],end_utc=iso(NOW-timedelta(hours=7)),revision=2)]}
    assert machine.apply(action)
    assert not machine.apply(action)
    a=evaluate(before,scenarios=False,end_utc=iso(NOW+timedelta(hours=2)))
    b=evaluate(machine.snapshot,scenarios=False,end_utc=iso(NOW+timedelta(hours=2)))
    assert b['initial_state']['S']>a['initial_state']['S']
    assert b['initial_state']['input_prefix_hash']!=a['initial_state']['input_prefix_hash']
    with pytest.raises(ValueError,match='stale'):
        machine.apply(dict(action,id='stale',expected_revision=0))


def test_a25_current_sleep_observation_has_no_second_latency_or_future_actual_tail():
    snap=synthetic_snapshot(duty=False)
    snap['sessions']={'ongoing':{'id':'ongoing','actual_start_utc':iso(NOW-timedelta(hours=2)),
        'actual_end_utc':None,'expected_end_utc':iso(NOW+timedelta(hours=2)),'planned_block_id':'main'}}
    result=evaluate(snap,scenarios=False,end_utc=iso(NOW+timedelta(hours=4)))
    assert result['predicted_sleep'][0]['start_utc']==iso(NOW)
    assert result['trace'][0]['asleep'] is True
    assert result['physiological_metrics']['actual_sleep_hours']==0
    assert all(dt(p['start_utc'])>=NOW for p in result['predicted_sleep'])


def test_a63_sensitivity_recomputes_history_and_preserves_observations():
    snap=synthetic_snapshot(duty=False)
    block=opportunity(NOW+timedelta(hours=1),NOW+timedelta(hours=6))
    result=evaluate(snap,[block],end_utc=iso(NOW+timedelta(hours=8)),scenarios=True)
    scenarios={s['name']:s for s in result['scenarios']}
    assert set(scenarios)=={'latency_low','sleep_disrupted','light_dim','light_bright'}
    assert scenarios['latency_low']['initial_state']['S']==result['initial_state']['S']
    assert scenarios['sleep_disrupted']['initial_state']['x']==result['initial_state']['x']
    assert scenarios['light_dim']['initial_state']['x']!=scenarios['light_bright']['initial_state']['x']
    assert dt(scenarios['sleep_disrupted']['predicted_sleep'][0]['start_utc'])-dt(result['predicted_sleep'][0]['start_utc'])==timedelta(minutes=30)
    assert result['sensitivity_range'][0]<=result['utility']<=result['sensitivity_range'][1]


def test_a63_checkpoint_rejects_changed_prefix_and_rebuild_matches_uncached():
    snap=synthetic_snapshot(duty=False)
    normalized,_,_=normalize(snap,[],NOW)
    state,info=build_state(normalized,NOW,configuration())
    cached,_=build_state(normalized,NOW,configuration(),checkpoint=state)
    assert cached==state
    other=deepcopy(normalized)
    other['events'][0]['source']='revised-source'
    with pytest.raises(ValueError,match='checkpoint'):
        build_state(other,NOW,configuration(),checkpoint=state)


def test_a46_rescore_different_weights_reuses_same_physiological_trajectory_offline():
    snap=synthetic_snapshot(duty=False)
    block=opportunity(NOW+timedelta(hours=1),NOW+timedelta(hours=6))
    with patch.object(socket.socket,'connect',side_effect=AssertionError('network forbidden')):
        a=evaluate(snap,[block],scenarios=False,end_utc=iso(NOW+timedelta(hours=8)))
        b=evaluate(snap,[block],scenarios=False,end_utc=iso(NOW+timedelta(hours=8)),
                   config={'rank_weights':[.5,.3,.1,.1]})
    assert a['provenance']['trace_hash']==b['provenance']['trace_hash']
    assert a['utility']!=b['utility']
    assert a['provenance']['configuration_hash']!=b['provenance']['configuration_hash']


def test_a38_short_block_and_missed_ten_minute_opportunity_are_distinguished():
    snap=synthetic_snapshot(duty=False)
    short=opportunity(NOW+timedelta(hours=1),NOW+timedelta(hours=1,minutes=10))
    result=evaluate(snap,[short],scenarios=False,end_utc=iso(NOW+timedelta(hours=2)), config={'latency_minutes': 15.0})
    assert result['publishability']=='valid'
    assert result['predicted_sleep']==[]
    assert result['missed_opportunities']==[short['id']]


def test_a56_declaring_a_watch_does_not_create_observation_coverage_or_precision():
    snap=synthetic_snapshot(duty=False)
    snap['events']=[e for e in snap['events'] if e['kind']!='sleep_actual']
    snap['wearable']={'device':'watch','accuracy':'high'}
    result=evaluate(snap,scenarios=False,end_utc=iso(NOW+timedelta(hours=2)))
    assert result['physiological_metrics']['history_observation_coverage']==0
    assert 'confidence_interval' not in result


def test_a2_display_timezone_does_not_teleport_biological_state():
    snap=synthetic_snapshot(duty=False)
    a=evaluate(snap,scenarios=False,end_utc=iso(NOW+timedelta(hours=2)))
    snap['display_timezone']='Pacific/Auckland'
    b=evaluate(snap,scenarios=False,end_utc=iso(NOW+timedelta(hours=2)))
    assert a['provenance']['trace_hash']==b['provenance']['trace_hash']


def test_a46_preference_changes_ranking_loss_but_not_physiology():
    snap=synthetic_snapshot(duty=False)
    b=opportunity(NOW+timedelta(hours=7),NOW+timedelta(hours=15))
    a=evaluate(snap,[b],scenarios=False,end_utc=iso(NOW+timedelta(hours=24)))
    p={'id':'bed','kind':'bedtime','target':'23:00','timezone':'UTC','tolerance':0,
       'scope_start_utc':iso(NOW),'scope_end_utc':iso(NOW+timedelta(hours=24))}
    altered=evaluate(snap,[b],preferences=[p],scenarios=False,end_utc=iso(NOW+timedelta(hours=24)))
    assert a['provenance']['trace_hash']==altered['provenance']['trace_hash']
    assert a['losses']['physiological']==altered['losses']['physiological']
    assert altered['losses']['preference']>0
    assert altered['utility']<a['utility']


def test_a63_seven_day_score_and_phase_markers_converge_when_step_halves():
    snap=synthetic_snapshot(duty=False)
    snap['horizon_end_utc']=iso(NOW+timedelta(days=7))
    snap['events'][-1]['end_utc']=snap['horizon_end_utc']
    plans=[opportunity(NOW+timedelta(days=i,hours=11),NOW+timedelta(days=i,hours=19),identifier=f'night-{i}') for i in range(7)]
    a=evaluate(snap,plans,scenarios=False)
    b=evaluate(snap,plans,scenarios=False,config={'max_step_minutes':.5})
    assert a['utility'] is not None and b['utility'] is not None
    assert abs(a['utility']-b['utility'])<.05
    for name in ('S','x','xc','n'):
        assert abs(a['final_state'][name]-b['final_state'][name])<1e-5
    am=a['physiological_metrics']['phase_markers'];bm=b['physiological_metrics']['phase_markers']
    assert len(am)==len(bm)>0
    assert all(abs((dt(x['cbtmin_utc'])-dt(y['cbtmin_utc'])).total_seconds())<60 for x,y in zip(am,bm))


def test_a15_incomplete_history_does_not_saturate_future_sleep_target():
    snap=synthetic_snapshot(duty=False)
    result=evaluate(snap,scenarios=False,end_utc=iso(NOW+timedelta(hours=24)))
    assert result['physiological_metrics']['rolling_actual_sleep_hours']['72']>=24
    assert result['losses']['need']==1
    assert result['physiological_metrics']['sleep_hours']==0


def test_checkpoint_invalidated_when_late_report_becomes_visible():
    snap=synthetic_snapshot(duty=False)
    record={'id':'late','kind':'sleep_actual','start_utc':iso(NOW-timedelta(hours=2)),
            'end_utc':iso(NOW-timedelta(hours=1)),'observed_at_utc':iso(NOW+timedelta(hours=1))}
    original=evaluate(snap,scenarios=False,end_utc=iso(NOW+timedelta(hours=2)))
    snap['events'].append(record)
    unseen=evaluate(snap,scenarios=False,end_utc=iso(NOW+timedelta(hours=2)))
    assert unseen['initial_state']==original['initial_state']
    snap['now_utc']=iso(NOW+timedelta(hours=1))
    seen=evaluate(snap,scenarios=False,end_utc=iso(NOW+timedelta(hours=2)))
    assert seen['initial_state']['input_prefix_hash']!=unseen['initial_state']['input_prefix_hash']


def test_a63_portable_archive_and_score_only_reweight_never_integrates(tmp_path):
    from app.sleep_model.archive import archive_evaluation,reweight
    import json
    snap=synthetic_snapshot(duty=False)
    result=evaluate(snap,scenarios=False,end_utc=iso(NOW+timedelta(hours=2)))
    archive_evaluation(tmp_path,snap,[],result)
    manifest=json.loads((tmp_path/'model_manifest.json').read_text(encoding='utf-8'))
    assert manifest['upstream']['commit']=='db6eb08b6c0b0586cdac7795bf10bbbbfd323c1a'
    assert (tmp_path/'model_trace.jsonl').exists()
    with patch('app.sleep_model.simulation.simulate',side_effect=AssertionError('ODE forbidden')):
        changed=reweight(result,[.5,.3,.1,.1])
    assert changed['provenance']['trace_hash']==result['provenance']['trace_hash']
    assert changed['utility']!=result['utility']
    assert json.loads((tmp_path/'evaluation.json').read_text(encoding='utf-8'))==result
    with pytest.raises(FileExistsError,match='immutable'):
        archive_evaluation(tmp_path,snap,[],changed)
