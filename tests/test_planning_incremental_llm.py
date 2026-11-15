"""Offline tool-loop acceptance; scripted Agent decisions, actual sleep model.

D2 selector tests are archived in the task directory and Git; their selector
contract is superseded. No test here calls an online provider.
"""
from copy import deepcopy
from datetime import timedelta
import json
import pytest
from app.planning import incremental as inc
from app.planning.agent import ModelClient
from app.planning.provider import MODEL
from app.planning.state import digest
from app.planning.archive import write
from tests.test_planning_incremental import make_snapshot, block, BASE_NOW, iso


def action(tool, **arguments):
    return {'tool': tool, 'arguments': arguments, 'explanation': '根据实际结果与本次用户意愿修订'}


class Script:
    def __init__(self, *actions):
        self.actions, self.calls = list(actions), []

    def complete(self, stage_id, system, body, validate):
        self.calls.append({'stage_id': stage_id, 'system': system, 'body': deepcopy(body)})
        item = self.actions.pop(0)
        if isinstance(item, Exception):
            raise item
        return deepcopy(item)


def window(snap):
    return {'start_utc': snap['now_utc'], 'end_utc': snap['horizon_end_utc']}


def add(start, hours=8):
    return {'op': 'add', 'after': {'start_utc': iso(start), 'end_utc': iso(start+timedelta(hours=hours)),
                                 'timezone':'UTC','role':'main'}}


def move(block_id, start, hours=8):
    return {'op':'move','event_id':block_id,'after':{
        'start_utc':iso(start),'end_utc':iso(start+timedelta(hours=hours))}}


def run(draft, client):
    return inc.step(draft, mode='continue', limit=len(client.actions), client=client)


def test_modes_remain_explicit():
    with pytest.raises(inc.DecisionClientRequiredError):
        inc.begin_run(make_snapshot(), request={'kind':'initial'})
    with pytest.raises(inc.InvalidRequestError):
        inc.begin_run(make_snapshot(), request={'kind':'initial'}, decision_mode='deterministic', client=Script())
    draft = inc.begin_run(make_snapshot(), request={'kind':'initial'}, decision_mode='deterministic')
    assert draft['provenance']['decision']['degraded']
    assert inc.step(draft, mode='continue', limit=100)['step_status']=='complete'


def test_fresh_roster_multistep_40day_full_model_horizon():
    snap = make_snapshot(days=40)
    actions = [action('read', **window(snap)), action('search', start_utc=snap['now_utc'],
               end_utc=iso(BASE_NOW+timedelta(hours=36)), strategy='beam')]
    for first in range(0,40,8):
        actions.extend([action('edit', edits=[add(BASE_NOW+timedelta(days=d,hours=16))
                                            for d in range(first,min(40,first+8))]),
                        action('evaluate'),action('retain')])
    actions.append(action('submit', **window(snap), disposition='planned'))
    client = Script(*actions)
    draft = inc.begin_run(snap, request={'kind':'initial'}, client=client, run_id='roster-trace')
    assert run(draft,client)['step_status']=='complete'
    assert len(draft['plan']['sleep_blocks'])==40
    assert all(s['status']=='processed' for s in draft['segments'])
    evidence=[r['tool_result']['evaluation'] for r in draft['step_log'] if r.get('tool')=='evaluate']
    assert len(evidence)==5
    assert all(e['evaluation_start_utc']==snap['now_utc'] and
               e['evaluation_end_utc']==snap['horizon_end_utc'] for e in evidence)
    assert all(e['input_hash'] and e['trace_hash'] and e['final_state'] for e in evidence)
    assert len({e['trace_hash'] for e in evidence})==5
    for call in client.calls:
        raw=json.dumps(call['body'])
        assert not any('"'+k+'"' in raw for k in ('segments','submissions','processed','coverage','propagation','ripple'))


def test_agent_may_retain_lower_utility_after_real_evaluation():
    old=block('s1',BASE_NOW+timedelta(hours=16),BASE_NOW+timedelta(hours=24))
    snap=make_snapshot(days=1,active_plan={'sleep_blocks':[old]})
    client=Script(action('edit',edits=[move('s1',BASE_NOW+timedelta(hours=16),1)]),
                  action('evaluate'),action('retain'))
    draft=inc.begin_run(snap,request={'kind':'initial'},client=client)
    baseline=draft['baseline']['utility']
    run(draft,client)
    assert draft['agent_state']['last_evaluation']['utility']<baseline
    assert draft['plan']['sleep_blocks'][0]['end_utc']==iso(BASE_NOW+timedelta(hours=17))


def test_agent_selected_evaluation_range_and_retain_invalidation():
    snap=make_snapshot(days=3)
    start,end=iso(BASE_NOW),iso(BASE_NOW+timedelta(hours=24))
    late=iso(BASE_NOW+timedelta(hours=16))
    client=Script(action('edit',edits=[add(BASE_NOW+timedelta(hours=16))]),
                  action('evaluate',start_utc=start,end_utc=end),action('retain'))
    draft=inc.begin_run(snap,request={'kind':'initial'},client=client)
    sequence_zero_key=inc._cache_key(draft,0)
    from app.planning.incremental_agent import _state
    state=_state(draft)
    state['submissions']=[{'start_utc':start,'end_utc':iso(BASE_NOW+timedelta(hours=8))},
                           {'start_utc':iso(BASE_NOW+timedelta(hours=8)),'end_utc':late},
                           {'start_utc':iso(BASE_NOW+timedelta(days=1)),'end_utc':iso(BASE_NOW+timedelta(days=2))}]
    run(draft,client)
    evaluation=next(r['tool_result'] for r in draft['step_log'] if r.get('tool')=='evaluate')
    assert evaluation['comparison']['range']=={'start_utc':start,'end_utc':end}
    assert evaluation['comparison']['baseline']['evaluation_start_utc']==start
    assert evaluation['comparison']['trial']['evaluation_end_utc']==end
    assert sequence_zero_key in draft['score_cache']
    assert inc._cache_key(draft,1) not in draft['score_cache']
    assert state['submissions']==[{'start_utc':start,'end_utc':iso(BASE_NOW+timedelta(hours=8))},
                                  {'start_utc':iso(BASE_NOW+timedelta(hours=8)),'end_utc':late}]


def test_local_retain_submit_windows_complete_initial_plan_without_losing_prefix():
    snap=make_snapshot(days=3)
    actions=[]
    for day in range(3):
        start=iso(BASE_NOW+timedelta(days=day))
        end=iso(BASE_NOW+timedelta(days=day+1))
        actions.append(action('read',start_utc=start,end_utc=end))
        actions.append(action('edit',edits=[add(BASE_NOW+timedelta(days=day,hours=16))]))
        actions.append(action('evaluate',start_utc=start,end_utc=end))
        actions.append(action('retain'))
        actions.append(action('submit',start_utc=start,end_utc=end,disposition='planned'))
    client=Script(*actions)
    draft=inc.begin_run(snap,request={'kind':'initial'},client=client)
    result=run(draft,client)
    assert result['step_status']=='complete'
    assert len(draft['plan']['sleep_blocks'])==3
    assert len(draft['agent_state']['submissions'])==3
    assert all(s['status']=='processed' for s in draft['segments'])


def test_feedback_local_completion_preserves_distant_recommendation():
    target=block('target',BASE_NOW+timedelta(hours=16),BASE_NOW+timedelta(hours=24),origin_event_id='evt1')
    distant=block('distant',BASE_NOW+timedelta(days=2,hours=16),BASE_NOW+timedelta(days=2,hours=24))
    snap=make_snapshot(days=3,active_plan={'sleep_blocks':[target,distant]},
                       extra_events=[dict(target,id='evt1',kind='sleep_plan')])
    start=iso(BASE_NOW+timedelta(hours=16)); end=iso(BASE_NOW+timedelta(hours=26))
    client=Script(action('read',start_utc=start,end_utc=end),
                  action('edit',edits=[move('target',BASE_NOW+timedelta(hours=17))]),
                  action('evaluate',start_utc=start,end_utc=end),action('retain'),
                  action('submit',start_utc=start,end_utc=end,disposition='planned'))
    draft=inc.begin_run(snap,request={'kind':'feedback','target_event_id':'evt1','meaning':'replace'},client=client)
    assert run(draft,client)['step_status']=='complete'
    assert next(b for b in draft['plan']['sleep_blocks'] if b['id']=='distant')==distant
    assert next(b for b in draft['plan']['sleep_blocks'] if b['id']=='target')['start_utc']==iso(BASE_NOW+timedelta(hours=17))


def test_feedback_completion_requires_ranges_for_other_retained_edits_too():
    target=block('target',BASE_NOW+timedelta(hours=16),BASE_NOW+timedelta(hours=24),origin_event_id='evt1')
    distant=block('distant',BASE_NOW+timedelta(days=2,hours=16),BASE_NOW+timedelta(days=2,hours=24))
    snap=make_snapshot(days=3,active_plan={'sleep_blocks':[target,distant]},
                       extra_events=[dict(target,id='evt1',kind='sleep_plan')])
    target_range={'start_utc':iso(BASE_NOW+timedelta(hours=16)),'end_utc':iso(BASE_NOW+timedelta(hours=26))}
    client=Script(action('read',**target_range),action('edit',edits=[move('target',BASE_NOW+timedelta(hours=17))]),
                  action('evaluate',**target_range),action('retain'),
                  action('edit',edits=[move('distant',BASE_NOW+timedelta(days=2,hours=15))]),
                  action('evaluate',start_utc=snap['now_utc'],end_utc=snap['horizon_end_utc']),action('retain'),
                  action('submit',**target_range,disposition='planned'))
    draft=inc.begin_run(snap,request={'kind':'feedback','target_event_id':'evt1','meaning':'replace'},client=client)
    result=run(draft,client)
    assert len(draft['agent_state']['retained_edit_ranges'])==4
    assert result['step_status']!='complete'
    assert len(draft['plan']['sleep_blocks'])==2


def test_local_evaluation_replays_physiology_from_real_now():
    from app.planning.evaluation import evaluate
    snap=make_snapshot(days=3)
    block_one=block('one',BASE_NOW+timedelta(hours=16),BASE_NOW+timedelta(hours=24))
    full=evaluate(snap,[block_one],end_utc=iso(BASE_NOW+timedelta(hours=48)),scenarios=False)
    local=evaluate(snap,[block_one],start_utc=iso(BASE_NOW+timedelta(hours=24)),
                   end_utc=iso(BASE_NOW+timedelta(hours=48)),scenarios=False)
    assert full['status']==local['status']=='complete'
    assert full['final_state']==local['final_state']
    assert local['evaluation_start_utc']==iso(BASE_NOW+timedelta(hours=24))


def test_out_of_range_schedule_does_not_change_local_change_cost():
    from app.planning.evaluation import evaluate
    baseline=block('local',BASE_NOW+timedelta(hours=16),BASE_NOW+timedelta(hours=24))
    far=block('far',BASE_NOW+timedelta(days=2,hours=16),BASE_NOW+timedelta(days=2,hours=24))
    snap=make_snapshot(days=3,active_plan={'sleep_blocks':[baseline]})
    snap['preferences']=[{
        'id':'local-change','kind':'change_cost','scope_start_utc':iso(BASE_NOW),
        'scope_end_utc':iso(BASE_NOW+timedelta(hours=24)),'weight':1}]
    result=evaluate(snap,[baseline,far],start_utc=snap['now_utc'],
                    end_utc=iso(BASE_NOW+timedelta(hours=24)),scenarios=False)
    assert result['status']=='complete'
    assert result['preference_metrics']['items'][0]['loss']==0


@pytest.mark.parametrize('operation', ['move', 'remove'])
def test_retain_rejects_evaluation_that_misses_edited_interval(operation):
    old=block('target',BASE_NOW+timedelta(hours=16),BASE_NOW+timedelta(hours=24))
    snap=make_snapshot(days=2,active_plan={'sleep_blocks':[old]})
    edits=[move('target',BASE_NOW+timedelta(hours=30))] if operation=='move' else [{'op':'remove','event_id':'target'}]
    client=Script(action('edit',edits=edits),
                  action('evaluate',start_utc=snap['now_utc'],end_utc=iso(BASE_NOW+timedelta(hours=20))),
                  action('retain'))
    draft=inc.begin_run(snap,request={'kind':'initial'},client=client)
    run(draft,client)
    assert draft['plan']['sleep_blocks']==[old]
    assert draft['agent_state']['observation']['status']=='rejected'
    assert 'cover every edited before and after interval' in draft['agent_state']['observation']['error']


def test_full_range_evaluate_repopulates_missing_compatibility_cache():
    snap=make_snapshot(days=2)
    client=Script(action('evaluate'))
    draft=inc.begin_run(snap,request={'kind':'initial'},client=client,budget={'max_model_calls':0})
    inc.resume_run(draft,budget_extension={'max_model_calls':2},client=client)
    run(draft,client)
    assert inc._current_score_entry(draft)['status']=='complete'
    assert draft['agent_state']['last_comparison']['range']==window(snap)


def test_beam_search_filters_oversized_candidates_without_truncating(monkeypatch):
    import app.planning.search as search
    from app.planning.incremental_proposals import search_seed_proposals
    def schedule(count):
        return [{'id':f'tmp-{i}','start_utc':iso(BASE_NOW+timedelta(hours=i*2)),
                 'end_utc':iso(BASE_NOW+timedelta(hours=i*2+1)),'role':'main','timezone':'UTC'}
                for i in range(count)]
    monkeypatch.setattr(search,'generate',lambda *args,**kwargs: ([schedule(9),schedule(2)],{'status':'ok'}))
    proposals=search_seed_proposals({'now_utc':iso(BASE_NOW)}, {'omitted_spans':[]},
        {'start_utc':iso(BASE_NOW),'end_utc':iso(BASE_NOW+timedelta(hours=24))},True)
    assert len(proposals)==1 and len(proposals[0]['blocks'])==2


def test_edit_fingerprint_ignores_temporary_block_ids():
    from app.planning.incremental_agent import _edit_fingerprint
    left=[{'op':'add','after':{'id':'tmp-a','start_utc':iso(BASE_NOW),'end_utc':iso(BASE_NOW+timedelta(hours=1))}}]
    right=[{'op':'add','after':{'id':'tmp-b','start_utc':iso(BASE_NOW),'end_utc':iso(BASE_NOW+timedelta(hours=1))}}]
    assert _edit_fingerprint(left)==_edit_fingerprint(right)


def test_reverted_candidate_is_annotated_and_may_be_retried_with_same_reason():
    edit=[add(BASE_NOW+timedelta(hours=16))]
    client=Script(action('edit',edits=edit),action('evaluate'),action('revert'),
                  action('edit',edits=edit))
    draft=inc.begin_run(make_snapshot(days=2),request={'kind':'initial'},client=client)
    run(draft,client)
    assert draft['agent_state']['candidate_attempts'][0]['disposition']=='reverted'
    assert draft['agent_state']['candidate_attempts'][0]['revert_reason']
    staged=[r['tool_result'] for r in draft['step_log'] if r.get('tool')=='edit'][-1]
    assert len(staged['prior_attempts'])==1
    assert draft['agent_state']['trial'] is not None


def test_plan_evidence_rejects_evaluation_after_facts_version_changes():
    snap=make_snapshot(days=1)
    client=Script(action('evaluate'))
    draft=inc.begin_run(snap,request={'kind':'initial'},client=client)
    run(draft,client)
    from app.planning.incremental_agent import _planning_evidence, _state
    state=_state(draft)
    assert state['last_comparison']['facts_version']==draft['facts_digest']
    draft['facts_digest']='changed-facts-version'
    assert _planning_evidence(draft,state)==[]


def test_react3_restore_keeps_published_plan_but_drops_unverifiable_confirmation():
    from app.planning.incremental_agent import _state
    published=block('published',BASE_NOW+timedelta(hours=16),BASE_NOW+timedelta(hours=24))
    draft=inc.begin_run(make_snapshot(days=1,active_plan={'sleep_blocks':[published]}),
                        request={'kind':'initial'},client=Script())
    draft['agent_state']={'version':'incremental-react-3','trial':{'stale':True},
                          'last_evaluation':{'status':'complete'},'submissions':[window(draft['snapshot'])]}
    draft['pending_confirmations']=[{'confirmation_id':'old'}]
    draft['status']='needs_confirmation'
    state=_state(draft)
    assert state['version']=='incremental-react-4' and state['trial'] is None
    assert draft['plan']['sleep_blocks']==[published]
    assert draft['pending_confirmations']==[] and draft['status']=='pending'
    assert state['last_comparison'] is None and state['submissions']==[]


def test_left_swipe_target_authority_revert_then_associated_edit():
    old=block('s1',BASE_NOW+timedelta(hours=16),BASE_NOW+timedelta(hours=24),origin_event_id='evt1',accepted=True)
    snap=make_snapshot(days=2,active_plan={'sleep_blocks':[old]},extra_events=[dict(old,id='evt1',kind='sleep_plan')])
    client=Script(action('read',**window(snap)),action('search',**window(snap),strategy='habitual'),
                  action('edit',edits=[move('s1',BASE_NOW+timedelta(hours=17))]),action('evaluate'),action('revert'),
                  action('edit',edits=[move('s1',BASE_NOW+timedelta(hours=18)),add(BASE_NOW+timedelta(hours=40))]),
                  action('evaluate'),action('retain'),action('submit',**window(snap),disposition='planned'))
    draft=inc.begin_run(snap,request={'kind':'feedback','target_event_id':'evt1','meaning':'replace',
                                    'reason':'希望晚一点'},client=client)
    assert run(draft,client)['step_status']=='complete'
    assert not draft['pending_confirmations']
    assert next(b for b in draft['plan']['sleep_blocks'] if b['id']=='s1')['start_utc']==iso(BASE_NOW+timedelta(hours=18))
    assert draft['snapshot']==snap


def test_omit_cannot_reinsert_equivalent_occurrence():
    old=block('s1',BASE_NOW+timedelta(hours=16),BASE_NOW+timedelta(hours=24),origin_event_id='evt1')
    snap=make_snapshot(active_plan={'sleep_blocks':[old]},extra_events=[dict(old,id='evt1',kind='sleep_plan')])
    client=Script(action('edit',edits=[{'op':'remove','event_id':'s1'}]),action('evaluate'),action('retain'),
                  action('edit',edits=[add(BASE_NOW+timedelta(hours=17),6)]))
    draft=inc.begin_run(snap,request={'kind':'feedback','target_event_id':'evt1','meaning':'omit_occurrence'},client=client)
    run(draft,client)
    assert not draft['plan']['sleep_blocks']
    assert draft['agent_state']['observation']['status']=='rejected'
    assert 'forbidden' in draft['agent_state']['observation']['error']


@pytest.mark.parametrize('last_decision',['apply','reject'])
def test_accepted_collateral_batch_confirmation_is_atomic(last_decision):
    a=block('a',BASE_NOW+timedelta(hours=2),BASE_NOW+timedelta(hours=6),accepted=True)
    b=block('b',BASE_NOW+timedelta(hours=15),BASE_NOW+timedelta(hours=19),accepted=True)
    snap=make_snapshot(active_plan={'sleep_blocks':[a,b]})
    client=Script(action('edit',edits=[move('a',BASE_NOW+timedelta(hours=3),4),
                                     move('b',BASE_NOW+timedelta(hours=16),4)]),action('evaluate'),action('retain'))
    draft=inc.begin_run(snap,request={'kind':'initial'},client=client)
    result=run(draft,client)
    assert result['step_status']=='needs_confirmation'
    assert draft['plan']['sleep_blocks']==[a,b]
    first,second=[c['confirmation_id'] for c in draft['pending_confirmations']]
    inc.resume_run(draft,input={'kind':'confirmation','confirmation_id':first,'decision':'apply'},client=client)
    assert draft['status']=='needs_confirmation' and draft['plan']['sleep_blocks']==[a,b]
    inc.resume_run(draft,input={'kind':'confirmation','confirmation_id':second,'decision':last_decision},client=client)
    assert (draft['plan']['sleep_blocks']==[a,b])==(last_decision=='reject')
    assert not draft['pending_confirmations']


def test_budget_json_resume_preserves_trial_without_replaying_edits():
    client=Script(action('edit',edits=[add(BASE_NOW+timedelta(hours=16))]),action('evaluate'))
    draft=inc.begin_run(make_snapshot(),request={'kind':'initial'},client=client,budget={'max_llm_calls':2})
    run(draft,client)
    assert inc.step(draft,client=client)['step_status']=='budget_exhausted'
    restored=json.loads(json.dumps(draft))
    client2=Script(action('retain'))
    inc.resume_run(restored,budget_extension={'max_llm_calls':4},client=client2)
    inc.step(restored,client=client2)
    assert len(restored['plan']['sleep_blocks'])==1
    assert restored['usage']['model_calls']==2 and restored['usage']['llm_calls']==3


@pytest.mark.parametrize('payload',[None,{'selected_id':'keep','explanation':'旧选择器','refusal':None},
                                   action('invent'),RuntimeError('provider unavailable')])
def test_invalid_or_unavailable_decision_fails_without_solver_fallback(payload):
    client=Script(payload)
    draft=inc.begin_run(make_snapshot(),request={'kind':'initial'},client=client)
    assert inc.step(draft,client=client)['step_status']=='failed'
    assert draft['step_log'][-1]['decision_mode']=='llm_failed'
    assert not draft['plan']['sleep_blocks']
    recovery=Script(action('read',**window(draft['snapshot'])))
    inc.resume_run(draft,client=recovery)
    assert inc.step(draft,client=recovery)['step_status']=='step'


def test_immutable_or_overlapping_edit_rejected_before_model():
    old=block('a',BASE_NOW-timedelta(hours=1),BASE_NOW+timedelta(hours=2))
    client=Script(action('edit',edits=[{'op':'remove','event_id':'a'}]),
                  action('edit',edits=[add(BASE_NOW+timedelta(hours=16)),add(BASE_NOW+timedelta(hours=17))]))
    draft=inc.begin_run(make_snapshot(active_plan={'sleep_blocks':[old]}),request={'kind':'initial'},client=client)
    before=draft['usage']['model_calls']
    run(draft,client)
    assert draft['usage']['model_calls']==before and draft['plan']['sleep_blocks']==[old]
    assert all(r['tool_result']['status']=='rejected' for r in draft['step_log'])


def test_clarification_answers_reach_agent_without_deleting_target():
    old=block('a',BASE_NOW+timedelta(hours=16),BASE_NOW+timedelta(hours=24),origin_event_id='evt1')
    snap=make_snapshot(active_plan={'sleep_blocks':[old]},extra_events=[dict(old,id='evt1',kind='sleep_plan')])
    client=Script(action('clarify',question='希望换到哪个时间段？'))
    draft=inc.begin_run(snap,request={'kind':'feedback','target_event_id':'evt1','meaning':'replace'},client=client)
    assert inc.step(draft,client=client)['step_status']=='needs_clarification'
    assert draft['plan']['sleep_blocks']==[old]
    inc.resume_run(draft,input={'kind':'answer','text':'明天晚一小时'},client=client)
    next_client=Script(action('read',**window(snap)))
    inc.step(draft,client=next_client)
    assert next_client.calls[0]['body']['clarification_answers'][-1]['text']=='明天晚一小时'


def test_actual_modelclient_offline_archive_replay_and_miss(tmp_path):
    client=ModelClient(tmp_path,online=False)
    snap=make_snapshot()
    draft=inc.begin_run(snap,request={'kind':'initial'},client=client,run_id='replay')
    duck=Script(action('read',**window(snap)))
    inc.step(deepcopy(draft),client=duck)
    call=duck.calls[0]
    request={'system':call['system'],'body':call['body'],'model':MODEL,'output_tokens':client.client.output_tokens}
    key=digest({'id':call['stage_id'],'request':request})
    write(tmp_path/'stages'/f'{key}.json',{'status':'complete','request_hash':digest(request),
                                        'parsed':action('read',**window(snap))})
    assert inc.step(draft,client=client)['step_status']=='step'
    assert inc.step(draft,client=client)['step_status']=='failed'


def test_submit_cannot_skip_reads_or_trial_evaluation():
    snap=make_snapshot()
    client=Script(action('submit',**window(snap),disposition='planned'),action('read',**window(snap)),
                  action('edit',edits=[add(BASE_NOW+timedelta(hours=16))]),action('retain'),
                  action('submit',**window(snap),disposition='planned'))
    draft=inc.begin_run(snap,request={'kind':'initial'},client=client)
    assert run(draft,client)['step_status']!='complete'
    assert not draft['plan']['sleep_blocks'] and draft['agent_state']['trial']
    assert draft['agent_state']['observation']['status']=='rejected'


def test_zero_model_budget_does_not_compute_baseline():
    draft=inc.begin_run(make_snapshot(),request={'kind':'initial'},client=Script(),budget={'max_model_calls':0})
    assert draft['status']=='budget_exhausted' and draft['usage']['model_calls']==0
    assert draft['baseline'] is None


def test_clock_rebase_does_not_keep_old_complete_or_old_trial():
    snap=make_snapshot(days=1)
    client=Script(action('read',**window(snap)),action('edit',edits=[add(BASE_NOW+timedelta(hours=16))]),
                  action('evaluate'),action('retain'),action('submit',**window(snap),disposition='planned'))
    draft=inc.begin_run(snap,request={'kind':'initial'},client=client)
    assert run(draft,client)['step_status']=='complete'
    later=dict(snap,now_utc=iso(BASE_NOW+timedelta(minutes=30)))
    next_client=Script(action('read',**window(later)))
    result=inc.step(draft,snapshot=later,client=next_client)
    assert result['step_status']=='step'
    assert draft['agent_state']['submissions']==[]
    assert draft['agent_state']['trial'] is None
    assert all(s['status']=='pending' for s in draft['segments'])


def test_generated_ids_skip_carried_ids_and_facts_changes_reject_draft():
    old=block('rec-0001',BASE_NOW+timedelta(hours=16),BASE_NOW+timedelta(hours=24))
    snap=make_snapshot(active_plan={'sleep_blocks':[old]})
    client=Script(action('edit',edits=[add(BASE_NOW+timedelta(hours=40))]),action('evaluate'),action('retain'))
    draft=inc.begin_run(snap,request={'kind':'initial'},client=client)
    run(draft,client)
    ids=[b['id'] for b in draft['plan']['sleep_blocks']]
    assert len(ids)==len(set(ids))==2
    changed=deepcopy(snap)
    changed['revision']+=1
    with pytest.raises(inc.StaleDraftError):
        inc.step(draft,snapshot=changed,client=Script())


def test_commitment_through_origin_event_protects_collateral_without_flag():
    old=block('s1',BASE_NOW+timedelta(hours=16),BASE_NOW+timedelta(hours=24),origin_event_id='evt1')
    snap=make_snapshot(active_plan={'sleep_blocks':[old]},extra_events=[
        dict(old,id='evt1',kind='sleep_plan'),
        {'id':'accept-1','kind':'commitment','locked':True,'target_event_id':'evt1'}])
    client=Script(action('edit',edits=[move('s1',BASE_NOW+timedelta(hours=17))]),
                  action('evaluate'),action('retain'))
    draft=inc.begin_run(snap,request={'kind':'initial'},client=client)
    assert run(draft,client)['step_status']=='needs_confirmation'
    assert draft['plan']['sleep_blocks']==[old]


@pytest.mark.parametrize('old_status',['complete','needs_confirmation','failed'])
def test_legacy_selector_state_is_rechecked_not_reused_as_agent_evidence(old_status):
    snap=make_snapshot()
    client=Script(action('read',**window(snap)))
    draft=inc.begin_run(snap,request={'kind':'initial'},client=client)
    draft['status']=old_status
    draft['provenance']['decision']['workflow_version']='incremental-workflow-llm-1'
    for s in draft['segments']:
        s['status']='processed'
    if old_status=='needs_confirmation':
        draft['pending_confirmations']=[{'confirmation_id':'legacy-card'}]
    assert inc.step(draft,client=client)['step_status']=='step'
    assert not draft['pending_confirmations']
    assert all(s['status']=='pending' for s in draft['segments'])


def test_unsafe_clarification_text_cannot_bypass_output_guard():
    client=Script(action('clarify',question='建议服用褪黑素应对进近'))
    draft=inc.begin_run(make_snapshot(),request={'kind':'initial'},client=client)
    assert inc.step(draft,client=client)['step_status']=='failed'
    assert draft['pending_clarification'] is None


def test_single_sleep_cannot_certify_forty_day_roster():
    snap=make_snapshot(days=40)
    client=Script(action('read',**window(snap)),action('edit',edits=[add(BASE_NOW+timedelta(hours=16))]),
                  action('evaluate'),action('retain'),action('submit',**window(snap),disposition='planned'))
    draft=inc.begin_run(snap,request={'kind':'initial'},client=client)
    assert run(draft,client)['step_status']!='complete'
    assert all(s['status']=='pending' for s in draft['segments'])
    assert not draft['agent_state']['submissions']
    issues=draft['agent_state']['planning_issues']
    assert len(issues)==40
    assert sum(i['status']=='unresolved' for i in issues)==39
    assert all(i['model_input_hash'] for i in issues)
    assert draft['agent_state']['observation']['status']=='incomplete'


def test_preserved_single_sleep_does_not_certify_remaining_month():
    old=block('one',BASE_NOW+timedelta(hours=16),BASE_NOW+timedelta(hours=24))
    snap=make_snapshot(days=40,active_plan={'sleep_blocks':[old]})
    client=Script(action('read',**window(snap)),action('submit',**window(snap),disposition='preserved'))
    draft=inc.begin_run(snap,request={'kind':'initial'},client=client)
    assert run(draft,client)['step_status']!='complete'
    assert any(i['status']=='preserved' for i in draft['agent_state']['planning_issues'])
    assert any(i['status']=='unresolved' for i in draft['agent_state']['planning_issues'])
    assert not draft['agent_state']['submissions']


def test_daytime_subrange_needs_no_sleep_if_evaluated_arrangement_explains_demand():
    old=block('night',BASE_NOW+timedelta(hours=16),BASE_NOW+timedelta(hours=24))
    snap=make_snapshot(days=1,active_plan={'sleep_blocks':[old]})
    morning={'start_utc':snap['now_utc'],'end_utc':iso(BASE_NOW+timedelta(hours=8))}
    client=Script(action('read',**morning),action('submit',**morning,disposition='preserved'))
    draft=inc.begin_run(snap,request={'kind':'initial'},client=client)
    run(draft,client)
    assert draft['agent_state']['observation']['status']=='submitted'
    assert draft['agent_state']['planning_issues'][0]['arrangement_ids']==['night']


def test_explicit_omit_explains_occurrence_without_forcing_daily_sleep():
    old=block('night',BASE_NOW+timedelta(hours=16),BASE_NOW+timedelta(hours=24),origin_event_id='evt1')
    snap=make_snapshot(days=1,active_plan={'sleep_blocks':[old]},extra_events=[dict(old,id='evt1',kind='sleep_plan')])
    client=Script(action('read',**window(snap)),action('edit',edits=[{'op':'remove','event_id':'night'}]),
                  action('evaluate'),action('retain'),action('submit',**window(snap),disposition='planned'))
    draft=inc.begin_run(snap,request={'kind':'feedback','target_event_id':'evt1','meaning':'omit_occurrence'},client=client)
    assert run(draft,client)['step_status']=='complete'
    assert not draft['plan']['sleep_blocks']
    assert draft['agent_state']['planning_issues'][0]['status']=='authorized_omit'


def test_no_environment_still_has_free_time_and_cannot_submit_no_window():
    snap=make_snapshot(days=2)
    snap['events']=[]
    client=Script(action('read',**window(snap)),action('submit',**window(snap),disposition='no_window'))
    draft=inc.begin_run(snap,request={'kind':'initial'},client=client)
    result=run(draft,client)
    assert result['step_status']=='continue'
    assert result['needs']['clarification'] is None
    assert all(s['status']=='pending' for s in draft['segments'])
    assert draft['agent_state']['observation']['status']=='rejected'
    assert 'known-free time exists' in draft['agent_state']['observation']['error']


def test_old_tool_completion_without_ledger_is_rechecked():
    snap=make_snapshot(days=2)
    client=Script(action('read',**window(snap)))
    draft=inc.begin_run(snap,request={'kind':'initial'},client=client)
    inc.step(draft,client=client)
    draft['agent_state']['version']='incremental-react-1'
    draft['agent_state']['submissions']=[window(snap)]
    draft['status']='complete'
    for s in draft['segments']:
        s['status']='processed'
    next_client=Script(action('read',**window(snap)))
    assert inc.step(draft,client=next_client)['step_status']=='step'
    assert not draft['agent_state']['submissions']
    assert all(s['status']=='pending' for s in draft['segments'])


def test_no_free_window_records_unsatisfied_demand_without_requesting_location():
    snap=make_snapshot(days=2,extra_events=[{
        'id':'fixed-flight','kind':'flight','protection':'fixed_external',
        'start_utc':iso(BASE_NOW),'end_utc':iso(BASE_NOW+timedelta(days=2)),'timezone':'UTC'}])
    client=Script(action('read',**window(snap)),action('submit',**window(snap),disposition='no_window'))
    draft=inc.begin_run(snap,request={'kind':'initial'},client=client)
    assert run(draft,client)['step_status']=='complete'
    assert all(s['status']=='processed' for s in draft['segments'])
    assert all(i['reason']=='constraints_leave_no_free_window' for i in draft['agent_state']['planning_issues'])

