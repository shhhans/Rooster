"""Authority-bound clarification, offline decisions and real sleep-model tools."""
from copy import deepcopy
from datetime import timedelta
import json
import pytest

from app.planning import incremental as inc
from tests.test_planning_incremental import make_snapshot, block, BASE_NOW
from tests.test_planning_incremental_llm import Script, action, add, move, window, run


def feedback_snapshot(*, requires=True, material=False, previous_status='active', reason=None):
    old=block('s1',BASE_NOW+timedelta(hours=16),BASE_NOW+timedelta(hours=24),origin_event_id='evt1')
    snap=make_snapshot(days=2,active_plan={'sleep_blocks':[old]},extra_events=[dict(old,id='evt1',kind='sleep_plan')])
    prior={'feedback_id':'feedback-first','target_event_id':'evt-root','target_revision':1,
           'meaning':'replace','reason':None,'status':previous_status,'source':'feedback_endpoint',
           'feedback_sequence':1,'occurrence_id':'evt-root',
           'applied_replacement':{'job_id':'previous-job','run_id':'previous-run',
               'before_event_id':'evt-root','after_event_id':'evt1','applied_at':'2026-01-15T05:00:00Z'}}
    current={'feedback_id':'feedback-second','target_event_id':'evt1','target_revision':2,
             'meaning':'replace','reason':reason,'status':'active','source':'feedback_endpoint',
             'feedback_sequence':2,'occurrence_id':'evt-root','applied_replacement':None}
    snap['feedback_context']={'schema_version':1,'current_feedback_id':'feedback-second',
        'target_event_id':'evt1','occurrence_id':'evt-root','requires_clarification':requires,
        'reason_code':'repeat_replacement_without_new_information' if requires else None,
        'has_material_new_reason':material,'history':[prior,current]}
    request={'kind':'feedback','target_event_id':'evt1','meaning':'replace','reason':reason}
    return snap,request


def test_second_rejection_pauses_before_provider_or_any_edit_and_is_stable():
    snap,request=feedback_snapshot()
    client=Script()  # would raise if any provider decision were attempted
    draft=inc.begin_run(snap,request=request,client=client)
    old=deepcopy(draft['plan'])
    first=inc.step(draft,client=client)
    second=inc.step(draft,client=client)
    assert first['step_status']==second['step_status']=='needs_clarification'
    assert first['needs']==second['needs']
    assert first['needs']['clarification']['question_id']=='repeat-rejection:feedback-second'
    assert not client.calls and draft['usage']['llm_calls']==0
    assert draft['plan']==old and not first['event_diffs']
    assert draft['usage']['model_calls']==1  # actual baseline only


def test_answer_resumes_same_run_and_real_evaluation_loop_after_json_restart():
    snap,request=feedback_snapshot()
    client=Script()
    draft=inc.begin_run(snap,request=request,client=client,run_id='repeat-resume')
    assert inc.step(draft,client=client)['step_status']=='needs_clarification'
    draft=json.loads(json.dumps(draft))
    inc.resume_run(draft,input={'kind':'answer','text':'希望凌晨零点睡，明晚原有时间保留'},client=client)
    assert draft['repeated_feedback_guard']['answered']
    client=Script(action('read',**window(snap)),
        action('edit',edits=[move('s1',BASE_NOW+timedelta(hours=18)),add(BASE_NOW+timedelta(hours=40))]),
        action('evaluate'),action('retain'),action('submit',**window(snap),disposition='planned'))
    result=run(draft,client)
    assert result['step_status']=='complete' and draft['run_id']=='repeat-resume'
    assert len(draft['plan']['sleep_blocks'])==2
    assert draft['usage']['model_calls']==2
    assert client.calls[0]['body']['clarification_answers'][-1]['question_id']=='repeat-rejection:feedback-second'
    assert client.calls[0]['body']['feedback_context']==snap['feedback_context']
    assert draft['snapshot']==snap


@pytest.mark.parametrize('variant',['first-replace','undone-previous','new-material-reason','no-authoritative-match'])
def test_authoritative_nonrepeat_context_does_not_trigger_gate(variant):
    snap,request=feedback_snapshot(requires=False)
    context=snap['feedback_context']
    if variant=='first-replace':
        context['history']=context['history'][1:]
    elif variant=='undone-previous':
        context['history'][0]['status']='undone'
    elif variant=='new-material-reason':
        context['has_material_new_reason']=True
        request['reason']='明早六点需要接家人，想提前完成休息'
        context['history'][-1]['reason']=request['reason']
    else:
        context.update(current_feedback_id=None,occurrence_id=None,history=[])
    client=Script(action('read',**window(snap)))
    draft=inc.begin_run(snap,request=request,client=client)
    assert inc.step(draft,client=client)['step_status']=='step'
    assert len(client.calls)==1 and not draft.get('repeated_feedback_guard')


@pytest.mark.parametrize('change',['new-feedback','undo-history','material-reason'])
def test_history_changes_make_old_answer_draft_stale(change):
    snap,request=feedback_snapshot()
    draft=inc.begin_run(snap,request=request,client=Script())
    inc.step(draft,client=Script())
    changed=deepcopy(snap)
    context=changed['feedback_context']
    if change=='new-feedback':
        context['current_feedback_id']='feedback-third'
        context['history'][-1]['feedback_id']='feedback-third'
    elif change=='undo-history':
        context['history'][0]['status']='undone'
        context['requires_clarification']=False
    else:
        context['requires_clarification']=False
        context['has_material_new_reason']=True
    with pytest.raises(inc.StaleDraftError):
        inc.resume_run(draft,snapshot=changed,input={'kind':'answer','text':'旧问题答案'},client=Script())
    assert not draft['repeated_feedback_guard']['answered']


def test_unrelated_old_question_answer_cannot_release_current_feedback_gate():
    snap,request=feedback_snapshot()
    draft=inc.begin_run(snap,request=request,client=Script())
    draft['pending_clarification']={'question_id':'agent-question-old','question':'旧问题'}
    inc.resume_run(draft,input={'kind':'answer','text':'旧问题的答案'},client=Script())
    result=inc.step(draft,client=Script())
    assert result['step_status']=='needs_clarification'
    assert result['needs']['clarification']['question_id']=='repeat-rejection:feedback-second'
    assert not draft['repeated_feedback_guard']['answered']
    assert not result['event_diffs']


@pytest.mark.parametrize('broken',['missing-publication','undone-current','material-but-required','wrong-target'])
def test_inconsistent_authority_evidence_is_rejected_before_provider(broken):
    snap,request=feedback_snapshot()
    context=snap['feedback_context']
    if broken=='missing-publication':
        context['history'][0]['applied_replacement']=None
    elif broken=='undone-current':
        context['history'][-1]['status']='undone'
    elif broken=='material-but-required':
        context['has_material_new_reason']=True
    else:
        context['target_event_id']='some-other-event'
        context['history'][-1]['target_event_id']='some-other-event'
    client=Script()
    with pytest.raises(inc.InvalidSnapshotError):
        inc.begin_run(snap,request=request,client=client)
    assert not client.calls


@pytest.mark.parametrize('field,value',[
    ('schema_version',True),('schema_version',2),('history',{}),
    ('requires_clarification','yes'),('has_material_new_reason',1),('current_feedback_id',9)])
def test_incorrect_context_version_and_types_fail_closed(field,value):
    snap,request=feedback_snapshot()
    snap['feedback_context'][field]=value
    with pytest.raises(inc.InvalidSnapshotError):
        inc.begin_run(snap,request=request,client=Script())
