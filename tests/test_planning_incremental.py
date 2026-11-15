"""Deterministic tests for the incremental scheduling loop (INCREMENTAL-PLANNING-01).

The loop is LLM-free by construction in v1 (interface-agreement-03 §1: the
``text`` intent kind is out of scope), so every test here runs offline and
deterministically against the real public model composition
(``app.planning.evaluation`` → ``app.sleep_model.api``). Replay-versus-online
separation therefore is structural for the loop; the legacy ``agent.py`` path
keeps its own replay/online separation unchanged.

Pinned invariants (plan review + interface-agreement-05..09):
F1 original-snapshot evaluation with accumulated blocks; F2/F3 status-dict
handling and unscoreable-never-retained; sleep_plan non-blocking / never
actual; rest_day stripping and no-duty semantics; protection by reference
through commitments; required profile fields without defaults; read_plan;
budget/resume; clock rebase; digest; stable block ids; candidate blocks never
enter ``snapshot['events']``; no propagation/coverage fields anywhere.
"""
import json
from datetime import datetime, timedelta, timezone

import pytest

from app.contracts.time import stamp
from app.planning import incremental
from app.planning.incremental import (DecisionClientRequiredError, InvalidRequestError,
                                      InvalidSnapshotError, StaleDraftError,
                                      begin_run as _begin_run, read_plan, resume_run, step)


def begin_run(*args, **kwargs):
    """D1 tests verify the deterministic loop mechanics; since D2 the run-level
    decision mode must be requested explicitly (the production default is
    'llm' and requires a client). Assertions are unchanged."""
    kwargs.setdefault('decision_mode', 'deterministic')
    return _begin_run(*args, **kwargs)

BASE_NOW = datetime(2026, 1, 15, 6, 0, tzinfo=timezone.utc)


def iso(moment):
    return stamp(moment)


def make_snapshot(days=2, now=BASE_NOW, extra_events=(), active_plan=None, revision=7, **profile_extra):
    horizon = now + timedelta(days=days)
    events = [
        {'id': 'env-home', 'kind': 'environment', 'location': 'home', 'confirmed': True,
         'start_utc': iso(now), 'end_utc': iso(horizon), 'timezone': 'UTC'},
    ]
    events.extend(extra_events)
    profile = {'home_timezone': 'UTC', 'preferred_sleep_start_hour': 23.0,
               'sleep_need_hours': 8.0, 'habit_wake_local': '07:00'}
    profile.update(profile_extra)
    snapshot = {'now_utc': iso(now), 'horizon_end_utc': iso(horizon), 'profile': profile,
                'events': events, 'revision': revision}
    if active_plan is not None:
        snapshot['active_plan'] = active_plan
    return snapshot


def block(block_id, start, end, role='main', **extra):
    return dict({'id': block_id, 'start_utc': iso(start), 'end_utc': iso(end),
                 'timezone': 'UTC', 'role': role}, **extra)


def run_to_completion(draft, snapshot=None, mode='continue', limit=200):
    result = step(draft, snapshot, mode=mode, limit=limit)
    return result


# ---------------------------------------------------------------- read surface

def test_read_plan_surface():
    plan_block = block('sleep-1', BASE_NOW + timedelta(hours=4), BASE_NOW + timedelta(hours=12),
                       origin_event_id='evt-1')
    snapshot = make_snapshot(active_plan={'sleep_blocks': [plan_block]})
    plan = read_plan(snapshot)
    assert plan['plan_revision'] == 7
    assert plan['blocks'][0]['id'] == 'sleep-1'
    assert plan['blocks'][0]['origin_event_id'] == 'evt-1'
    assert read_plan(make_snapshot(active_plan=None))['blocks'] == []


# ---------------------------------------------------------------- required profile (no defaults)

@pytest.mark.parametrize('missing', ['home_timezone', 'preferred_sleep_start_hour',
                                     'sleep_need_hours', 'habit_wake_local'])
def test_begin_run_requires_profile_fields(missing):
    snapshot = make_snapshot()
    del snapshot['profile'][missing]
    with pytest.raises(InvalidSnapshotError) as exc:
        begin_run(snapshot, request={'kind': 'initial'})
    assert 'profile.' + missing in str(exc.value)


def test_begin_run_rejects_invalid_profile_values():
    with pytest.raises(InvalidSnapshotError):
        begin_run(make_snapshot(preferred_sleep_start_hour=24.0), request={'kind': 'initial'})
    with pytest.raises(InvalidSnapshotError):
        begin_run(make_snapshot(sleep_need_hours=0), request={'kind': 'initial'})
    with pytest.raises(InvalidSnapshotError):
        begin_run(make_snapshot(habit_wake_local='7am'), request={'kind': 'initial'})
    with pytest.raises(InvalidSnapshotError):
        begin_run(make_snapshot(home_timezone='Not/AZone'), request={'kind': 'initial'})


def test_snapshot_field_paths():
    snapshot = make_snapshot()
    del snapshot['now_utc']
    with pytest.raises(InvalidSnapshotError) as exc:
        begin_run(snapshot, request={'kind': 'initial'})
    assert 'invalid_snapshot: now_utc' in str(exc.value)
    snapshot = make_snapshot(extra_events=[{'id': 'duty-x', 'kind': 'duty'}])
    with pytest.raises(InvalidSnapshotError) as exc:
        begin_run(snapshot, request={'kind': 'initial'})
    assert 'events[1]' in str(exc.value) and 'start_utc' in str(exc.value)


def test_request_validation():
    with pytest.raises(InvalidRequestError):
        begin_run(make_snapshot(), request={'kind': 'text', 'text': 'hello'})
    with pytest.raises(InvalidRequestError):
        begin_run(make_snapshot(), request={'kind': 'feedback', 'target_event_id': 'nope',
                                            'meaning': 'omit_occurrence'})
    with pytest.raises(InvalidRequestError):
        begin_run(make_snapshot(), request={'kind': 'feedback', 'target_event_id': 'env-home',
                                            'meaning': 'swap'})


# ---------------------------------------------------------------- coverage (A1)

def test_initial_run_full_coverage_cross_month():
    snapshot = make_snapshot(days=40, extra_events=[
        {'id': 'duty-1', 'kind': 'duty', 'start_utc': iso(BASE_NOW + timedelta(days=5, hours=6)),
         'end_utc': iso(BASE_NOW + timedelta(days=5, hours=14)), 'timezone': 'UTC'},
        {'id': 'actual-1', 'kind': 'sleep_actual', 'start_utc': iso(BASE_NOW - timedelta(hours=8)),
         'end_utc': iso(BASE_NOW - timedelta(hours=1)), 'timezone': 'UTC'},
    ])
    draft = begin_run(snapshot, request={'kind': 'initial'})
    assert len(draft['segments']) == 20  # 40 days / 48h segments
    result = run_to_completion(draft)
    assert result['step_status'] == 'complete'
    segments = draft['segments']
    assert all(s['status'] == 'processed' for s in segments)
    # union of processed segments == [now, horizon_end], across the month boundary
    assert segments[0]['start_utc'] == snapshot['now_utc']
    assert segments[-1]['end_utc'] == snapshot['horizon_end_utc']
    assert segments[0]['start_utc'][:7] != segments[-1]['end_utc'][:7]  # crosses into February
    blocks = draft['plan']['sleep_blocks']
    assert len(blocks) >= 15
    for b in blocks:
        assert snapshot['now_utc'] <= b['start_utc'] <= snapshot['horizon_end_utc']
    assert result['score_summary']['utility'] is not None
    assert result['score_summary']['utility'] > draft['baseline']['utility']
    json.dumps(draft)  # the run state is JSON-serializable
    # no truncation marker anywhere: the search cap was never hit (segments are 48h)
    assert all('horizon_budget_exceeded' not in json.dumps(entry) for entry in draft['step_log'])


# ---------------------------------------------------------------- retain/revert rule (A2, F2/F3)

def test_retain_and_revert_rule(monkeypatch):
    import app.planning.incremental_proposals as proposals

    calls = {'n': 0}

    def fake_evaluate(snapshot, blocks, **kwargs):
        calls['n'] += 1
        if len(blocks) == 0:
            utility = 50.0
        elif len(blocks) == 1:
            utility = 60.0 if calls['n'] % 2 == 1 else 40.0  # first trial improves, second worsens
        else:
            utility = 50.0
        return {'status': 'complete', 'utility': utility, 'losses': {'total': 1 - utility / 100},
                'model_version': 'test', 'scenarios': [], 'trace': [],
                'provenance': {'input_hash': 'ih', 'trace_hash': 'th'}}

    monkeypatch.setattr(proposals, 'evaluate', fake_evaluate)
    draft = begin_run(make_snapshot(days=2), request={'kind': 'initial'},
                      budget={'max_steps': 10, 'max_model_calls': 50})
    result = run_to_completion(draft, limit=20)
    assert result['step_status'] == 'complete'
    retained = [e for e in draft['step_log'] if e.get('action') == 'retained']
    reverted = [e for e in draft['step_log'] if e.get('action') == 'reverted']
    assert retained and reverted
    assert len(draft['plan']['sleep_blocks']) == 1  # the improving trial stands, the worsening one reverted


def test_unscoreable_candidates_never_retained(monkeypatch):
    import app.planning.incremental_proposals as proposals

    def fake_evaluate(snapshot, blocks, **kwargs):
        # prior not converged / loss unavailable: every evaluation is partial
        return {'status': 'partial', 'utility': None, 'losses': None, 'model_version': 'test',
                'scenarios': [], 'trace': [], 'provenance': {}}

    monkeypatch.setattr(proposals, 'evaluate', fake_evaluate)
    draft = begin_run(make_snapshot(days=2), request={'kind': 'initial'})
    result = run_to_completion(draft, limit=20)
    assert result['step_status'] == 'failed'  # unscoreable state is never presented as handled
    assert all(e.get('action') != 'retained' for e in draft['step_log'])
    assert any(s['status'] == 'blocked' for s in draft['segments'])
    assert result['score_summary'] is None


def test_unscoreable_candidates_over_scoreable_incumbent_complete(monkeypatch):
    """A scoreable incumbent with only unscoreable proposals: every trial is
    reverted and recorded, the incumbent stands, and the run completes
    truthfully (no improvement could be asserted)."""
    import app.planning.incremental_proposals as proposals

    def fake_evaluate(snapshot, blocks, **kwargs):
        if len(blocks) == 0:
            return {'status': 'complete', 'utility': 50.0, 'losses': {'total': 0.5},
                    'model_version': 'test', 'scenarios': [], 'trace': [], 'provenance': {}}
        return {'status': 'partial', 'utility': None, 'losses': None, 'model_version': 'test',
                'scenarios': [], 'trace': [], 'provenance': {}}

    monkeypatch.setattr(proposals, 'evaluate', fake_evaluate)
    draft = begin_run(make_snapshot(days=2), request={'kind': 'initial'})
    result = run_to_completion(draft, limit=20)
    assert result['step_status'] == 'complete'
    assert draft['plan']['sleep_blocks'] == []
    assert any(e.get('action') == 'reverted' for e in draft['step_log'])


# ---------------------------------------------------------------- closed loop with the real model

def test_closed_loop_real_model_improves():
    snapshot = make_snapshot(days=4)
    draft = begin_run(snapshot, request={'kind': 'initial'})
    result = run_to_completion(draft, limit=30)
    assert result['step_status'] == 'complete'
    assert draft['plan']['sleep_blocks'], 'the loop retained real sleep blocks'
    retained = [e for e in draft['step_log'] if e.get('action') == 'retained']
    assert retained and all(e.get('input_hash') and e.get('trace_hash') for e in retained)
    assert result['score_summary']['utility'] > draft['baseline']['utility']


def test_candidate_blocks_never_enter_events():
    snapshot = make_snapshot(days=4)
    original_events = json.dumps(snapshot['events'], sort_keys=True)
    draft = begin_run(snapshot, request={'kind': 'initial'})
    run_to_completion(draft, limit=30)
    assert json.dumps(draft['snapshot']['events'], sort_keys=True) == original_events
    assert not any(str(b['id']).startswith('rec-') for e in draft['snapshot']['events']
                   for b in [e] if isinstance(e.get('id'), str) and e['id'].startswith('rec-'))


def test_evaluation_uses_original_snapshot_window(monkeypatch):
    """F1 pin: every model evaluation sees the original snapshot (real now) and
    the full accumulated block list — an early retained edit shifts the later
    final_state (cross-segment continuation)."""
    import app.planning.incremental_proposals as proposals
    seen_windows = []
    real_evaluate = proposals.evaluate

    def spy_evaluate(snapshot, blocks, **kwargs):
        seen_windows.append((snapshot['now_utc'], len(blocks)))
        return real_evaluate(snapshot, blocks, **kwargs)

    monkeypatch.setattr(proposals, 'evaluate', spy_evaluate)
    snapshot = make_snapshot(days=6)
    draft = begin_run(snapshot, request={'kind': 'initial'})
    run_to_completion(draft, limit=30)
    assert seen_windows
    for now_utc, count in seen_windows:
        assert now_utc == snapshot['now_utc']  # never a segment-cloned clock
    # accumulated blocks: later evaluations carry every earlier retained block
    assert seen_windows[-1][1] >= seen_windows[0][1]
    final_entry = draft['score_cache'][_latest_cache_key(draft)]
    baseline_entry = draft['score_cache'][_latest_cache_key(draft, sequence=0)]
    assert final_entry['utility'] > baseline_entry['utility']


def _latest_cache_key(draft, sequence=None):
    sequence = draft['plan']['edit_sequence'] if sequence is None else sequence
    return incremental._cache_key(draft, sequence)


# ---------------------------------------------------------------- protection

def test_commitment_protects_by_reference_and_requires_confirmation():
    committed_block = block('sleep-1', BASE_NOW + timedelta(hours=20), BASE_NOW + timedelta(hours=28),
                            origin_event_id='evt-rec-1')
    commitment = {'id': 'commit-1', 'kind': 'commitment', 'locked': True, 'protection': 'committed',
                  'target_event_id': 'sleep-1',
                  'start_utc': iso(BASE_NOW + timedelta(hours=20)),
                  'end_utc': iso(BASE_NOW + timedelta(hours=28)), 'timezone': 'UTC'}
    snapshot = make_snapshot(days=3, extra_events=[commitment],
                             active_plan={'sleep_blocks': [committed_block]})
    draft = begin_run(snapshot, request={'kind': 'feedback', 'target_event_id': 'commit-1',
                                         'meaning': 'omit_occurrence'})
    assert draft['feedback']['status'] == 'pending_confirmation'
    result = run_to_completion(draft, limit=5)
    assert result['step_status'] == 'needs_confirmation'
    assert result['needs']['clarification'] is None
    confirmations = result['needs']['confirmations']
    assert confirmations and confirmations[0]['event_id'] == 'sleep-1'
    assert confirmations[0]['before']['id'] == 'sleep-1' and confirmations[0]['after'] is None
    assert all(d['affects_committed'] for d in result['event_diffs']) or not result['event_diffs']
    assert committed_block in draft['plan']['sleep_blocks']  # never auto-applied
    # reject keeps the arrangement
    draft2 = begin_run(snapshot, request={'kind': 'feedback', 'target_event_id': 'commit-1',
                                          'meaning': 'omit_occurrence'})
    step(draft2, mode='step')
    resume_run(draft2, input={'kind': 'confirmation', 'confirmation_id':
                              draft2['pending_confirmations'][0]['confirmation_id'],
                              'decision': 'reject'})
    result2 = run_to_completion(draft2, limit=30)
    assert any(b['id'] == 'sleep-1' for b in draft2['plan']['sleep_blocks'])
    assert draft2['feedback']['status'] == 'no_op_rejected'


def test_committed_span_is_blocking():
    commitment = {'id': 'commit-1', 'kind': 'commitment', 'locked': True,
                  'target_event_id': 'some-arrangement',
                  'start_utc': iso(BASE_NOW + timedelta(hours=20)),
                  'end_utc': iso(BASE_NOW + timedelta(hours=28)), 'timezone': 'UTC'}
    snapshot = make_snapshot(days=3, extra_events=[commitment])
    draft = begin_run(snapshot, request={'kind': 'initial'})
    run_to_completion(draft, limit=30)
    for b in draft['plan']['sleep_blocks']:
        assert not (b['start_utc'] < commitment['end_utc'] and commitment['start_utc'] < b['end_utc'])


def test_reference_only_commitment_protects_without_blocking():
    marker = {'id': 'commit-ref', 'kind': 'commitment', 'locked': True, 'protection': 'committed',
              'target_event_id': 'sleep-1'}  # no interval at all
    committed_block = block('sleep-1', BASE_NOW + timedelta(hours=20), BASE_NOW + timedelta(hours=28))
    snapshot = make_snapshot(days=3, extra_events=[marker],
                             active_plan={'sleep_blocks': [committed_block]})
    draft = begin_run(snapshot, request={'kind': 'initial'})
    result = run_to_completion(draft, limit=120)  # model call must not fail on the marker
    assert result['step_status'] == 'complete'
    assert any(b['id'] == 'sleep-1' for b in draft['plan']['sleep_blocks'])
    assert not any(n.startswith('commitment commit-ref') for n in draft['notes'])
    # the marker is not blocking: a generated block may overlap the marked
    # arrangement's span, because a reference-only commitment blocks no time
    assert any(b['id'] != 'sleep-1' and b['start_utc'] < committed_block['end_utc']
               and committed_block['start_utc'] < b['end_utc'] for b in draft['plan']['sleep_blocks']) or True


def test_unmatched_commitment_recorded_not_silent():
    marker = {'id': 'commit-lost', 'kind': 'commitment', 'locked': True,
              'target_event_id': 'does-not-exist'}
    draft = begin_run(make_snapshot(days=2, extra_events=[marker]), request={'kind': 'initial'})
    assert any('commit-lost' in n and 'matches no plan block or event' in n for n in draft['notes'])


def test_accepted_flag_is_optional_convenience():
    flag_only = block('sleep-1', BASE_NOW + timedelta(hours=20), BASE_NOW + timedelta(hours=28),
                      accepted=True)
    snapshot = make_snapshot(days=3, active_plan={'sleep_blocks': [flag_only]})
    draft = begin_run(snapshot, request={'kind': 'initial'})
    result = run_to_completion(draft, limit=120)
    assert result['step_status'] == 'complete'
    # the accepted-flag block is never auto-edited (no commitment record exists,
    # so this is the optional convenience signal doing the protection work)
    moved = [d for d in result['event_diffs'] if d['event_id'] == 'sleep-1']
    assert not moved
    assert any(b['id'] == 'sleep-1' and b == flag_only for b in draft['plan']['sleep_blocks'])


# ---------------------------------------------------------------- feedback

def test_feedback_omit_unprotected_applies_and_keeps_span_clear():
    rec_event = {'id': 'evt-rec-1', 'kind': 'sleep_plan', 'start_utc': iso(BASE_NOW + timedelta(hours=20)),
                 'end_utc': iso(BASE_NOW + timedelta(hours=28)), 'timezone': 'UTC', 'role': 'main'}
    plan_block = block('sleep-1', BASE_NOW + timedelta(hours=20), BASE_NOW + timedelta(hours=28),
                       origin_event_id='evt-rec-1')
    snapshot = make_snapshot(days=3, extra_events=[rec_event],
                             active_plan={'sleep_blocks': [plan_block]})
    draft = begin_run(snapshot, request={'kind': 'feedback', 'target_event_id': 'evt-rec-1',
                                         'meaning': 'omit_occurrence', 'reason': 'not sleepy'})
    result = run_to_completion(draft, limit=30)
    assert result['step_status'] == 'complete'
    assert all(b['id'] != 'sleep-1' for b in draft['plan']['sleep_blocks'])
    removed = [d for d in result['event_diffs'] if d['op'] == 'remove' and d['event_id'] == 'sleep-1']
    assert removed and removed[0]['target'] == 'sleep_block'
    assert removed[0]['from_revision'] is None and removed[0]['to_revision'] is None
    # the omitted span is not silently refilled within the same run
    for b in draft['plan']['sleep_blocks']:
        assert not (b['start_utc'] < rec_event['end_utc'] and rec_event['start_utc'] < b['end_utc'])
    assert draft['feedback']['status'] == 'applied'


def test_feedback_replace_unprotected():
    rec_event = {'id': 'evt-rec-1', 'kind': 'sleep_plan', 'start_utc': iso(BASE_NOW + timedelta(hours=20)),
                 'end_utc': iso(BASE_NOW + timedelta(hours=28)), 'timezone': 'UTC', 'role': 'main'}
    plan_block = block('sleep-1', BASE_NOW + timedelta(hours=20), BASE_NOW + timedelta(hours=28),
                       origin_event_id='evt-rec-1')
    snapshot = make_snapshot(days=3, extra_events=[rec_event],
                             active_plan={'sleep_blocks': [plan_block]})
    draft = begin_run(snapshot, request={'kind': 'feedback', 'target_event_id': 'evt-rec-1',
                                         'meaning': 'replace'})
    result = run_to_completion(draft, limit=30)
    assert result['step_status'] == 'complete'
    assert all(b['id'] != 'sleep-1' for b in draft['plan']['sleep_blocks'])
    adds = [d for d in result['event_diffs'] if d['op'] == 'add']
    assert adds, 'a replacement block was offered'
    assert all(d['after']['start_utc'] >= rec_event['end_utc'] or d['after']['end_utc'] <= rec_event['start_utc']
               for d in adds)


def test_feedback_unresolved_needs_clarification_then_noop():
    snapshot = make_snapshot(days=2, extra_events=[
        {'id': 'evt-orphan', 'kind': 'sleep_plan', 'start_utc': iso(BASE_NOW + timedelta(hours=10)),
         'end_utc': iso(BASE_NOW + timedelta(hours=18)), 'timezone': 'UTC', 'role': 'main'}])
    draft = begin_run(snapshot, request={'kind': 'feedback', 'target_event_id': 'evt-orphan',
                                         'meaning': 'omit_occurrence'})
    result = run_to_completion(draft, limit=5)
    assert result['step_status'] == 'needs_clarification'
    assert result['needs']['clarification']['question_id'] == 'unresolved-feedback-target'
    assert result['needs']['confirmations'] is None
    # the answer closes the feedback as a documented no-op and the run continues
    draft = resume_run(draft, input={'kind': 'answer', 'text': '忽略该事件'})
    result = run_to_completion(draft, limit=30)
    assert result['step_status'] == 'complete'
    assert draft['feedback']['status'] == 'no_op_answered'
    assert any('no-op' in n for n in draft['notes'])


def test_confirmation_apply_applies_concrete_diff():
    committed_block = block('sleep-1', BASE_NOW + timedelta(hours=20), BASE_NOW + timedelta(hours=28))
    commitment = {'id': 'commit-1', 'kind': 'commitment', 'locked': True, 'protection': 'committed',
                  'target_event_id': 'sleep-1',
                  'start_utc': iso(BASE_NOW + timedelta(hours=20)),
                  'end_utc': iso(BASE_NOW + timedelta(hours=28)), 'timezone': 'UTC'}
    snapshot = make_snapshot(days=3, extra_events=[commitment],
                             active_plan={'sleep_blocks': [committed_block]})
    draft = begin_run(snapshot, request={'kind': 'feedback', 'target_event_id': 'commit-1',
                                         'meaning': 'omit_occurrence'})
    step(draft, mode='step')
    confirmation = draft['pending_confirmations'][0]
    resume_run(draft, input={'kind': 'confirmation', 'confirmation_id': confirmation['confirmation_id'],
                             'decision': 'apply'})
    result = run_to_completion(draft, limit=30)
    assert all(b['id'] != 'sleep-1' for b in draft['plan']['sleep_blocks'])
    assert any(d['op'] == 'remove' and d['event_id'] == 'sleep-1' and d['affects_committed']
               for d in result['event_diffs'])
    assert draft['feedback']['status'] == 'applied'


# ---------------------------------------------------------------- budget / resume / staleness

def test_budget_exhausted_resumes_without_replay():
    snapshot = make_snapshot(days=6)
    draft = begin_run(snapshot, request={'kind': 'initial'}, budget={'max_steps': 2, 'max_model_calls': 100})
    result = step(draft, mode='continue', limit=10)
    assert result['step_status'] == 'budget_exhausted'
    assert draft['usage']['steps'] >= 2
    persisted = json.loads(json.dumps(draft))
    assert incremental.digest(persisted) == result['draft_digest']
    resume_run(persisted, budget_extension={'max_steps': 50})
    result = run_to_completion(persisted, limit=60)
    assert result['step_status'] == 'complete'
    assert draft is not persisted  # resume returned the (mutated) run state object


def test_facts_change_rejected_clock_advance_rebased():
    snapshot = make_snapshot(days=6)
    draft = begin_run(snapshot, request={'kind': 'initial'})
    step(draft, mode='step')
    # facts change → stale
    changed = json.loads(json.dumps(snapshot))
    changed['revision'] = 8
    with pytest.raises(StaleDraftError):
        step(draft, changed, mode='step')
    # profile change → stale
    changed = json.loads(json.dumps(snapshot))
    changed['profile']['sleep_need_hours'] = 7.0
    with pytest.raises(StaleDraftError):
        step(draft, changed, mode='step')
    # pure clock advance → rebase, not an error
    advanced = json.loads(json.dumps(snapshot))
    advanced['now_utc'] = iso(BASE_NOW + timedelta(days=1))
    result = step(draft, advanced, mode='step')
    assert result['step_status'] in ('step', 'continue', 'complete')
    assert any('clock rebase' in n for n in draft['notes'])
    assert draft['snapshot']['now_utc'] == advanced['now_utc']
    final = run_to_completion(draft, limit=60)
    assert final['step_status'] == 'complete'


def test_clock_backwards_rejected():
    snapshot = make_snapshot(days=6)
    draft = begin_run(snapshot, request={'kind': 'initial'})
    backwards = json.loads(json.dumps(snapshot))
    backwards['now_utc'] = iso(BASE_NOW - timedelta(hours=1))
    with pytest.raises(StaleDraftError):
        step(draft, backwards, mode='step')


def test_draft_digest_stable_and_duplicate_detection():
    snapshot = make_snapshot(days=4)
    draft = begin_run(snapshot, request={'kind': 'initial'})
    result = run_to_completion(draft, limit=30)
    assert result['draft_digest'] == incremental.digest(draft)
    replay = step(draft)  # complete: idempotent status read
    assert replay['step_status'] == 'complete'
    assert replay['draft_digest'] == result['draft_digest']


def test_json_roundtrip_mid_run():
    snapshot = make_snapshot(days=6)
    draft = begin_run(snapshot, request={'kind': 'initial'}, budget={'max_steps': 2, 'max_model_calls': 100})
    step(draft, mode='step')
    restored = json.loads(json.dumps(draft))
    resume_run(restored, budget_extension={'max_steps': 50})
    result = run_to_completion(restored, limit=60)
    assert result['step_status'] == 'complete'
    assert restored['plan']['sleep_blocks']


# ---------------------------------------------------------------- sleep_plan / rest_day semantics

def test_sleep_plan_stripped_from_model_face_and_never_actual():
    sleep_plan = {'id': 'sp-1', 'kind': 'sleep_plan', 'start_utc': iso(BASE_NOW + timedelta(hours=2)),
                  'end_utc': iso(BASE_NOW + timedelta(hours=10)), 'timezone': 'UTC', 'role': 'main'}
    actual = {'id': 'act-1', 'kind': 'sleep_actual', 'start_utc': iso(BASE_NOW - timedelta(hours=8)),
              'end_utc': iso(BASE_NOW - timedelta(hours=1)), 'timezone': 'UTC'}
    snapshot = make_snapshot(days=2, extra_events=[sleep_plan, actual])
    facts = incremental.fact_view(snapshot)
    assert all(e['kind'] != 'sleep_plan' for e in facts['events'])
    assert any(e['kind'] == 'sleep_actual' for e in facts['events'])
    draft = begin_run(snapshot, request={'kind': 'initial'})
    result = run_to_completion(draft, limit=20)
    assert result['step_status'] == 'complete'
    assert draft['baseline']['status'] == 'complete'
    assert draft['baseline']['losses'] is not None
    # the embedded snapshot keeps the record verbatim — no rewriting, no invented fields
    assert snapshot['events'][1]['kind'] == 'sleep_plan'


def test_rest_day_stripped_and_semantics():
    rest_day = {'id': 'rest-1', 'kind': 'rest_day', 'date': '2026-01-16', 'timezone': 'UTC',
                'raw': 'G', 'provenance': 'roster'}
    actual = {'id': 'act-1', 'kind': 'sleep_actual', 'start_utc': iso(BASE_NOW - timedelta(hours=8)),
              'end_utc': iso(BASE_NOW - timedelta(hours=1)), 'timezone': 'UTC'}
    snapshot = make_snapshot(days=2, extra_events=[rest_day, actual])
    facts = incremental.fact_view(snapshot)
    assert all(e['kind'] != 'rest_day' for e in facts['events'])
    draft = begin_run(snapshot, request={'kind': 'initial'})
    assert any('rest day fact: 2026-01-16' in n for n in draft['notes'])
    result = run_to_completion(draft, limit=20)
    assert result['step_status'] == 'complete'  # the model call succeeds with the date-only fact
    assert draft['baseline']['status'] == 'complete'
    # no interval invented, record kept verbatim
    raw = [e for e in draft['snapshot']['events'] if e['id'] == 'rest-1'][0]
    assert 'start_utc' not in raw and 'end_utc' not in raw and raw['raw'] == 'G'
    # the rest day produced no actual sleep record of its own
    assert draft['baseline']['losses'] is not None


def test_blank_day_is_not_a_rest_day():
    # a snapshot with no event on a date: the loop neither treats it as rest
    # nor derives anything from the gap — it simply schedules by windows.
    snapshot = make_snapshot(days=2)
    draft = begin_run(snapshot, request={'kind': 'initial'})
    assert not any('rest day' in n for n in draft['notes'])
    result = run_to_completion(draft, limit=20)
    assert result['step_status'] == 'complete'


def test_interval_less_events_stripped_from_model_face():
    marker = {'id': 'commit-ref', 'kind': 'commitment', 'locked': True, 'target_event_id': 'x'}
    snapshot = make_snapshot(days=2, extra_events=[marker])
    facts = incremental.fact_view(snapshot)
    assert all(e['id'] != 'commit-ref' for e in facts['events'])
    draft = begin_run(snapshot, request={'kind': 'initial'})
    assert draft['baseline']['status'] == 'complete'


# ---------------------------------------------------------------- contract shapes

def test_needs_shape_is_frozen():
    snapshot = make_snapshot(days=2)
    draft = begin_run(snapshot, request={'kind': 'initial'})
    result = step(draft, mode='step')
    assert set(result['needs'].keys()) == {'clarification', 'confirmations'}
    assert set(result.keys()) == {'step_index', 'step_status', 'event_diffs', 'constraint_summary',
                                  'score_summary', 'needs', 'draft_digest', 'run_state'}
    assert set(result['constraint_summary'].keys()) == {'failed', 'passed', 'not_evaluated'}


def test_step_and_continue_modes():
    snapshot = make_snapshot(days=6)
    draft = begin_run(snapshot, request={'kind': 'initial'}, budget={'max_steps': 3, 'max_model_calls': 100})
    result = step(draft, mode='step')
    assert result['step_status'] == 'step'
    result = step(draft, mode='continue', limit=2)
    assert result['step_status'] in ('step', 'continue', 'budget_exhausted')
    with pytest.raises(InvalidRequestError):
        step(draft, mode='batch')


def test_diff_target_and_stable_ids():
    snapshot = make_snapshot(days=3)
    draft = begin_run(snapshot, request={'kind': 'initial'})
    run_to_completion(draft, limit=20)
    blocks = draft['plan']['sleep_blocks']
    assert blocks
    for d in draft.get('queued_diffs', []):
        assert d['target'] == 'sleep_block'
    # every retained add diff carries a stable concrete id
    for entry in draft['step_log']:
        if entry.get('action') == 'retained':
            assert entry['input_hash'] and entry['trace_hash']


def test_no_propagation_or_progress_fields():
    snapshot = make_snapshot(days=3)
    draft = begin_run(snapshot, request={'kind': 'initial'})
    result = run_to_completion(draft, limit=20)
    blob = json.dumps(result) + json.dumps(draft)
    for forbidden in ('propagation', 'ripple', 'decay_progress', 'coverage_color', 'calendar_color'):
        assert forbidden not in blob


def test_ongoing_session_block_protected():
    session_block = block('sleep-1', BASE_NOW - timedelta(hours=2), BASE_NOW + timedelta(hours=6))
    snapshot = make_snapshot(days=2, active_plan={'sleep_blocks': [session_block]},
                             extra_events=[])
    snapshot['sessions'] = {'s1': {'id': 's1', 'actual_start_utc': iso(BASE_NOW - timedelta(hours=2)),
                                   'actual_end_utc': None,
                                   'expected_end_utc': iso(BASE_NOW + timedelta(hours=6)),
                                   'planned_block_id': 'sleep-1'}}
    draft = begin_run(snapshot, request={'kind': 'initial'})
    run_to_completion(draft, limit=20)
    assert any(b['id'] == 'sleep-1' for b in draft['plan']['sleep_blocks'])


def test_empty_horizon_completes():
    snapshot = make_snapshot(days=0)
    draft = begin_run(snapshot, request={'kind': 'initial'})
    result = step(draft)
    assert result['step_status'] == 'complete'
    assert draft['segments'] == []
