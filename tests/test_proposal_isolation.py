from datetime import datetime, timezone

import pytest

from app.domain.timeline.models import ScheduleEvent
from app.domain.timeline.proposals import ProposalConflict, create_proposal, decide_proposal, proposal_diff
from app.domain.timeline.store import EventStore


def baseline(store):
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    event = ScheduleEvent(event_id='A', track='sleep', user_id='Eva', dedupe_key='A',
                          occurred_at_utc=now, local_time='2026-01-01T00:00:00',
                          timezone='UTC', source='baseline', status='planned', title='sleep',
                          payload={'sleep_start': '2026-01-01T00:00:00', 'sleep_end': '2026-01-01T08:00:00'},
                          confirmation='proposed', created_at=now, updated_at=now)
    store.insert(event)
    return event


def window(store):
    return store.list_for_user_window('Eva', 'sleep', datetime(2026, 1, 1, tzinfo=timezone.utc),
                                      datetime(2026, 1, 2, tzinfo=timezone.utc), False)


def make(store):
    return create_proposal(store, 'Eva', ['A'], [
        {'sleep_start': '2026-01-01T00:00:00', 'sleep_end': '2026-01-01T03:00:00', 'timezone': 'UTC', 'parent_event_ids': ['A']},
        {'sleep_start': '2026-01-01T05:00:00', 'sleep_end': '2026-01-01T08:00:00', 'timezone': 'UTC', 'parent_event_ids': ['A']},
    ])


def test_split_isolated_until_atomic_approval():
    store = EventStore(':memory:')
    baseline(store)
    rev = store.get_input_revision('Eva')
    txn = make(store)
    assert store.get_input_revision('Eva') == rev
    assert [e.event_id for e in window(store)] == ['A']
    diff = proposal_diff(store, 'Eva', txn)
    assert len(diff['after']) == 2
    assert all(store.event_lineage('Eva', e['event_id'])['parents'] == ['A'] for e in diff['after'])
    decide_proposal(store, 'Eva', txn, True)
    assert {e.event_id for e in window(store)} == {e['event_id'] for e in diff['after']}
    assert all(e.confirmation == 'accepted' for e in window(store))


def test_rejection_and_stale_proposal_preserve_effective_view():
    store = EventStore(':memory:')
    baseline(store)
    txn = make(store)
    decide_proposal(store, 'Eva', txn, False)
    assert [e.event_id for e in window(store)] == ['A']
    txn = make(store)
    with store.transaction():
        store._bump_input_revision('Eva', '2026-01-01')
    with pytest.raises(ProposalConflict):
        decide_proposal(store, 'Eva', txn, True)
    assert [e.event_id for e in window(store)] == ['A']


def test_actual_cannot_be_retired():
    store = EventStore(':memory:')
    event = baseline(store)
    store.connection.execute("UPDATE events SET realization='actual' WHERE event_id=?", (event.event_id,))
    with pytest.raises(ProposalConflict):
        make(store)


def test_merge_keeps_both_parents_and_originals_until_approval():
    from dataclasses import replace
    store = EventStore(':memory:')
    original = baseline(store)
    store.insert(replace(original, event_id='D', dedupe_key='D',
                         occurred_at_utc=datetime(2026, 1, 1, 10, tzinfo=timezone.utc),
                         local_time='2026-01-01T10:00:00',
                         payload={'sleep_start': '2026-01-01T10:00:00', 'sleep_end': '2026-01-01T11:00:00'}))
    txn = create_proposal(store, 'Eva', ['A', 'D'], [{'sleep_start': '2026-01-01T00:00:00',
        'sleep_end': '2026-01-01T11:00:00', 'timezone': 'UTC', 'parent_event_ids': ['A', 'D']}])
    new = proposal_diff(store, 'Eva', txn)['after'][0]['event_id']
    assert set(store.event_lineage('Eva', new)['parents']) == {'A', 'D'}
    assert {e.event_id for e in window(store)} == {'A', 'D'}
    decide_proposal(store, 'Eva', txn, True)
    assert [e.event_id for e in window(store)] == [new]


def test_pending_and_rejected_candidates_are_hidden_from_public_sleep_api(tmp_path):
    from fastapi.testclient import TestClient
    from app.main import create_app
    with TestClient(create_app(str(tmp_path / 'timeline.db'))) as client:
        store = client.app.state.timeline_service.store
        baseline(store)
        txn = make(store)
        candidate = proposal_diff(store, 'Eva', txn)['after'][0]['event_id']
        assert client.get(f'/timeline/sleep/events/{candidate}').status_code == 404
        assert client.get(f'/timeline/schedule/events/{candidate}').status_code == 404
        assert client.post(f'/timeline/schedule/events/{candidate}/retract', json={'reason': 'x'}).status_code == 404
        assert client.post(f'/timeline/sleep/events/{candidate}/retract', json={'reason': 'x'}).status_code == 404
        params = {'user_id': 'Eva', 'window_start': '2026-01-01T00:00:00Z', 'window_end': '2026-01-02T00:00:00Z', 'include_history': 'true'}
        assert [event['event_id'] for event in client.get('/timeline/sleep/events', params=params).json()] == ['A']
        decide_proposal(store, 'Eva', txn, False)
        assert client.get(f'/timeline/sleep/events/{candidate}').status_code == 404


def test_review_rejects_a_proposal_after_its_sleep_has_started(tmp_path):
    from fastapi.testclient import TestClient
    from app.main import create_app
    with TestClient(create_app(str(tmp_path / 'timeline.db'))) as client:
        store = client.app.state.timeline_service.store
        baseline(store)
        txn = make(store)
        store.connection.execute('INSERT INTO reactive_jobs VALUES (?, ?, ?, ?, ?, ?, ?, NULL, NULL, ?, ?)',
            ('job', 'Eva', 'awaiting_review', '2026-01-01T00:00:00Z', '2026-01-02T00:00:00Z',
             store.get_input_revision('Eva'), txn, '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z'))
        response = client.post(f'/api/v1/reactive/proposals/{txn}/decision', json={'user_id': 'Eva', 'accept': True})
        assert response.status_code == 409
        assert proposal_diff(store, 'Eva', txn)['status'] == 'pending'
        assert [e.event_id for e in window(store)] == ['A']
