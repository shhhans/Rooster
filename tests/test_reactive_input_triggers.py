import json
from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient

from app.domain.timeline.baseline import generate_baseline
from app.main import create_app


def test_actual_sleep_and_personal_changes_queue_reviews_once(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr('app.services.reactive_trigger.run_queued_review', lambda store, job: calls.append(job))
    now = datetime.now(timezone.utc).replace(microsecond=0)
    before = now - timedelta(hours=10)
    after = now - timedelta(hours=3)
    profile = {'home_timezone': 'UTC', 'preferred_sleep_start_hour': 23, 'sleep_need_hours': 8}
    with TestClient(create_app(str(tmp_path / 'timeline.db'))) as client:
        store = client.app.state.timeline_service.store
        generate_baseline(store, 'Eva', now.date(), now.date(), profile)
        body = {'user_id': 'Eva', 'dedupe_key': 'actual', 'sleep_start': before.replace(tzinfo=None).isoformat(),
                'sleep_end': after.replace(tzinfo=None).isoformat(), 'timezone': 'UTC', 'source': 'manual'}
        result = client.post('/timeline/sleep/events', json=body)
        assert result.status_code == 200
        assert result.json()['source'] == 'user' and result.json()['realization'] == 'actual'
        assert len(calls) == 1
        row = store.connection.execute('SELECT state_json FROM reactive_jobs WHERE job_id=?', (calls[-1],)).fetchone()
        assert json.loads(row['state_json'])['trigger'] == 'actual_sleep_updated'
        assert client.post('/timeline/sleep/events', json=body).status_code == 200
        assert len(calls) == 1
        event_id = result.json()['event_id']
        result = client.post(f'/timeline/sleep/events/{event_id}/correct', json={'sleep_start': (before + timedelta(minutes=30)).replace(tzinfo=None).isoformat(),
             'sleep_end': after.replace(tzinfo=None).isoformat(), 'timezone': 'UTC'})
        assert result.status_code == 200 and len(calls) == 2
        assert client.post(f"/timeline/sleep/events/{result.json()['event_id']}/retract", json={'reason': 'correction'}).status_code == 200
        assert len(calls) == 3
        personal = {'user_id': 'Eva', 'dedupe_key': 'meeting', 'occurred_at': now.replace(tzinfo=None).isoformat(), 'timezone': 'UTC',
                    'source': 'user', 'status': 'planned', 'title': 'meeting', 'payload': {'end_utc': (now + timedelta(hours=1)).isoformat()}}
        result = client.post('/timeline/schedule/events', json=personal)
        assert result.status_code == 200 and len(calls) == 4
        row = store.connection.execute('SELECT state_json FROM reactive_jobs WHERE job_id=?', (calls[-1],)).fetchone()
        assert json.loads(row['state_json'])['trigger'] == 'personal_event_added'
        event_id = result.json()['event_id']
        corrected_end = now + timedelta(hours=4)
        result = client.post(f'/timeline/schedule/events/{event_id}/correct', json={
            'occurred_at': (now + timedelta(hours=2)).replace(tzinfo=None).isoformat(),
            'timezone': 'UTC', 'status': 'planned', 'title': 'meeting',
            'payload': {'end_utc': corrected_end.isoformat()}})
        assert result.status_code == 200 and len(calls) == 5
        row = store.connection.execute('SELECT start_utc, end_utc FROM reactive_jobs WHERE job_id=?', (calls[-1],)).fetchone()
        assert row['start_utc'] == now.strftime('%Y-%m-%dT%H:%M:%SZ')
        assert row['end_utc'] == corrected_end.strftime('%Y-%m-%dT%H:%M:%SZ')
        assert client.post(f"/timeline/schedule/events/{result.json()['event_id']}/retract", json={'reason': 'cancelled'}).status_code == 200
        assert len(calls) == 6


def test_actual_sleep_refreshes_future_physiology(monkeypatch):
    from app.domain.timeline.store import EventStore
    from app.services.reactive_timeline import queue_review, run_queued_review
    store = EventStore(':memory:')
    now = datetime.now(timezone.utc).replace(microsecond=0)
    job = queue_review(store, 'Eva', now - timedelta(hours=10), now - timedelta(hours=3), trigger='actual_sleep_updated')
    ranges = []
    def refresh(store, user, start, end):
        ranges.append((start, end))
        return {'revision': store.get_input_revision(user)}, {'issues': []}
    monkeypatch.setattr('app.services.reactive_timeline.refresh_issues', refresh)
    monkeypatch.setattr('app.services.reactive_timeline.run_job', lambda store, job: None)
    run_queued_review(store, job)
    assert len(ranges) == 2
    assert ranges[0][0] < now and ranges[1][0] >= now
    assert ranges[1][1] - ranges[1][0] >= timedelta(hours=48)


def test_actual_sleep_cannot_be_future_or_overwrite_planned_baseline(tmp_path):
    now = datetime.now(timezone.utc).replace(microsecond=0)
    with TestClient(create_app(str(tmp_path / 'timeline.db'))) as client:
        future = {'user_id': 'Eva', 'dedupe_key': 'future', 'sleep_start': now.replace(tzinfo=None).isoformat(),
                  'sleep_end': (now + timedelta(hours=2)).replace(tzinfo=None).isoformat(), 'timezone': 'UTC'}
        assert client.post('/timeline/sleep/events', json=future).status_code == 400
        store = client.app.state.timeline_service.store
        generate_baseline(store, 'Eva', now.date(), now.date(), {'home_timezone': 'UTC', 'preferred_sleep_start_hour': 23, 'sleep_need_hours': 8})
        baseline = store.get_by_dedupe_key('Eva', 'sleep', 'baseline:' + now.date().isoformat())
        result = client.post(f'/timeline/sleep/events/{baseline.event_id}/correct', json={
            'sleep_start': (now - timedelta(hours=8)).replace(tzinfo=None).isoformat(),
            'sleep_end': (now - timedelta(hours=1)).replace(tzinfo=None).isoformat(), 'timezone': 'UTC'})
        assert result.status_code == 400
        assert store.get_by_id(baseline.event_id).realization == 'planned'
