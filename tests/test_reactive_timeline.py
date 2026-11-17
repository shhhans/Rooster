from datetime import datetime, timedelta, timezone

from app.domain.timeline.baseline import generate_baseline
from app.services.reactive_timeline import create_job, run_job
from app.domain.timeline.proposals import proposal_diff, decide_proposal
from app.services.timeline_service import TimelineService
from fastapi.testclient import TestClient
from app.main import create_app


def test_manual_review_and_stale_proposals(tmp_path, monkeypatch):
    from app.domain.timeline.proposals import create_proposal
    day = (datetime.now(timezone.utc) + timedelta(days=10)).date()
    start = datetime.combine(day, datetime.min.time(), timezone.utc)
    with TestClient(create_app(str(tmp_path / 'timeline.db'))) as client:
        store = client.app.state.timeline_service.store
        generate_baseline(store, 'Eva', day, day, {'home_timezone': 'UTC', 'preferred_sleep_start_hour': 23, 'sleep_need_hours': 8})
        event = store.get_by_dedupe_key('Eva', 'sleep', 'baseline:' + day.isoformat())
        job, _ = create_job(store, 'Eva', start, start + timedelta(days=1))
        txn = create_proposal(store, 'Eva', [event.event_id], [])
        store.connection.execute("UPDATE reactive_jobs SET status='awaiting_review', txn_id=? WHERE job_id=?", (txn, job))
        generate_baseline(store, 'Eva', day + timedelta(days=1), day + timedelta(days=1), {'home_timezone': 'UTC', 'preferred_sleep_start_hour': 23, 'sleep_need_hours': 8})
        assert client.get('/api/v1/reactive/jobs', params={'user_id':'Eva'}).json()['jobs'][0]['stale'] is True
        called = []
        monkeypatch.setattr('app.api.reactive_routes.run_queued_review', lambda store, job_id: called.append(job_id))
        response = client.post('/api/v1/reactive/jobs', json={'user_id':'Eva', 'start_utc':start.isoformat(), 'end_utc':(start + timedelta(days=2)).isoformat()})
        assert response.status_code == 200 and called == [response.json()['job_id']]
        assert client.post('/api/v1/reactive/jobs', json={'user_id':'Eva', 'start_utc':start.isoformat(), 'end_utc':(start + timedelta(days=2)).isoformat()}).status_code == 409


def test_manual_review_without_issues_does_not_call_agent(monkeypatch):
    from app.services.reactive_timeline import queue_review, run_queued_review
    service = TimelineService(':memory:')
    start = datetime.now(timezone.utc) + timedelta(days=1)
    job = queue_review(service.store, 'Eva', start, start + timedelta(days=1), trigger='manual')
    revision = service.store.get_input_revision('Eva')
    monkeypatch.setattr('app.services.reactive_timeline.refresh_issues', lambda *args: ({'revision': revision}, {'issues': []}))
    monkeypatch.setattr('app.services.reactive_timeline.run_job', lambda *args: (_ for _ in ()).throw(AssertionError('unexpected model call')))
    run_queued_review(service.store, job)
    row = service.store.connection.execute('SELECT status, state_json FROM reactive_jobs WHERE job_id=?', (job,)).fetchone()
    assert row['status'] == 'complete' and 'no issues' in row['state_json']


def test_roster_baseline_conflict_is_persisted_before_agent_error(monkeypatch):
    monkeypatch.delenv('ARK_AGENT_API_KEY', raising=False)
    service = TimelineService(':memory:')
    day = (datetime.now(timezone.utc) + timedelta(days=5)).date()
    generate_baseline(service.store, 'Eva', day, day,
                      {'home_timezone': 'UTC', 'preferred_sleep_start_hour': 23, 'sleep_need_hours': 8})
    start = datetime.combine(day, datetime.min.time(), timezone.utc)
    flight_start = start + timedelta(hours=23, minutes=30)
    service.ingest_schedule_event(user_id='Eva', dedupe_key='flight', occurred_at=flight_start.replace(tzinfo=None).isoformat(),
        timezone='UTC', source='system', status='planned', title='flight',
        payload={'arrival': (flight_start + timedelta(hours=2)).isoformat(), 'flight_id': 'T1', 'origin_tz': 'UTC', 'destination_tz': 'UTC'})
    job, report = create_job(service.store, 'Eva', start, start + timedelta(days=2))
    assert any(i['type'] == 'hard_overlap' for i in report['issues'])
    run_job(service.store, job)
    row = service.store.connection.execute('SELECT status, error FROM reactive_jobs WHERE job_id=?', (job,)).fetchone()
    assert row['status'] == 'failed' and 'ARK_AGENT_API_KEY' in row['error']
    assert service.store.connection.execute("SELECT count(*) FROM reactive_issues WHERE status='active'").fetchone()[0] > 0


def test_agent_edits_are_staged_and_approved_only_after_review():
    import json
    service = TimelineService(':memory:')
    day = (datetime.now(timezone.utc) + timedelta(days=5)).date()
    generate_baseline(service.store, 'Eva', day, day,
                      {'home_timezone': 'UTC', 'preferred_sleep_start_hour': 23, 'sleep_need_hours': 8})
    original = service.store.get_by_dedupe_key('Eva', 'sleep', 'baseline:' + day.isoformat())
    start = datetime.combine(day, datetime.min.time(), timezone.utc)
    flight = start + timedelta(hours=23, minutes=30)
    service.ingest_schedule_event(user_id='Eva', dedupe_key='flight', occurred_at=flight.replace(tzinfo=None).isoformat(),
        timezone='UTC', source='system', status='planned', title='flight',
        payload={'arrival': (flight + timedelta(hours=2)).isoformat(), 'flight_id': 'T1', 'origin_tz': 'UTC', 'destination_tz': 'UTC'})
    job, _ = create_job(service.store, 'Eva', start, start + timedelta(days=2))
    def client(messages, tools):
        if any(m.get('role') == 'tool' for m in messages):
            return {'role': 'assistant', 'content': '已调整'}
        args = {'operations': [{'op': 'delete', 'event_id': original.event_id}], 'display_timezone': 'UTC', 'context_hours': 24}
        return {'role': 'assistant', 'content': '', 'tool_calls': [{'id': '1', 'type': 'function', 'function': {'name': 'edit_schedule', 'arguments': json.dumps(args)}}]}
    run_job(service.store, job, client=client)
    row = service.store.connection.execute('SELECT status, txn_id, error FROM reactive_jobs WHERE job_id=?', (job,)).fetchone()
    assert row['status'] == 'awaiting_review', row['error']
    assert proposal_diff(service.store, 'Eva', row['txn_id'])['before'][0]['event_id'] == original.event_id
    assert service.store.get_by_id(original.event_id).retired_in_txn is None
    decide_proposal(service.store, 'Eva', row['txn_id'], True)
    assert service.store.get_by_id(original.event_id).retired_in_txn == row['txn_id']


def test_confirm_roster_generates_month_and_exposes_issues(tmp_path, monkeypatch):
    monkeypatch.delenv('ARK_AGENT_API_KEY', raising=False)
    day = (datetime.now(timezone.utc) + timedelta(days=35)).date().replace(day=10)
    start = datetime.combine(day, datetime.min.time(), timezone.utc) + timedelta(hours=23, minutes=30)
    end = start + timedelta(hours=2)
    item = {'event_type': 'flight', 'schedule_payload': {'pending': [], 'flight_id': 'T1',
             'departure': start.isoformat(), 'arrival': end.isoformat(), 'origin_tz': 'UTC', 'destination_tz': 'UTC'},
            'timeline': {'departure_local': start.replace(tzinfo=None).isoformat(), 'departure_timezone': 'UTC'}}
    body = {'user_id': 'Eva', 'file_sha256': 'c' * 64, 'coverage_month': day.strftime('%Y-%m'),
            'proposal': {'proposed_events': [item]},
            'profile': {'home_timezone': 'UTC', 'preferred_sleep_start_hour': 23, 'sleep_need_hours': 8}}
    with TestClient(create_app(str(tmp_path / 'timeline.db'))) as client:
        response = client.post('/api/v1/roster/confirm', json=body)
        assert response.status_code == 200, response.text
        assert response.json()['baseline']['created'] >= 28
        issues = client.get('/api/v1/reactive/issues', params={'user_id': 'Eva'}).json()['issues']
        assert any(issue['type'] == 'hard_overlap' for issue in issues)
        jobs = client.get('/api/v1/reactive/jobs', params={'user_id': 'Eva'}).json()['jobs']
        assert jobs and jobs[0]['status'] == 'failed' and 'ARK_AGENT_API_KEY' in jobs[0]['error']


def test_approval_api_removes_only_resolved_conflict_and_refreshes_timeline(tmp_path):
    with TestClient(create_app(str(tmp_path / 'timeline.db'))) as client:
        store = client.app.state.timeline_service.store
        service = client.app.state.timeline_service
        day = (datetime.now(timezone.utc) + timedelta(days=5)).date()
        generate_baseline(store, 'Eva', day, day, {'home_timezone': 'UTC', 'preferred_sleep_start_hour': 23, 'sleep_need_hours': 8})
        baseline = store.get_by_dedupe_key('Eva', 'sleep', 'baseline:' + day.isoformat())
        start = datetime.combine(day, datetime.min.time(), timezone.utc)
        flight = start + timedelta(hours=23, minutes=30)
        service.ingest_schedule_event(user_id='Eva', dedupe_key='flight', occurred_at=flight.replace(tzinfo=None).isoformat(),
            timezone='UTC', source='system', status='planned', title='flight',
            payload={'arrival': (flight + timedelta(hours=2)).isoformat(), 'flight_id': 'T1', 'origin_tz': 'UTC', 'destination_tz': 'UTC'})
        job, _ = create_job(store, 'Eva', start, start + timedelta(days=2))
        from app.domain.timeline.proposals import create_proposal
        txn = create_proposal(store, 'Eva', [baseline.event_id], [])
        store.connection.execute("UPDATE reactive_jobs SET status='awaiting_review', txn_id=? WHERE job_id=?", (txn, job))
        before = client.get('/api/v1/reactive/issues', params={'user_id': 'Eva'}).json()['issues']
        assert any(i['type'] == 'hard_overlap' for i in before)
        result = client.post(f'/api/v1/reactive/proposals/{txn}/decision', json={'user_id': 'Eva', 'accept': True})
        assert result.status_code == 200, result.text
        after = client.get('/api/v1/reactive/issues', params={'user_id': 'Eva'}).json()['issues']
        assert not any(i['type'] == 'hard_overlap' and baseline.event_id in i.get('event_ids', []) for i in after)
        visible = client.get('/timeline/sleep/events', params={'user_id': 'Eva', 'window_start': start.isoformat(), 'window_end': (start + timedelta(days=2)).isoformat()}).json()
        assert all(e['event_id'] != baseline.event_id for e in visible)


def test_personal_event_with_interval_participates_in_conflict_detection():
    service = TimelineService(':memory:')
    day = (datetime.now(timezone.utc) + timedelta(days=5)).date()
    generate_baseline(service.store, 'Eva', day, day, {'home_timezone': 'UTC', 'preferred_sleep_start_hour': 23, 'sleep_need_hours': 8})
    start = datetime.combine(day, datetime.min.time(), timezone.utc)
    personal = start + timedelta(hours=23, minutes=30)
    item = service.ingest_schedule_event(user_id='Eva', dedupe_key='meeting', occurred_at=personal.replace(tzinfo=None).isoformat(),
        timezone='UTC', source='user', status='planned', title='personal',
        payload={'end_utc': (personal + timedelta(hours=1)).isoformat()})
    _, report = create_job(service.store, 'Eva', start, start + timedelta(days=2))
    assert any(issue['type'] == 'hard_overlap' and item.event_id in issue['event_ids'] for issue in report['issues'])


def test_snapshot_uses_configured_bedtime_for_physiology():
    from app.services.reactive_timeline import snapshot
    service = TimelineService(':memory:')
    day = (datetime.now(timezone.utc) + timedelta(days=5)).date()
    generate_baseline(service.store, 'Eva', day, day, {'home_timezone': 'UTC', 'preferred_sleep_start_hour': 21.5, 'sleep_need_hours': 8})
    start = datetime.combine(day, datetime.min.time(), timezone.utc)
    profile = snapshot(service.store, 'Eva', start, start + timedelta(days=1))['profile']
    assert profile['habit_bed_local'] == '21:30'
    assert profile['habit_wake_local'] == '05:30'


def test_accepted_nap_keeps_role_in_next_planning_snapshot():
    from app.services.reactive_timeline import snapshot
    from app.domain.timeline.proposals import create_proposal
    service = TimelineService(':memory:')
    day = (datetime.now(timezone.utc) + timedelta(days=5)).date()
    generate_baseline(service.store, 'Eva', day, day, {'home_timezone': 'UTC', 'preferred_sleep_start_hour': 23, 'sleep_need_hours': 8})
    old = service.store.get_by_dedupe_key('Eva', 'sleep', 'baseline:' + day.isoformat())
    txn = create_proposal(service.store, 'Eva', [old.event_id], [{'sleep_start': f'{day}T14:00:00',
        'sleep_end': f'{day}T15:30:00', 'timezone': 'UTC', 'role': 'nap', 'parent_event_ids': [old.event_id]}])
    decide_proposal(service.store, 'Eva', txn, True)
    start = datetime.combine(day, datetime.min.time(), timezone.utc)
    blocks = snapshot(service.store, 'Eva', start, start + timedelta(days=2))['active_plan']['sleep_blocks']
    assert len(blocks) == 1 and blocks[0]['role'] == 'nap'


def test_unavailable_model_does_not_resolve_physiological_issues(monkeypatch):
    import json
    from app.services.reactive_timeline import refresh_issues
    service = TimelineService(':memory:')
    day = (datetime.now(timezone.utc) + timedelta(days=5)).date()
    generate_baseline(service.store, 'Eva', day, day, {'home_timezone': 'UTC', 'preferred_sleep_start_hour': 23, 'sleep_need_hours': 8})
    start = datetime.combine(day, datetime.min.time(), timezone.utc)
    end = start + timedelta(days=1)
    for kind in ('sleep_deficit', 'sleep_onset_pressure_low', 'hard_overlap'):
        issue = {'type': kind, 'start_utc': start.isoformat(), 'end_utc': end.isoformat()}
        service.store.connection.execute('INSERT INTO reactive_issues VALUES (?, ?, ?, ?, ?, ?, ?)',
            (kind, 'Eva', 'active', start.isoformat().replace('+00:00', 'Z'), end.isoformat().replace('+00:00', 'Z'), json.dumps(issue), start.isoformat()))
    monkeypatch.setattr('app.services.reactive_timeline.review', lambda *args: {'issues': [], 'model_unavailable': 'not computed'})
    refresh_issues(service.store, 'Eva', start, end)
    rows = service.store.connection.execute('SELECT issue_key, status FROM reactive_issues').fetchall()
    assert {r['issue_key']: r['status'] for r in rows} == {'sleep_deficit': 'active', 'sleep_onset_pressure_low': 'active', 'hard_overlap': 'resolved'}


def test_roster_to_agent_to_review_end_to_end(tmp_path, monkeypatch):
    import json
    from app.planning import timeline_agent
    day = (datetime.now(timezone.utc) + timedelta(days=35)).date().replace(day=10)
    start = datetime.combine(day, datetime.min.time(), timezone.utc) + timedelta(hours=23, minutes=30)
    item = {'event_type': 'flight', 'schedule_payload': {'pending': [], 'flight_id': 'T1', 'departure': start.isoformat(),
            'arrival': (start + timedelta(hours=2)).isoformat(), 'origin_tz': 'UTC', 'destination_tz': 'UTC'},
            'timeline': {'departure_local': start.replace(tzinfo=None).isoformat(), 'departure_timezone': 'UTC'}}
    app = create_app(str(tmp_path / 'timeline.db'))
    with TestClient(app) as client:
        def agent(messages, tools, **kwargs):
            if any(m.get('role') == 'tool' for m in messages):
                return {'role': 'assistant', 'content': '完成'}
            event = app.state.timeline_service.store.get_by_dedupe_key('Eva', 'sleep', 'baseline:' + day.isoformat())
            args = {'operations': [{'op': 'delete', 'event_id': event.event_id}], 'display_timezone': 'UTC', 'context_hours': 24}
            return {'role': 'assistant', 'content': '', 'tool_calls': [{'id': 'call-1', 'type': 'function', 'function': {'name': 'edit_schedule', 'arguments': json.dumps(args)}}]}
        monkeypatch.setattr(timeline_agent, 'request', agent)
        response = client.post('/api/v1/roster/confirm', json={'user_id': 'Eva', 'file_sha256': 'd' * 64,
            'coverage_month': day.strftime('%Y-%m'), 'proposal': {'proposed_events': [item]},
            'profile': {'home_timezone': 'UTC', 'preferred_sleep_start_hour': 23, 'sleep_need_hours': 8}})
        assert response.status_code == 200, response.text
        jobs = client.get('/api/v1/reactive/jobs', params={'user_id': 'Eva'}).json()['jobs']
        assert jobs[0]['status'] == 'awaiting_review', jobs[0]
        assert jobs[0]['tool_calls'][0]['name'] == 'edit_schedule'
        assert all(jobs[0]['tool_calls'][0]['range'])
        job_path = '/api/v1/reactive/jobs/' + jobs[0]['job_id']
        activity = client.get(job_path + '/activity', params={'user_id': 'Eva'}).json()
        tool = activity['steps'][0]['tools'][0]
        assert tool['name'] == 'edit_schedule' and tool['started_at'] and tool['finished_at']
        messages = client.get(job_path + '/messages', params={'user_id': 'Eva'}).json()['messages']
        assert [message['role'] for message in messages] == ['system', 'user', 'assistant', 'tool', 'assistant']
        txn = jobs[0]['txn_id']
        diff = client.get(f'/api/v1/reactive/proposals/{txn}', params={'user_id': 'Eva'}).json()
        assert len(diff['before']) == 1 and not diff['after']
        original = diff['before'][0]['event_id']
        assert app.state.timeline_service.store.get_by_id(original).retired_in_txn is None
        response = client.post(f'/api/v1/reactive/proposals/{txn}/decision', json={'user_id': 'Eva', 'accept': True})
        assert response.status_code == 200, response.text
        assert app.state.timeline_service.store.get_by_id(original).retired_in_txn == txn
        issues = client.get('/api/v1/reactive/issues', params={'user_id': 'Eva'}).json()['issues']
        assert not any(i['type'] == 'hard_overlap' and original in i.get('event_ids', []) for i in issues)
