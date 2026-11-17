import sqlite3
from fastapi.testclient import TestClient
from app.main import create_app


def test_confirmation_persists_preview_and_complete_events_once(tmp_path):
    db = tmp_path / 'timeline.db'
    proposal = {'contract_version': 'roster-preview-v1', 'proposed_events': [
        {'event_type': 'flight', 'schedule_payload': {'flight_id': 'XX1', 'departure': '2026-11-03T09:00:00+08:00', 'arrival': '2026-11-03T12:00:00+08:00', 'origin_tz': 'Asia/Shanghai', 'destination_tz': 'Asia/Shanghai', 'pending': []}, 'timeline': {'departure_local': '2026-11-03T09:00', 'departure_timezone': 'Asia/Shanghai'}, 'raw': {'row': 'XX1'}, 'provenance': {'page': 1, 'quote': 'XX1'}},
        {'event_type': 'duty', 'schedule_payload': {'pending': ['duty_end']}, 'timeline': {'duty_start_local': '2026-11-04T09:00', 'duty_timezone': 'Asia/Shanghai'}}],
        'unresolved_items': [{'code': 'unknown', 'message': 'check'}], 'warnings': [{'code': 'warning', 'message': 'review'}]}
    body = {'user_id': 'test-user', 'file_sha256': 'a' * 64, 'coverage_month': '2026-11', 'proposal': proposal}
    with TestClient(create_app(str(db))) as client:
        first = client.post('/api/v1/roster/confirm', json=body)
        assert first.status_code == 200, first.text
        result = first.json()
        assert len(result['imported']) == 1 and len(result['pending_items']) == 1
        assert client.post('/api/v1/roster/confirm', json=body).json() == result
        changed = {**body, 'proposal': {**proposal, 'warnings': []}}
        assert client.post('/api/v1/roster/confirm', json=changed).status_code == 409
        events = client.get('/timeline/schedule/events', params={'user_id':'test-user', 'window_start':'2026-11-01T00:00:00Z', 'window_end':'2026-12-01T00:00:00Z'}).json()
        assert len(events) == 1 and events[0]['payload']['roster_item']['raw'] == {'row': 'XX1'}
        assert client.get('/api/v1/roster/confirmations', params={'user_id':'test-user'}).json()[0]['proposal'] == proposal
    with sqlite3.connect(db) as conn:
        import json
        stored = conn.execute('SELECT proposal_json FROM roster_confirmations').fetchone()
        assert json.loads(stored[0]) == proposal


def test_confirmation_rolls_back_if_any_complete_event_is_invalid(tmp_path):
    db = tmp_path / 'timeline.db'
    good = {'event_type':'duty', 'schedule_payload':{'pending':[], 'duty_start':'2026-11-03T09:00:00+08:00', 'duty_end':'2026-11-03T17:00:00+08:00'}, 'timeline':{'duty_start_local':'2026-11-03T09:00', 'duty_timezone':'Asia/Shanghai'}}
    bad = {**good, 'timeline':{'duty_start_local':'2026-11-03T09:00', 'duty_timezone':'invalid-zone'}}
    with TestClient(create_app(str(db)), raise_server_exceptions=False) as client:
        response = client.post('/api/v1/roster/confirm', json={'user_id':'u', 'file_sha256':'b'*64, 'coverage_month':'2026-11', 'proposal':{'proposed_events':[good, bad]}})
        assert response.status_code >= 400
        events = client.get('/timeline/schedule/events', params={'user_id':'u', 'window_start':'2026-11-01T00:00:00Z', 'window_end':'2026-12-01T00:00:00Z'}).json()
        assert events == []
