from datetime import datetime, timezone
from app.domain.timeline.store import EventStore
from app.services.timeline_service import TimelineService
from app.services.commute import sync_commutes, trim_commutes_for_actual_sleep

def test_commute_generation_and_recalculation(tmp_path):
    service = TimelineService(str(tmp_path / 'test.db'))
    store = service.store
    service.ingest_schedule_event('Eva', 'flight1', '2027-01-03T10:00:00', 'Asia/Hong_Kong', 'system', 'planned', 'flight', {'origin':'HKG', 'destination':'SYD', 'departure':'2027-01-03T02:00:00Z', 'arrival':'2027-01-03T11:00:00Z', 'duty_start_utc':'2027-01-03T01:00:00Z', 'duty_end_utc':'2027-01-03T12:00:00Z'})
    now = datetime(2027, 1, 1, tzinfo=timezone.utc)
    profile = {'homeport':'HKG', 'commute_minutes':60}
    result = sync_commutes(store, 'Eva', profile, now)
    assert result['created'] == 2
    events = [dict(r) for r in store.connection.execute("SELECT * FROM events WHERE title='commute' AND retracted=0 ORDER BY occurred_at_utc")]
    assert [x['occurred_at_utc'] for x in events] == ['2027-01-03T00:00:00Z', '2027-01-03T11:00:00Z']
    assert sync_commutes(store, 'Eva', profile, now)['created'] == 0
    result = sync_commutes(store, 'Eva', {'homeport':'HKG', 'commute_minutes':120}, now)
    assert (result['created'], result['removed']) == (1, 1)
    service.close()

def test_arrival_home_and_missing_homeport(tmp_path):
    service = TimelineService(str(tmp_path / 'test.db'))
    service.ingest_schedule_event('Eva', 'flight2', '2027-01-03T10:00:00', 'Australia/Sydney', 'system', 'planned', 'flight', {'origin':'SYD', 'destination':'HKG', 'departure':'2027-01-02T23:00:00Z', 'arrival':'2027-01-03T08:00:00Z', 'duty_start_utc':'2027-01-02T22:00:00Z', 'duty_end_utc':'2027-01-03T09:00:00Z'})
    now = datetime(2027, 1, 1, tzinfo=timezone.utc)
    assert sync_commutes(service.store, 'Eva', {}, now)['status'] == 'homeport_required'
    assert sync_commutes(service.store, 'Eva', {'homeport':'HKG', 'commute_minutes':60}, now)['created'] == 2
    import json
    rows = service.store.connection.execute("SELECT occurred_at_utc,payload FROM events WHERE title='commute' AND retracted=0 ORDER BY occurred_at_utc").fetchall()
    assert rows[0]['occurred_at_utc'] == '2027-01-02T20:00:00Z'
    assert rows[1]['occurred_at_utc'] == '2027-01-03T08:00:00Z'
    assert json.loads(rows[1]['payload'])['end_utc'] == '2027-01-03T10:00:00+00:00'
    service.close()

def test_only_printed_duty_sides_generate_commute(tmp_path):
    service = TimelineService(str(tmp_path / 'test.db'))
    common = {'origin':'HKG', 'destination':'SYD', 'departure':'2027-01-03T02:00:00Z', 'arrival':'2027-01-03T11:00:00Z'}
    service.ingest_schedule_event('Eva', 'missing', '2027-01-03T10:00:00', 'Asia/Hong_Kong', 'system', 'planned', 'flight', common)
    service.ingest_schedule_event('Eva', 'one', '2027-01-03T10:00:00', 'Asia/Hong_Kong', 'system', 'planned', 'flight', {**common, 'duty_start_utc':'2027-01-03T01:00:00Z'})
    result = sync_commutes(service.store, 'Eva', {'homeport':'HKG'}, datetime(2027,1,1,tzinfo=timezone.utc))
    assert result['created'] == 1
    service.close()

def test_persisted_duty_boundaries_drive_commutes(tmp_path):
    service = TimelineService(str(tmp_path / 'test.db'))
    event = service.ingest_schedule_event('Eva', 'flight', '2027-01-03T10:00:00', 'Asia/Hong_Kong', 'system', 'planned', 'flight', {'origin':'HKG', 'destination':'SYD', 'departure':'2027-01-03T02:00:00Z', 'arrival':'2027-01-03T11:00:00Z', 'duty_start_utc':'2027-01-03T00:30:00Z', 'duty_end_utc':'2027-01-03T11:45:00Z'})
    assert service.store.get_by_id(event.event_id).to_dict()['duty_start_utc'] == '2027-01-03T00:30:00Z'
    sync_commutes(service.store, 'Eva', {'homeport':'HKG', 'commute_minutes':60}, datetime(2027, 1, 1, tzinfo=timezone.utc))
    rows = service.store.connection.execute("SELECT occurred_at_utc,payload FROM events WHERE title='commute' AND retracted=0 ORDER BY occurred_at_utc").fetchall()
    assert [row[0] for row in rows] == ['2027-01-02T23:30:00Z', '2027-01-03T11:00:00Z']
    import json
    assert json.loads(rows[0][1])['end_utc'] == '2027-01-03T02:00:00+00:00'
    assert json.loads(rows[1][1])['end_utc'] == '2027-01-03T13:45:00+00:00'
    assert all(not json.loads(row[1])['duty_boundary_estimated'] for row in rows)
    service.close()

def test_explicit_historical_recalculation(tmp_path):
    service = TimelineService(str(tmp_path / 'test.db'))
    service.ingest_schedule_event('Eva', 'flight', '2027-01-03T10:00:00', 'Asia/Hong_Kong', 'system', 'planned', 'flight', {'origin':'HKG', 'destination':'SYD', 'departure':'2027-01-03T02:00:00Z', 'arrival':'2027-01-03T11:00:00Z', 'duty_end_utc':'2027-01-03T12:00:00Z'})
    profile = {'homeport':'HKG', 'commute_minutes':120}
    now = datetime(2027, 2, 1, tzinfo=timezone.utc)
    assert sync_commutes(service.store, 'Eva', profile, now)['created'] == 0
    assert sync_commutes(service.store, 'Eva', profile, now, include_history=True)['created'] == 1
    assert sync_commutes(service.store, 'Eva', profile, now, include_history=True)['created'] == 0
    service.close()

def test_saving_settings_recalculates_future_commute(tmp_path):
    from fastapi.testclient import TestClient
    from app.main import create_app
    with TestClient(create_app(str(tmp_path / 'test.db'))) as client:
        service = client.app.state.timeline_service
        service.ingest_schedule_event('Eva', 'flight', '2099-01-03T10:00:00', 'Asia/Hong_Kong', 'system', 'planned', 'flight', {'origin':'HKG', 'destination':'SYD', 'departure':'2099-01-03T02:00:00Z', 'arrival':'2099-01-03T11:00:00Z', 'duty_start_utc':'2099-01-03T01:00:00Z'})
        first = client.post('/api/v1/commute/settings', json={'user_id':'Eva','homeport':'HKG','commute_minutes':180})
        second = client.post('/api/v1/commute/settings', json={'user_id':'Eva','homeport':'HKG','commute_minutes':120})
        assert first.status_code == second.status_code == 200
        assert second.json()['created'] == 1 and second.json()['removed'] == 1
        row = service.store.connection.execute("select occurred_at_utc from events where title='commute' and retracted=0").fetchone()
        assert row[0] == '2099-01-02T23:00:00Z'

def test_actual_sleep_trims_near_flight_and_restores_when_retracted(tmp_path):
    import json
    service = TimelineService(str(tmp_path / 'test.db'))
    service.ingest_schedule_event('Eva', 'flight', '2020-01-03T10:00:00', 'Asia/Hong_Kong', 'system', 'planned', 'flight', {'origin':'HKG', 'destination':'SYD', 'departure':'2020-01-03T02:00:00Z', 'arrival':'2020-01-03T11:00:00Z', 'duty_start_utc':'2020-01-03T01:00:00Z', 'duty_end_utc':'2020-01-03T12:00:00Z'})
    now = datetime(2020, 1, 1, tzinfo=timezone.utc)
    sync_commutes(service.store, 'Eva', {'homeport':'HKG', 'commute_minutes':120}, now)
    before = service.ingest_sleep_event('Eva', 'sleep1', '2020-01-03T07:30:00', '2020-01-03T08:00:00', 'Asia/Hong_Kong', 'user')
    after = service.ingest_sleep_event('Eva', 'sleep2', '2020-01-03T19:30:00', '2020-01-03T20:00:00', 'Asia/Hong_Kong', 'user')
    assert trim_commutes_for_actual_sleep(service.store, 'Eva') == 2
    rows = service.store.connection.execute("SELECT occurred_at_utc,payload FROM events WHERE title='commute' AND retracted=0 ORDER BY occurred_at_utc").fetchall()
    assert rows[0][0] == '2020-01-03T00:00:00Z'
    assert json.loads(rows[1][1])['end_utc'] == '2020-01-03T11:30:00+00:00'
    service.retract_sleep_event(before.event_id); service.retract_sleep_event(after.event_id)
    assert trim_commutes_for_actual_sleep(service.store, 'Eva') == 2
    rows = service.store.connection.execute("SELECT occurred_at_utc,payload FROM events WHERE title='commute' AND retracted=0 ORDER BY occurred_at_utc").fetchall()
    assert rows[0][0] == '2020-01-02T23:00:00Z'
    assert json.loads(rows[1][1])['end_utc'] == '2020-01-03T14:00:00+00:00'
    service.close()

def test_fully_trimmed_commute_stays_hidden_until_sleep_retracted(tmp_path):
    service = TimelineService(str(tmp_path / 'test.db'))
    service.ingest_schedule_event('Eva', 'flight', '2020-01-03T10:00:00', 'Asia/Hong_Kong', 'system', 'planned', 'flight', {'origin':'HKG', 'destination':'SYD', 'departure':'2020-01-03T02:00:00Z', 'arrival':'2020-01-03T11:00:00Z', 'duty_start_utc':'2020-01-03T01:00:00Z'})
    now = datetime(2020, 1, 1, tzinfo=timezone.utc); profile = {'homeport':'HKG', 'commute_minutes':120}
    sync_commutes(service.store, 'Eva', profile, now)
    sleep = service.ingest_sleep_event('Eva', 'sleep', '2020-01-03T09:00:00', '2020-01-03T10:00:00', 'Asia/Hong_Kong', 'agent')
    assert trim_commutes_for_actual_sleep(service.store, 'Eva') == 1
    assert service.store.connection.execute("select count(*) from events where title='commute' and retracted=0").fetchone()[0] == 0
    assert sync_commutes(service.store, 'Eva', profile, now)['created'] == 0
    service.retract_sleep_event(sleep.event_id)
    assert trim_commutes_for_actual_sleep(service.store, 'Eva') == 1
    assert service.store.connection.execute("select count(*) from events where title='commute' and retracted=0").fetchone()[0] == 1
    service.close()
