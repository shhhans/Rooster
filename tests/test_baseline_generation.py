from datetime import date

from app.domain.timeline.baseline import generate_baseline
from app.domain.timeline.proposals import create_proposal, decide_proposal
from app.domain.timeline.store import EventStore
import sqlite3


PROFILE = {'home_timezone': 'Asia/Shanghai', 'preferred_sleep_start_hour': 23, 'sleep_need_hours': 8}


def test_generation_progress_survives_deletion_and_fills_forward_gap():
    store = EventStore(':memory:')
    generate_baseline(store, 'Eva', date(2026, 1, 1), date(2026, 1, 3), PROFILE)
    store.connection.execute("DELETE FROM events WHERE user_id='Eva' AND dedupe_key='baseline:2026-01-02'")
    result = generate_baseline(store, 'Eva', date(2026, 1, 6), date(2026, 1, 7), PROFILE)
    assert result['created'] == 4
    assert store.get_by_dedupe_key('Eva', 'sleep', 'baseline:2026-01-02') is None
    assert store.get_by_dedupe_key('Eva', 'sleep', 'baseline:2026-01-04') is not None
    assert store.connection.execute('SELECT generated_through FROM baseline_generation_state WHERE user_id=?', ('Eva',)).fetchone()[0] == '2026-01-07'


def test_existing_events_initialize_generation_progress():
    store = EventStore(':memory:')
    generate_baseline(store, 'Eva', date(2026, 1, 1), date(2026, 1, 3), PROFILE)
    store.connection.execute('DELETE FROM baseline_generation_state')
    store.connection.execute("DELETE FROM events WHERE dedupe_key='baseline:2026-01-02'")
    assert generate_baseline(store, 'Eva', date(2026, 1, 1), date(2026, 1, 3), PROFILE)['created'] == 0
    assert store.connection.execute('SELECT generated_through FROM baseline_generation_state').fetchone()[0] == '2026-01-03'


def test_baseline_idempotent_and_does_not_resurrect_retired_event():
    store = EventStore(':memory:')
    day = date(2026, 1, 1)
    assert generate_baseline(store, 'Eva', day, day, None)['status'] == 'profile_required'
    assert generate_baseline(store, 'Eva', day, day, PROFILE)['created'] == 1
    event = store.get_by_dedupe_key('Eva', 'sleep', 'baseline:2026-01-01')
    assert (event.source, event.confirmation, event.realization) == ('baseline', 'proposed', 'planned')
    txn = create_proposal(store, 'Eva', [event.event_id], [])
    decide_proposal(store, 'Eva', txn, True)
    assert generate_baseline(store, 'Eva', day, day, PROFILE)['created'] == 0


def test_existing_database_is_backed_up_before_source_migration(tmp_path):
    db = tmp_path / 'events.db'
    store = EventStore(str(db))
    generate_baseline(store, 'Eva', date(2026, 1, 1), date(2026, 1, 1), PROFILE)
    store.connection.execute("UPDATE events SET source='roster' WHERE user_id='Eva'")
    store.close()
    # Reconstruct the pre-dimensions shape without discarding event content.
    with sqlite3.connect(db) as conn:
        for column in ('confirmation', 'realization', 'created_in_txn', 'retired_in_txn'):
            conn.execute(f'ALTER TABLE events DROP COLUMN {column}')
    migrated = EventStore(str(db))
    assert migrated.connection.execute('SELECT source FROM events').fetchone()[0] == 'system'
    backups = list((tmp_path / 'backups').glob('events-before-event-dimensions-*.db'))
    assert len(backups) == 1
    with sqlite3.connect(backups[0]) as backup:
        assert backup.execute('SELECT source FROM events').fetchone()[0] == 'roster'
        assert backup.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
    migrated.close()


def test_profile_change_invalidates_pending_proposal():
    from app.domain.timeline.proposals import ProposalConflict
    import pytest
    store = EventStore(':memory:')
    day = date(2026, 1, 1)
    generate_baseline(store, 'Eva', day, day, PROFILE)
    event = store.get_by_dedupe_key('Eva', 'sleep', 'baseline:2026-01-01')
    txn = create_proposal(store, 'Eva', [event.event_id], [])
    generate_baseline(store, 'Eva', day, day, {**PROFILE, 'sleep_need_hours': 7})
    with pytest.raises(ProposalConflict):
        decide_proposal(store, 'Eva', txn, True)


def test_rolling_endpoint_returns_profile_required_without_writing(tmp_path):
    from fastapi.testclient import TestClient
    from app.main import create_app
    with TestClient(create_app(str(tmp_path / 'timeline.db'))) as client:
        result = client.post('/api/v1/baseline/generate', json={'user_id': 'Eva', 'start_date': '2026-01-01'})
        assert result.status_code == 200
        assert result.json() == {'status': 'profile_required', 'created': 0}
        assert client.app.state.timeline_service.store.connection.execute('SELECT COUNT(*) FROM events').fetchone()[0] == 0


def test_profile_can_backfill_confirmed_roster_month(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from app.main import create_app
    monkeypatch.setattr('app.api.baseline_routes._refresh_baseline_issues', lambda *args: None)
    with TestClient(create_app(str(tmp_path / 'timeline.db'))) as client:
        result = client.post('/api/v1/roster/confirm', json={'user_id': 'Eva', 'file_sha256': 'e' * 64,
            'coverage_month': '2026-02', 'proposal': {'proposed_events': []}})
        assert result.json()['baseline']['status'] == 'profile_required'
        result = client.post('/api/v1/baseline/roster-month', json={'user_id': 'Eva', 'month': '2026-02', 'profile': PROFILE})
        assert result.status_code == 200 and result.json()['created'] == 28
        assert client.app.state.timeline_service.store.connection.execute('SELECT count(*) FROM reactive_jobs').fetchone()[0] == 0
        result = client.post('/api/v1/baseline/roster-month', json={'user_id': 'Eva', 'month': '2026-02', 'profile': PROFILE})
        assert result.json()['created'] == 0
