"""Realize completed, accepted agent sleep plans."""
from datetime import datetime, timedelta, timezone

from app.domain.timeline.sleep_track import _resolve_sleep_local
from app.domain.timeline.store import _iso_z, _row_to_event


def realize_completed_sleep(store, now=None):
    now = now or datetime.now(timezone.utc)
    changed = set()
    with store.transaction() as conn:
        rows = conn.execute("""SELECT * FROM events WHERE track='sleep' AND source='agent'
            AND confirmation='accepted' AND realization='planned'
            AND retracted=0 AND superseded_by IS NULL
            AND (created_in_txn IS NULL OR created_in_txn IN (SELECT txn_id FROM proposal_batches WHERE status='accepted'))
            AND (retired_in_txn IS NULL OR retired_in_txn NOT IN (SELECT txn_id FROM proposal_batches WHERE status='accepted'))
            AND occurred_at_utc < ?""", (_iso_z(now),)).fetchall()
        for row in rows:
            event = _row_to_event(row)
            end = _resolve_sleep_local('sleep_end', event.payload['sleep_end'], event.timezone)
            if end > now:
                continue
            actuals = conn.execute("SELECT * FROM events WHERE user_id=? AND track='sleep' AND realization='actual' AND retracted=0 AND superseded_by IS NULL", (event.user_id,)).fetchall()
            overlap = [other for other in map(_row_to_event, actuals)
                       if other.occurred_at_utc < end and _resolve_sleep_local('sleep_end', other.payload['sleep_end'], other.timezone) > event.occurred_at_utc]
            exact = next((other for other in overlap if other.occurred_at_utc == event.occurred_at_utc
                          and _resolve_sleep_local('sleep_end', other.payload['sleep_end'], other.timezone) == end), None)
            if exact:
                store.add_lineage(event.user_id, exact.event_id, [event.event_id])
                store.set_retracted(event.event_id, now, 'matched_actual_sleep')
                changed.add(event.user_id)
                continue
            if overlap:
                continue
            conn.execute("UPDATE events SET realization='actual', status='occurred', updated_at=? WHERE event_id=?", (_iso_z(now), event.event_id))
            store._bump_input_revision(event.user_id, event.local_time, affected_at_utc=event.occurred_at_utc)
            changed.add(event.user_id)
    return changed


def realize_and_refresh(store):
    from app.services.reactive_timeline import refresh_issues
    users = realize_completed_sleep(store)
    now = datetime.now(timezone.utc)
    for user_id in users:
        refresh_issues(store, user_id, now, now + timedelta(hours=48))
