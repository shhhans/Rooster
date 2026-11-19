"""Maintain future system commute events derived from flights."""
from datetime import datetime, timedelta, timezone
import json
import uuid
from zoneinfo import ZoneInfo

from app.domain.timeline.models import ScheduleEvent
from app.roster_parser.normalize import AIRPORT_TIMEZONES
from app.services.reactive_timeline import _parse
from app.domain.timeline.sleep_track import _resolve_sleep_local


def trim_commutes_for_actual_sleep(store, user_id, now=None):
    """Reconcile generated commute windows against effective actual sleep."""
    now = now or datetime.now(timezone.utc)
    sleeps = store.connection.execute("""SELECT * FROM events WHERE user_id=? AND track='sleep'
        AND realization='actual' AND retracted=0 AND superseded_by IS NULL
        AND (created_in_txn IS NULL OR created_in_txn IN (SELECT txn_id FROM proposal_batches WHERE status='accepted'))
        AND (retired_in_txn IS NULL OR retired_in_txn NOT IN (SELECT txn_id FROM proposal_batches WHERE status='accepted'))""", (user_id,)).fetchall()
    intervals = [(_parse(row['occurred_at_utc']), _resolve_sleep_local('sleep_end', json.loads(row['payload'])['sleep_end'], row['timezone'])) for row in sleeps]
    rows = store.connection.execute("""SELECT * FROM events WHERE user_id=? AND title='commute' AND track='schedule'
        AND source='system' AND superseded_by IS NULL ORDER BY rowid DESC""", (user_id,)).fetchall()
    latest = {}
    for row in rows:
        payload = json.loads(row['payload'])
        if payload.get('generated_commute') and payload.get('commute_key'):
            latest.setdefault(payload['commute_key'], (row, payload))
    changed = 0
    for row, payload in latest.values():
        if row['retracted'] and row['retracted_reason'] != 'actual sleep commute trim':
            continue
        base_start = _parse(payload.get('untrimmed_start_utc') or row['occurred_at_utc'])
        base_end = _parse(payload.get('untrimmed_end_utc') or payload['end_utc'])
        start, end = base_start, base_end
        overlapping = [(a, b) for a, b in intervals if a < base_end and b > base_start]
        if overlapping:
            if payload.get('side') == 'before':
                start = min(base_end, max(base_start, max(b for _, b in overlapping)))
            else:
                end = max(base_start, min(base_end, min(a for a, _ in overlapping)))
        visible = start < end
        if not row['retracted'] and visible and _parse(row['occurred_at_utc']) == start and _parse(payload['end_utc']) == end:
            continue
        if row['retracted'] and not visible:
            continue
        if not row['retracted']:
            store.set_retracted(row['event_id'], now, 'actual sleep commute trim')
        if visible:
            zone = ZoneInfo(row['timezone'])
            updated = {**payload, 'untrimmed_start_utc': base_start.isoformat(), 'untrimmed_end_utc': base_end.isoformat(), 'end_utc': end.isoformat()}
            event = ScheduleEvent(event_id=str(uuid.uuid4()), track='schedule', user_id=user_id,
                dedupe_key='commute-trim:'+uuid.uuid4().hex, occurred_at_utc=start,
                local_time=start.astimezone(zone).replace(tzinfo=None).isoformat(), timezone=zone.key,
                source='system', status='planned', title='commute', confirmation='accepted', realization='planned',
                payload=updated, created_at=now, updated_at=now)
            store.insert(event)
            store.add_lineage(user_id, event.event_id, [row['event_id']])
        changed += 1
    return changed


def sync_commutes(store, user_id, profile, now=None, *, include_history=False):
    now = now or datetime.now(timezone.utc)
    home = str(profile.get('homeport') or '').strip().upper()
    if not home:
        return {'status': 'homeport_required', 'created': 0, 'removed': 0, 'missing_airports': []}
    if home not in AIRPORT_TIMEZONES:
        raise ValueError('无法识别基地机场代码')
    minutes = profile.get('commute_minutes')
    minutes = 120 if minutes is None else float(minutes)
    if not 0 <= minutes <= 1440:
        raise ValueError('通勤时间必须在 0 至 1440 分钟之间')
    desired, missing = {}, []
    rows = store.connection.execute("SELECT * FROM events WHERE user_id=? AND track='schedule' AND title='flight' AND retracted=0 AND superseded_by IS NULL", (user_id,)).fetchall()
    for row in rows:
        data = json.loads(row['payload']); raw = data.get('roster_item', {}).get('raw', {})
        origin = str(data.get('origin') or raw.get('origin') or '').upper()
        destination = str(data.get('destination') or raw.get('destination') or '').upper()
        if not origin or not destination:
            missing.append(row['event_id']); continue
        duty_start = _parse(row['duty_start_utc']) if row['duty_start_utc'] else None
        duty_end = _parse(row['duty_end_utc']) if row['duty_end_utc'] else None
        before = timedelta(minutes=minutes if origin == home else 120)
        after = timedelta(minutes=minutes if destination == home else 120)
        spans = []
        if duty_start:
            try:
                departure = _parse(data['departure'])
            except (KeyError, ValueError, TypeError):
                missing.append(row['event_id'])
            else:
                if departure >= duty_start:
                    spans.append(('before', duty_start-before, departure))
                else:
                    missing.append(row['event_id'])
        if duty_end:
            try:
                arrival = _parse(data['arrival'])
            except (KeyError, ValueError, TypeError):
                missing.append(row['event_id'])
            else:
                if arrival <= duty_end:
                    spans.append(('after', arrival, duty_end+after))
                else:
                    missing.append(row['event_id'])
        for side, start, end in spans:
            if (not include_history and start <= now) or start == end:
                continue
            key = f"commute:{row['event_id']}:{side}"
            desired[key] = (start, end, row['event_id'], side, origin, destination, False)
    trim_commutes_for_actual_sleep(store, user_id, now)
    existing = store.connection.execute("SELECT * FROM events WHERE user_id=? AND title='commute' AND track='schedule' AND source='system' AND retracted=0 AND superseded_by IS NULL", (user_id,)).fetchall()
    created = removed = 0
    for row in existing:
        payload = json.loads(row['payload'])
        if not payload.get('generated_commute') or (not include_history and _parse(row['occurred_at_utc']) <= now):
            continue
        key = payload.get('commute_key')
        spec = desired.get(key)
        if spec and _parse(payload.get('untrimmed_start_utc') or row['occurred_at_utc']) == spec[0] and _parse(payload.get('untrimmed_end_utc') or payload['end_utc']) == spec[1] and payload.get('duty_boundary_estimated') == spec[6]:
            desired.pop(key); continue
        store.set_retracted(row['event_id'], now, 'commute recalculated')
        removed += 1
    # A fully trimmed event stays retracted until its actual sleep is changed.
    for key in list(desired):
        tombstone = store.connection.execute("SELECT * FROM events WHERE user_id=? AND title='commute' AND json_extract(payload,'$.commute_key')=? ORDER BY rowid DESC LIMIT 1", (user_id, key)).fetchone()
        if tombstone and tombstone['retracted'] and tombstone['retracted_reason'] == 'actual sleep commute trim':
            payload = json.loads(tombstone['payload']); spec = desired[key]
            if _parse(payload.get('untrimmed_start_utc') or tombstone['occurred_at_utc']) == spec[0] and _parse(payload.get('untrimmed_end_utc') or payload['end_utc']) == spec[1]:
                desired.pop(key)
    for key, (start, end, flight_id, side, origin, destination, estimated) in desired.items():
        zone = ZoneInfo(AIRPORT_TIMEZONES[home] if (side == 'before' and origin == home) or (side == 'after' and destination == home) else AIRPORT_TIMEZONES.get(origin if side == 'before' else destination, 'UTC'))
        event = ScheduleEvent(event_id=str(uuid.uuid4()), track='schedule', user_id=user_id,
            dedupe_key=key+':'+uuid.uuid4().hex, occurred_at_utc=start, local_time=start.astimezone(zone).replace(tzinfo=None).isoformat(), timezone=zone.key,
            source='system', status='planned', title='commute', confirmation='accepted', realization='planned',
            payload={'generated_commute': True, 'commute_key': key, 'flight_id': flight_id, 'side': side, 'end_utc': end.isoformat(), 'duty_boundary_estimated': estimated}, created_at=now, updated_at=now)
        store.insert(event); created += 1
    trimmed = trim_commutes_for_actual_sleep(store, user_id, now)
    return {'status': 'generated', 'created': created, 'removed': removed, 'trimmed': trimmed, 'missing_airports': missing}
