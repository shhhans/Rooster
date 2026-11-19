"""Materialize the user's home sleep routine as effective baseline events."""
from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
import json
from zoneinfo import ZoneInfo

from app.domain.timeline.models import ScheduleEvent
from app.domain.timeline.errors import InvalidInputError, UnknownTimezoneError
from app.domain.timeline.sleep_track import _resolve_pair
from app.domain.timeline.store import EventStore


def generate_baseline(store: EventStore, user_id: str, first_day: date, last_day: date, profile: dict | None) -> dict:
    if not profile or not profile.get('home_timezone') or profile.get('preferred_sleep_start_hour') is None or profile.get('sleep_need_hours') is None:
        return {'status': 'profile_required', 'created': 0}
    zone = profile['home_timezone']
    ZoneInfo(zone)
    hour, duration = float(profile['preferred_sleep_start_hour']), float(profile['sleep_need_hours'])
    if not 0 <= hour < 24 or not 0 < duration <= 24 or last_day < first_day:
        raise ValueError('invalid baseline profile or date range')
    minutes = round(hour * 60)
    if minutes >= 1440:
        raise ValueError('invalid bedtime')
    count, skipped = 0, []
    with store.transaction():
        old_profile = store.connection.execute('SELECT profile_json FROM sleep_profiles WHERE user_id=?', (user_id,)).fetchone()
        if old_profile:
            previous = json.loads(old_profile['profile_json'])
            profile = {**previous, **profile}
            if profile.get('homeport'):
                profile['home_timezone'] = previous.get('home_timezone', profile['home_timezone'])
        profile_changed = bool(old_profile and json.loads(old_profile['profile_json']) != profile)
        if profile_changed:
            store._bump_input_revision(user_id, first_day.isoformat())
        store.connection.execute('INSERT INTO sleep_profiles VALUES (?, ?, ?) ON CONFLICT(user_id) DO UPDATE SET profile_json=excluded.profile_json, updated_at=excluded.updated_at',
                                 (user_id, json.dumps(profile), datetime.now(timezone.utc).isoformat()))
        progress = store.connection.execute('SELECT generated_through FROM baseline_generation_state WHERE user_id=?', (user_id,)).fetchone()
        if progress:
            day = date.fromisoformat(progress['generated_through']) + timedelta(days=1)
        else:
            latest = store.connection.execute("SELECT MAX(substr(dedupe_key, 10, 10)) AS last_day FROM events WHERE user_id=? AND track='sleep' AND source='baseline' AND dedupe_key GLOB 'baseline:????-??-??'", (user_id,)).fetchone()['last_day']
            day = date.fromisoformat(latest) + timedelta(days=1) if latest else first_day
        while day <= last_day:
            # The key persists even if the event is later retired by a proposal.
            key = 'baseline:' + day.isoformat()
            if store.get_by_dedupe_key(user_id, 'sleep', key) is None:
                local_start = datetime.combine(day, time()) + timedelta(minutes=minutes)
                local_end = local_start + timedelta(hours=duration)
                start_text, end_text = local_start.isoformat(), local_end.isoformat()
                try:
                    start_utc, _ = _resolve_pair(start_text, end_text, zone)
                except (InvalidInputError, UnknownTimezoneError):
                    skipped.append(day.isoformat())
                    day += timedelta(days=1)
                    continue
                now = datetime.now(timezone.utc)
                import uuid
                event = ScheduleEvent(event_id=str(uuid.uuid4()), track='sleep', user_id=user_id,
                    dedupe_key=key, occurred_at_utc=start_utc, local_time=start_text,
                    timezone=zone, source='baseline', status='planned', title='sleep',
                    payload={'sleep_start': start_text, 'sleep_end': end_text},
                    confirmation='proposed', realization='planned', created_at=now, updated_at=now)
                store.insert(event)
                count += 1
            day += timedelta(days=1)
        if day > last_day and (not progress or last_day > date.fromisoformat(progress['generated_through'])):
            generated_through = max(last_day, date.fromisoformat(latest)) if not progress and latest else last_day
            store.connection.execute('INSERT INTO baseline_generation_state VALUES (?, ?) ON CONFLICT(user_id) DO UPDATE SET generated_through=excluded.generated_through', (user_id, generated_through.isoformat()))
    return {'status': 'generated', 'created': count, 'profile_changed': profile_changed, 'skipped_dates': skipped}
