"""UTC normalization, revision resolution and observed/unknown separation.

Neutral time/interval helpers live in the Lead-managed shared contract
``app.contracts.time`` and are re-exported here so every historical import
path (``app.sleep_model.normalize.dt`` etc.) keeps working unchanged.
"""
from copy import deepcopy
from zoneinfo import ZoneInfo

from app.contracts.time import clock_hours, dt, interval, local_instant, merge, overlap, stamp

__all__ = ['clock_hours', 'dt', 'interval', 'local_instant', 'merge', 'normalize', 'overlap', 'stamp']


def normalize(snapshot, blocks, cutoff):
    snap = deepcopy(snapshot)
    profile = snap.setdefault('profile', {})
    ZoneInfo(profile.get('home_timezone', 'UTC'))
    bed_values = [clock_hours(profile[k]) for k in ('habit_bed_local', 'habit_bed_hour',
                  'preferred_bed_hour', 'preferred_sleep_start_hour') if profile.get(k) is not None]
    if bed_values and max(bed_values)-min(bed_values) > 1e-8:
        raise ValueError('conflicting_profile_fields')
    profile['_bed'] = bed_values[0] if bed_values else 23.
    profile['_wake'] = clock_hours(profile.get('habit_wake_local', 7.))
    if profile['_bed'] == profile['_wake']:
        raise ValueError('habit sleep must be shorter than 24 hours')
    if not 0 < profile.get('sleep_need_hours', 8.) <= 24:
        raise ValueError('invalid sleep need')
    assumptions = ['population parameters; no individual physiological calibration']
    if not bed_values:
        assumptions.append('habit_bed_local assumed 23:00')
    if 'habit_wake_local' not in profile:
        assumptions.append('habit_wake_local assumed 07:00')
    # Keep only visible revisions, remove explicitly superseded ids.
    by_id = {}
    superseded = set()
    records = list(snap.get('events', [])) + list(snap.get('history', [])) + list(snap.get('location_timeline', []))
    for i, raw in enumerate(records):
        e = deepcopy(raw)
        if e.get('observed_at_utc') and dt(e['observed_at_utc']) > cutoff:
            continue
        e.setdefault('id', f'input-{i}')
        e.setdefault('kind', 'sleep_actual' if e.get('state') == 'asleep' else 'wake_actual' if e.get('state') == 'awake' else 'location')
        if e.get('supersedes_id'):
            superseded.add(e['supersedes_id'])
        old = by_id.get(e['id'])
        if old and old.get('revision', 0) == e.get('revision', 0) and old != e:
            raise ValueError('conflicting record revision: ' + e['id'])
        if not old or old.get('revision', 0) < e.get('revision', 0):
            by_id[e['id']] = e
    for session in snap.get('sessions', {}).values():
        if session.get('actual_end_utc') is None and dt(session['actual_start_utc']) < cutoff:
            by_id['session-'+session['id']] = {'id': 'session-'+session['id'], 'kind': 'sleep_actual',
                'start_utc': session['actual_start_utc'], 'end_utc': stamp(cutoff),
                'source': 'ongoing_observation', 'episode_id': session['id']}
    initial = snap.get('initial_state') or {}
    if initial.get('last_sleep_start_utc') and initial.get('last_sleep_end_utc'):
        if not any(e['kind'] == 'sleep_actual' for e in by_id.values()):
            by_id['initial-sleep'] = {'id': 'initial-sleep', 'kind': 'sleep_actual',
                'start_utc': initial['last_sleep_start_utc'], 'end_utc': initial['last_sleep_end_utc'], 'source': 'initial_record'}
    events = [e for key, e in by_id.items() if key not in superseded]
    for e in events:
        if e.get('end_utc') is None:
            e['end_utc'] = stamp(cutoff)
        interval(e)
        if e['kind'] in ('sleep_actual', 'wake_actual') and dt(e['end_utc']) > cutoff:
            raise ValueError('future actual observation: ' + e['id'])
    for kind in ('location', 'light'):
        chosen = [e for e in events if e['kind'] == kind]
        for i, e in enumerate(chosen):
            for other in chosen[i+1:]:
                if overlap(interval(e), interval(other)) and any(e.get(k) != other.get(k) for k in
                    (('timezone', 'location') if kind == 'location' else ('lux', 'metric'))):
                    raise ValueError('conflicting ' + kind + ' records')
    asleep = [e for e in events if e['kind'] == 'sleep_actual']
    awake = [e for e in events if e['kind'] == 'wake_actual']
    if any(overlap(interval(a), interval(b)) for a in asleep for b in awake):
        raise ValueError('conflicting sleep/wake observations')
    # An explicit waking time corrects the habit prior until the next observed sleep.
    # This is an inference, not a fabricated waking observation or extra coverage.
    for i, e in enumerate(sorted(asleep,key=lambda e:e['start_utc'])):
        a = dt(e['end_utc'])
        later = [dt(x['start_utc']) for x in asleep if dt(x['start_utc']) >= a]
        b = min(later+[cutoff])
        if b>a:
            events.append({'id':'inferred-awake-'+e['id'],'kind':'wake_inferred',
                           'start_utc':stamp(a),'end_utc':stamp(b),'source':'inferred_after_reported_wake'})
    if initial.get('last_wake_utc') and dt(initial['last_wake_utc']) < cutoff:
        a=dt(initial['last_wake_utc'])
        later=[dt(e['start_utc']) for e in asleep if dt(e['start_utc'])>=a]
        b=min(later+[cutoff])
        if b>a:
            events.append({'id':'initial-wake-anchor','kind':'wake_inferred',
                           'start_utc':stamp(a),'end_utc':stamp(b),'source':'initial_wake_anchor'})
    for b in blocks:
        interval(b)
    if any(overlap(interval(a), interval(b)) for i, a in enumerate(blocks) for b in blocks[i+1:]):
        raise ValueError('overlapping sleep opportunities')
    snap['events'] = sorted(events, key=lambda e: (e['start_utc'], e['id']))
    return snap, deepcopy(blocks), assumptions
