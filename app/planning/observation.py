"""Read-only views of tracks and events for planning experiments."""
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from app.contracts.time import dt, local_instant, stamp

TRACKS = ('duty', 'activity', 'environment', 'intake', 'location')
KINDS = {
    'duty': {'duty', 'flight', 'commute', 'preparation', 'protected_rest', 'commitment', 'crew_rest'},
    'activity': {'sleep_plan', 'sleep_actual', 'wake_actual', 'sleep', 'exercise', 'meal'},
    'environment': {'environment', 'light', 'sunglasses', 'temperature'},
    'intake': {'caffeine', 'melatonin', 'intake'},
    'location': {'location'},
}


def _instant(value, timezone):
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if parsed.tzinfo is None:
        return local_instant(parsed.date(), parsed.time().isoformat(), timezone)
    return dt(parsed)


def _times(start, end, zone):
    return {'start_utc': stamp(start), 'end_utc': stamp(end) if end is not None else None,
            'start_local': dt(start).astimezone(zone).isoformat(),
            'end_local': dt(end).astimezone(zone).isoformat() if end is not None else None}


def _event(item, track, zone, planned=False):
    start = item.get('start_utc') or item.get('occurred_at_utc')
    if not start:
        return None
    end = item.get('end_utc')
    beginning = dt(start)
    ending = dt(end) if end else beginning
    if ending < beginning:
        raise ValueError('event end is before start: ' + str(item.get('id')))
    output = {'id': item.get('id') or item.get('event_id'), 'track': track,
              'kind': item.get('kind', 'sleep_plan'),
              **_times(beginning, ending if end else None, zone),
              'display_timezone': zone.key, 'event_timezone': item.get('timezone'),
              'role': item.get('role'),
              'status': item.get('status', 'suggested' if planned else 'unspecified'),
              'source': item.get('source', 'planning_suggestion' if planned else 'unspecified')}
    for key in ('origin', 'destination', 'flight_number', 'duty_id',
                'label', 'code', 'location', 'airport', 'availability', 'confirmed', 'accepted',
                'origin_event_id', 'title', 'source_quote', 'source_quotes', 'change',
                'crew_rest_id', 'authorized', 'authorized_event_ids', 'parent_event_ids'):
        if key in item:
            output[key] = item[key]
    for boundary in ('start', 'end'):
        original = item.get('original_' + boundary + '_utc')
        if original:
            output['original_' + boundary + '_utc'] = stamp(dt(original))
            output['original_' + boundary + '_local'] = dt(original).astimezone(zone).isoformat()
    if item.get('arrival_timezone'):
        output['destination_timezone'] = item['arrival_timezone']
    return output


def _events(snapshot, blocks, zone):
    result = []
    flights_by_duty = {}
    for flight in snapshot.get('events', []):
        duty_id = flight.get('duty_id')
        if flight.get('kind') != 'flight' or not duty_id or not flight.get('end_utc'):
            continue
        previous = flights_by_duty.get(duty_id)
        if previous is None or dt(flight['end_utc']) > dt(previous['end_utc']):
            flights_by_duty[duty_id] = flight
    for item in snapshot.get('events', []):
        for track, kinds in KINDS.items():
            if item.get('kind') in kinds:
                event = _event(item, track, zone)
                if event:
                    result.append(event)
                break
    for item in blocks:
        track = next((name for name, kinds in KINDS.items() if item.get('kind') in kinds), 'activity')
        event = _event(item, track, zone, planned=True)
        if event:
            result.append(event)
    for event in result:
        flight = flights_by_duty.get(event['id']) if event['kind'] == 'duty' else None
        if flight and flight.get('arrival_timezone'):
            event['destination_timezone'] = flight['arrival_timezone']
    return sorted(result, key=lambda item: (item['start_utc'], item['id'] or ''))


def _overlaps(item, start, end):
    beginning = dt(item['start_utc'])
    if item['end_utc'] is None:
        return start <= beginning < end
    return beginning < end and dt(item['end_utc']) > start


def observe_track(snapshot, blocks, tracks, start, end, timezone):
    """Return events intersecting [start,end), preserving full event boundaries."""
    if not tracks or any(track not in TRACKS for track in tracks):
        raise ValueError('tracks must contain registered track names')
    zone = ZoneInfo(timezone)
    start, end = _instant(start, timezone), _instant(end, timezone)
    if end <= start:
        raise ValueError('end must be later than start')
    selected = []
    for item in _events(snapshot, blocks, zone):
        if item['track'] not in tracks or not _overlaps(item, start, end):
            continue
        item['cut_by_window_start'] = dt(item['start_utc']) < start
        item['cut_by_window_end'] = bool(item['end_utc'] and dt(item['end_utc']) > end)
        selected.append(item)
    sleep = [x for x in selected if x['track'] == 'activity'
             and x['kind'] in ('sleep', 'sleep_plan') and x['end_utc']]
    groups = []
    for item in sleep:
        if groups and groups[-1]['end_utc'] == item['start_utc']:
            groups[-1]['ids'].append(item['id'])
            groups[-1]['end_utc'] = item['end_utc']
        else:
            groups.append({'ids': [item['id']], 'start_utc': item['start_utc'],
                           'end_utc': item['end_utc']})
    for group in groups:
        group['minutes'] = (dt(group['end_utc']) - dt(group['start_utc'])).total_seconds() / 60
        group.update(_times(dt(group['start_utc']), dt(group['end_utc']), zone))
    activities = [item for item in selected if item['track'] == 'activity' and item['end_utc']]
    activity_overlaps = []
    for index, first in enumerate(activities):
        for second in activities[index + 1:]:
            overlap_start = max(dt(first['start_utc']), dt(second['start_utc']))
            overlap_end = min(dt(first['end_utc']), dt(second['end_utc']))
            if overlap_start < overlap_end:
                activity_overlaps.append({'ids': [first['id'], second['id']],
                    **_times(overlap_start, overlap_end, zone),
                    'minutes': (overlap_end - overlap_start).total_seconds() / 60,
                    'statuses': [first['status'], second['status']]})
    return {'window': {**_times(start, end, zone), 'display_timezone': timezone,
                       'end_exclusive': True},
            'tracks': tracks, 'events': selected,
            'contiguous_planned_sleep': [group for group in groups if len(group['ids']) > 1],
            'contiguous_scope': '仅合并与本窗口相交并返回的睡眠建议；未包含窗口外相接事项。',
            'activity_overlaps': activity_overlaps,
            'coverage': '没有已知占用的未来时间默认可自由支配；本结果仅涵盖所选轨道。'}


def observe_event(snapshot, blocks, event_id, before_hours, after_hours, tracks, timezone):
    """Return one event and its explicitly bounded cross-track context."""
    if before_hours < 0 or after_hours < 0:
        raise ValueError('before_hours and after_hours must be nonnegative')
    zone = ZoneInfo(timezone)
    events = _events(snapshot, blocks, zone)
    matches = [item for item in events if item['id'] == event_id]
    if len(matches) != 1:
        raise ValueError('event_id must resolve to exactly one event')
    item = matches[0]
    beginning = dt(item['start_utc'])
    ending = dt(item['end_utc']) if item['end_utc'] else beginning
    start, end = beginning - timedelta(hours=before_hours), ending + timedelta(hours=after_hours)
    if end == start:
        raise ValueError('a point event needs a nonzero context window')
    context = observe_track(snapshot, blocks, tracks, stamp(start), stamp(end), timezone)
    neighbors = {}
    for track in tracks:
        peers = [entry for entry in context['events'] if entry['track'] == track and entry['id'] != event_id]
        previous = [entry for entry in peers if dt(entry['end_utc'] or entry['start_utc']) <= beginning]
        following = [entry for entry in peers if dt(entry['start_utc']) >= ending]
        concurrent = [entry['id'] for entry in peers if _overlaps(entry, beginning, ending)] if ending > beginning else [
            entry['id'] for entry in peers if dt(entry['start_utc']) <= beginning
            and (dt(entry['end_utc']) > beginning if entry['end_utc'] else dt(entry['start_utc']) == beginning)]
        previous_end = max((dt(entry['end_utc'] or entry['start_utc']) for entry in previous), default=None)
        next_start = min((dt(entry['start_utc']) for entry in following), default=None)
        neighbors[track] = {
            'previous': [{'id': entry['id'], 'gap_minutes': (beginning - previous_end).total_seconds()/60}
                         for entry in previous if dt(entry['end_utc'] or entry['start_utc']) == previous_end],
            'next': [{'id': entry['id'], 'gap_minutes': (next_start - ending).total_seconds()/60}
                     for entry in following if dt(entry['start_utc']) == next_start],
            'overlapping_ids': concurrent,
            'scope': '仅在此次上下文窗口内；空列表不表示窗口外没有事项'}
    return {'event': item,
            'windows': {'display_timezone': timezone,
                        'before': _times(start, beginning, zone),
                        'event': _times(beginning, ending, zone),
                        'after': _times(ending, end, zone)},
            'context': context, 'nearest_by_track': neighbors}
