"""Isolated batch editing prototype with observation-based feedback."""
from copy import deepcopy
from datetime import timedelta
from uuid import uuid4
from zoneinfo import ZoneInfo

from app.contracts.time import dt, stamp
from app.planning.observation import KINDS, TRACKS, _event, _events, _instant, _times, observe_event, observe_track


WRITABLE = {track: kinds for track, kinds in KINDS.items()
            if track in ('activity', 'environment', 'intake')}
FIELDS = {'kind', 'start', 'end', 'timezone', 'title', 'role', 'crew_rest_id', 'parent_event_ids'}


def edit_schedule(snapshot, blocks, operations, display_timezone, context_hours=24):
    """Return a new draft and feedback; never mutate inputs, even on failure.

    This experimental interface edits suggested blocks only. Times are ISO
    strings interpreted in each event's timezone when no offset is supplied.
    """
    if not operations or context_hours <= 0:
        raise ValueError('operations and a positive context_hours are required')
    working = deepcopy(blocks)
    original = {b['id']: deepcopy(b) for b in blocks}
    if len(original) != len(blocks):
        raise ValueError('duplicate block IDs')
    changes, touched = [], set()
    for op in operations:
        action = op.get('op')
        if action not in ('create', 'update', 'delete'):
            raise ValueError('unknown operation')
        if action == 'create':
            before = None
            event_id = 'suggestion-' + uuid4().hex
        else:
            event_id = op.get('event_id')
            if event_id not in original or event_id in touched:
                raise ValueError('unknown or duplicate target: ' + str(event_id))
            before = original[event_id]
            if before.get('status', 'suggested') != 'suggested':
                raise ValueError('only suggested events can be edited')
            touched.add(event_id)
        if action == 'delete':
            working = [b for b in working if b['id'] != event_id]
            changes.append({'op': action, 'id': event_id, 'before': before, 'after': None})
            continue
        fields = op.get('fields', {})
        if not isinstance(fields, dict) or set(fields) - FIELDS:
            raise ValueError('invalid fields')
        after = deepcopy(before) if before else {'id': event_id, 'status': 'suggested', 'source': 'planning_suggestion'}
        for key in ('kind', 'timezone', 'title', 'role', 'crew_rest_id', 'parent_event_ids'):
            if key in fields:
                after[key] = fields[key]
        if 'parent_event_ids' in fields:
            parents = fields['parent_event_ids']
            if (not isinstance(parents, list) or any(not isinstance(p, str) or not p for p in parents)
                    or len(set(parents)) != len(parents)):
                raise ValueError('parent_event_ids must contain unique nonempty IDs')
        zone = after.get('timezone')
        if not zone:
            raise ValueError('event timezone is required')
        for key in ('start', 'end'):
            if key in fields:
                after[key + '_utc'] = stamp(_instant(fields[key], zone))
        if not after.get('start_utc') or not after.get('end_utc'):
            raise ValueError('start and end are required')
        if dt(after['end_utc']) <= dt(after['start_utc']):
            raise ValueError('end must be after start')
        kind = after.get('kind', 'sleep_plan')
        if not any(kind in kinds for kinds in WRITABLE.values()) or kind in ('sleep_actual', 'wake_actual'):
            raise ValueError('kind is not writable')
        after['kind'] = kind
        if before:
            working = [after if b['id'] == event_id else b for b in working]
        else:
            working.append(after)
        changes.append({'op': action, 'id': event_id, 'before': before, 'after': deepcopy(after)})

    # Check only overlaps introduced by this transaction; historical conflicts
    # elsewhere must not prevent repairing one part of an existing draft.
    changed_ids = {c['id'] for c in changes if c['after']}
    activities = [b for b in _events(snapshot, working, ZoneInfo(display_timezone))
                  if b['track'] == 'activity' and b['end_utc']]
    for index, left in enumerate(activities):
        for right in activities[index + 1:]:
            if not ({left['id'], right['id']} & changed_ids):
                continue
            if dt(left['start_utc']) < dt(right['end_utc']) and dt(right['start_utc']) < dt(left['end_utc']):
                raise ValueError('primary activity overlap: ' + left['id'] + ', ' + right['id'])

    contexts = []
    for change in changes:
        if change['after']:
            view = observe_event(snapshot, working, change['id'], context_hours, context_hours,
                                 list(TRACKS), display_timezone)
            event = view['event']
            overlaps = []
            for peer in view['context']['events']:
                if peer['id'] == event['id'] or not peer['end_utc']:
                    continue
                start = max(dt(event['start_utc']), dt(peer['start_utc']))
                end = min(dt(event['end_utc']), dt(peer['end_utc']))
                if start < end:
                    overlaps.append({'id': peer['id'], 'track': peer['track'],
                                     **_times(start, end, ZoneInfo(display_timezone)),
                                     'minutes': (end - start).total_seconds() / 60})
            contexts.append({'id': change['id'], 'event': view['event'],
                             'nearest_by_track': view['nearest_by_track'],
                             'overlaps': overlaps,
                             'context': view['context']})
        else:
            before = change['before']
            start = dt(before['start_utc']) - timedelta(hours=context_hours)
            end = dt(before['end_utc']) + timedelta(hours=context_hours)
            contexts.append({'id': change['id'], 'deleted': True,
                             'context': observe_track(snapshot, working, list(TRACKS),
                                                      stamp(start), stamp(end), display_timezone)})
    zone = ZoneInfo(display_timezone)
    def normalized(block):
        if block is None:
            return None
        track = next((name for name, kinds in WRITABLE.items()
                      if block.get('kind', 'sleep_plan') in kinds), 'activity')
        return _event(block, track, zone, planned=True)
    return {'blocks': working,
            'changes': [{'op': c['op'], 'id': c['id'], 'before': normalized(c['before']),
                         'after': normalized(c['after'])} for c in changes],
            'local_context': contexts}
