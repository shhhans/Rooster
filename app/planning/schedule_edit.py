"""Isolated batch editing prototype with observation-based feedback."""
from copy import deepcopy
from uuid import uuid4
from zoneinfo import ZoneInfo

from app.contracts.time import dt, stamp
from app.planning.observation import KINDS, _event, _instant


WRITABLE = {track: kinds for track, kinds in KINDS.items()
            if track in ('activity', 'environment', 'intake')}
FIELDS = {'kind', 'start', 'end', 'timezone', 'title', 'role', 'crew_rest_id', 'parent_event_ids'}


def edit_schedule(snapshot, blocks, operations, display_timezone):
    """Return a new draft and feedback; never mutate inputs, even on failure.

    This experimental interface edits suggested blocks only. Times are ISO
    strings interpreted in each event's timezone when no offset is supplied.
    """
    if not operations:
        raise ValueError('operations are required')
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

    zone = ZoneInfo(display_timezone)
    def normalized(block):
        if block is None:
            return None
        track = next((name for name, kinds in WRITABLE.items()
                      if block.get('kind', 'sleep_plan') in kinds), 'activity')
        return _event(block, track, zone, planned=True)
    return {'blocks': working,
            'changes': [{'op': c['op'], 'id': c['id'], 'before': normalized(c['before']),
                         'after': normalized(c['after'])} for c in changes]}
