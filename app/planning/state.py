"""Event-driven planning state. Observations never inherit predicted sleep tails."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def instant(value):
    dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if dt.tzinfo is None:
        raise ValueError('timezone-aware instant required')
    return dt.astimezone(timezone.utc)


def stamp(value):
    return value.astimezone(timezone.utc).isoformat().replace('+00:00', 'Z')


class PlanningState:
    def __init__(self, snapshot):
        self.snapshot = deepcopy(snapshot)
        self.snapshot.setdefault('active_plan', None)
        self.snapshot.setdefault('sessions', {})
        self.snapshot.setdefault('applied_event_ids', [])

    def apply(self, event, *, validate_plan=True):
        """Atomic reducer used by both the live service and archived replay."""
        old = self.snapshot
        if event['id'] in old['applied_event_ids']:
            return False
        if event.get('expected_revision', old['revision']) != old['revision']:
            raise ValueError('stale event revision')
        at = instant(event['at_utc'])
        if at < instant(old['now_utc']):
            raise ValueError('reported time cannot move backwards')
        new = deepcopy(old)
        action = event['action']
        if action == 'upsert':
            for item in event['events']:
                if instant(item['end_utc']) <= instant(item['start_utc']):
                    raise ValueError('invalid interval')
                if item['kind'] == 'sleep_actual' and instant(item['end_utc']) > at:
                    raise ValueError('future sleep cannot be recorded as fact')
                new['events'] = [e for e in new['events'] if e['id'] != item['id']] + [deepcopy(item)]
        elif action == 'accept':
            from .engine import check_plan
            if event['base_revision'] != old['revision']:
                raise ValueError('stale plan revision')
            if validate_plan:
                checks = check_plan(old, event['plan']['sleep_blocks'])
                if any(c.get('status') == 'fail' for c in checks):
                    raise ValueError('cannot accept invalid plan')
            new['active_plan'] = deepcopy(event['plan'])
        elif action == 'sleep_started':
            if any(s['actual_end_utc'] is None for s in new['sessions'].values()):
                raise ValueError('sleep already in progress')
            if instant(event['actual_start_utc']) > at:
                raise ValueError('future actual start')
            new['sessions'][event['session_id']] = {
                'id': event['session_id'], 'actual_start_utc': event['actual_start_utc'],
                'actual_end_utc': None, 'expected_end_utc': event['expected_end_utc'],
                'planned_block_id': event['planned_block_id'],
            }
        elif action == 'wake':
            session = new['sessions'][event['session_id']]
            if session['actual_end_utc'] is not None:
                raise ValueError('session already completed')
            end = instant(event['actual_end_utc'])
            if not instant(session['actual_start_utc']) < end <= at:
                raise ValueError('invalid actual end')
            session['actual_end_utc'] = event['actual_end_utc']
            new['events'].append({'id': session['id'] + '-actual', 'kind': 'sleep_actual',
                'start_utc': session['actual_start_utc'], 'end_utc': event['actual_end_utc'],
                'timezone': new['profile']['home_timezone'], 'source': 'simulated_observation'})
            if new['active_plan']:
                new['active_plan']['sleep_blocks'] = [b for b in new['active_plan']['sleep_blocks']
                    if b['id'] != session['planned_block_id']]
        elif action != 'advance':
            raise ValueError('unknown planning event action')
        new['now_utc'] = event['at_utc']
        new['revision'] += 1
        new['applied_event_ids'].append(event['id'])
        self.snapshot = new
        return True
