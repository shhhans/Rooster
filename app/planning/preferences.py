"""Structured intent loss and the fixed need/excess buckets.

Planning owns the preference policy: the loss, the weighting and the fixed
24 h need/excess buckets are product policy, not physiology. Moved from
``app.sleep_model.preferences`` (module decoupling contract step 3); the
historical import path is a lazy compatibility bridge with unchanged behaviour.

The only model helper used here was the sleep-hour booking used by the fixed
buckets; it is plain interval bookkeeping (clip, merge, sum) over the shared
time contract and is therefore planning-owned now.
"""
from datetime import timedelta
from zoneinfo import ZoneInfo

from app.contracts.time import clock_hours, dt, interval, merge


def _sleep_hours(records, start, end):
    """Sleep hours inside a window: clip the intervals, merge them, sum them."""
    return sum((b - a).total_seconds() / 3600 for a, b in merge(
        (max(start, a), min(end, b)) for e in records for a, b in [interval(e)]
        if a < end and b > start))


def evaluate_preferences(preferences, blocks, sleeps, previous=()):
    rows = []
    unsupported = []
    for p in preferences:
        kind = p.get('kind')
        if kind not in ('bedtime', 'continuous_sleep', 'total_sleep', 'free_window', 'wake_by', 'change_cost'):
            unsupported.append(p)
            continue
        a, b = dt(p['scope_start_utc']), dt(p['scope_end_utc'])
        if b <= a or p.get('weight', 1) <= 0:
            raise ValueError('invalid preference scope or weight')
        deficit = 0.
        unit = 'hours'
        chosen = [e for e in blocks if e.get('role', 'main') == 'main' and a <= dt(e['start_utc']) < b and
                  (not p.get('block_id') or p['block_id'] == e['id'])]
        if kind in ('continuous_sleep', 'total_sleep'):
            target = float(p['target'])
            if target <= 0:
                raise ValueError('positive sleep target required')
            intervals = merge((max(a, x), min(b, y)) for e in sleeps for x, y in [interval(e)] if x < b and y > a)
            amount = max(((y - x).total_seconds() / 3600 for x, y in intervals), default=0.) if kind == 'continuous_sleep' else _sleep_hours(sleeps, a, b)
            deficit = max(0., target - amount)
            loss = deficit / target
        elif kind == 'bedtime':
            target = clock_hours(p['target'])
            zone = ZoneInfo(p['timezone'])
            unit = 'minutes'
            if float(p.get('tolerance', 0)) < 0:
                raise ValueError('negative bedtime tolerance')
            differences = []
            for e in chosen:
                t = dt(e['start_utc']).astimezone(zone)
                hour = t.hour + t.minute / 60
                delta = abs(hour - target) % 24
                differences.append(max(0, min(delta, 24 - delta) * 60 - float(p.get('tolerance', 0))))
            deficit = max(differences) if differences else 180.
            loss = min(1., deficit / 180)
        elif kind == 'free_window':
            amount = _sleep_hours(blocks, a, b)
            deficit = amount
            loss = amount / ((b - a).total_seconds() / 3600)
        elif kind == 'wake_by':
            deadline = dt(p['target'])
            unit = 'minutes'
            deficit = max(0, (max(dt(e['end_utc']) for e in chosen) - deadline).total_seconds() / 60) if chosen else 120.
            loss = min(1, deficit / 120)
        else:
            old = {e['id']: e for e in previous}
            new = {e['id']: e for e in blocks}
            cost = []
            unit = 'normalized'
            for key in sorted(old.keys() | new.keys()):
                if key not in old or key not in new:
                    cost.append(1.)
                else:
                    cost.append(min(1., sum(abs((dt(old[key][k]) - dt(new[key][k])).total_seconds()) / 60
                                            for k in ('start_utc', 'end_utc')) / 120))
            deficit = sum(cost) / len(cost) if cost else 0.
            loss = deficit
        rows.append({'id': p.get('id', kind), 'kind': kind, 'weight': p.get('weight', 1.),
                     'loss': float(loss), 'shortfall': float(deficit), 'unit': unit})
    loss = sum(p['loss'] * p['weight'] for p in rows) / sum(p['weight'] for p in rows) if rows else 0.
    return {'items': rows, 'unsupported_preferences': unsupported, 'loss': loss}


def sleep_targets(sleeps, start, end, need=8.):
    cursor = start
    rows = []
    while cursor < end:
        right = min(end, cursor + timedelta(hours=24))
        length = (right - cursor).total_seconds() / 3600
        target = need * length / 24
        slept = _sleep_hours(sleeps, cursor, right)
        rows.append({'start_utc': cursor.isoformat(), 'hours': length, 'sleep_hours': slept,
                     'target_hours': target, 'partial': length < 24,
                     'deficit': max(0., target - slept) / target,
                     'excess': min(1., max(0., slept - target - 2 * length / 24) / target)})
        cursor = right
    return rows, sum(r['hours'] * r['deficit'] for r in rows) / sum(r['hours'] for r in rows), \
        sum(r['hours'] * r['excess'] for r in rows) / sum(r['hours'] for r in rows)
