"""Neutral time and interval utilities shared across modules.

Lead-managed shared contract (module decoupling contract): only time/interval
parsing, validation and arithmetic live here. No scheduling policy, model
math, or product logic. The implementations are verbatim moves from
``app.sleep_model.normalize`` so behavior, digests, and error strings are
unchanged; the historical import paths re-export them.
"""
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import math


def dt(value):
    value = datetime.fromisoformat(value.replace('Z', '+00:00')) if isinstance(value, str) else value
    if value.tzinfo is None:
        raise ValueError('timezone-aware instant required')
    return value.astimezone(timezone.utc)


def stamp(value):
    return dt(value).isoformat().replace('+00:00', 'Z')


def interval(e):
    a, b = dt(e['start_utc']), dt(e['end_utc'])
    if b <= a:
        raise ValueError('invalid interval: ' + str(e.get('id')))
    return a, b


def overlap(a, b):
    return max(a[0], b[0]) < min(a[1], b[1])


def merge(pieces):
    out = []
    for a, b in sorted(pieces):
        if b <= a:
            continue
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def local_instant(day, clock, zone, fold=None):
    """Strict wall-clock conversion: reject DST gaps and unqualified folds."""
    naive = datetime.fromisoformat(str(day) + 'T' + clock)
    tz = ZoneInfo(zone)
    options = [naive.replace(tzinfo=tz, fold=f) for f in (0, 1)]
    valid = [v for v in options if v.astimezone(timezone.utc).astimezone(tz).replace(tzinfo=None) == naive]
    if not valid:
        raise ValueError('nonexistent local time')
    if valid[0].utcoffset() != valid[-1].utcoffset() and fold is None:
        raise ValueError('ambiguous local time requires fold')
    return dt(options[fold or 0])


def clock_hours(value):
    if isinstance(value, (int, float)):
        result = float(value)
    else:
        h, m = value.split(':')
        if not (h.isdigit() and m.isdigit() and 0<=int(h)<24 and 0<=int(m)<60):
            raise ValueError('invalid habit clock')
        result = int(h) + int(m)/60
    if not math.isfinite(result) or not 0 <= result < 24:
        raise ValueError('invalid habit clock')
    return result
