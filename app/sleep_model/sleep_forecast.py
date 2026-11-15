"""Opportunities, forecasts, and observations never share a semantic label."""
from datetime import timedelta
from .normalize import interval, stamp, merge


def forecast_sleep(blocks, config):
    result = []
    for b in blocks:
        a, end = interval(b)
        start = a + timedelta(minutes=config['latency_minutes'])
        end -= timedelta(minutes=config['early_minutes'])
        if end > start:
            result.append({'id': b['id']+'-predicted', 'opportunity_id': b['id'],
                           'start_utc': stamp(start), 'end_utc': stamp(end),
                           'role': b.get('role', 'main'), 'source': 'predicted',
                           'episode_id': b.get('parent_sleep_episode_id', b['id'])})
    # Adjacent sleep without a wake interval is one physiological sleep episode.
    return result


def sleep_hours(records, start, end):
    return sum((b-a).total_seconds()/3600 for a,b in merge(
        (max(start,a),min(end,b)) for e in records for a,b in [interval(e)] if a < end and b > start))
