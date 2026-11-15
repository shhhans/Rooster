"""Declared photopic scenarios; unknown location is not inferred from a flight."""
from zoneinfo import ZoneInfo
import math
from .normalize import dt


def light_at(at, asleep, events, profile, factor=1.):
    active = [e for e in events if dt(e['start_utc']) <= at < dt(e['end_utc'])]
    measurements = [e for e in active if e['kind'] == 'light' and e.get('metric', 'photopic_lux') == 'photopic_lux']
    if measurements:
        value = float(measurements[0]['lux'])
        if value < 0 or not math.isfinite(value):
            raise ValueError('negative light observation')
        return value, 'observed'
    if asleep:
        return 0., 'assumed_dark'
    locations = [e for e in active if e['kind'] in ('location', 'environment')]
    # A resource says where sleep is allowed, not necessarily where the person is.
    explicit=[e for e in locations if e['kind']=='location']
    choices=explicit or locations
    alternatives={(e.get('location'),e.get('timezone')) for e in choices}
    location,zone=next(iter(alternatives)) if len(alternatives)==1 else (None,None)
    if location == 'aircraft' or any(e['kind'] == 'flight' for e in active):
        return 50.*factor, 'assumed_aircraft'
    if any(e['kind'] in ('duty', 'commute', 'preparation') for e in active):
        return 150.*factor, 'assumed_work'
    if location in ('home', 'hotel') and zone:
        hour = at.astimezone(ZoneInfo(zone)).hour
        return (250. if 7 <= hour < 19 else 50.)*factor, 'assumed_indoor'
    return 100.*factor, 'assumed_location_unknown'
