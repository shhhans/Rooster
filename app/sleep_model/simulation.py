"""Deterministic event-split simulation with exact S and inertia updates."""
from datetime import timedelta
import math
import numpy as np
from scipy.signal import find_peaks
from . import circadian
from .homeostat import advance
from .light import light_at
from .normalize import dt, stamp, interval, merge


def habit_asleep(at, profile):
    from zoneinfo import ZoneInfo
    local = at.astimezone(ZoneInfo(profile.get('home_timezone', 'UTC')))
    hour = local.hour + local.minute/60 + local.second/3600
    bed, wake = profile['_bed'], profile['_wake']
    return bed <= hour < wake if bed < wake else hour >= bed or hour < wake


def simulate(state, start, end, sleeps, events, profile, config, *, history=False, markers=()):
    """sleeps are facts/predictions; unspecified history follows a tagged prior."""
    if end <= start:
        return [], dict(state)
    merged_sleep = merge(interval(e) for e in sleeps)
    boundaries = {start, end}
    for e in list(events)+list(sleeps):
        for value in interval(e):
            if start < value < end:
                boundaries.add(value)
    for value in markers:
        if start <= value <= end:
            boundaries.add(value)
    # Default light and habits change on wall-clock hour boundaries (offsets can be :30/:45).
    from zoneinfo import ZoneInfo
    zones = {profile.get('home_timezone', 'UTC')} | {e.get('timezone') for e in events if e.get('timezone')}
    for name in zones:
        zone = ZoneInfo(name)
        cursor = start.replace(minute=0, second=0, microsecond=0)-timedelta(hours=1)
        while cursor < end:
            local = cursor.astimezone(zone)
            # UTC stepping at 15m also covers historic fractional-hour offsets.
            if local.minute == 0 and start < cursor < end:
                boundaries.add(cursor)
            cursor += timedelta(minutes=15)
    times = sorted(boundaries)
    S = float(state['S']); osc = np.array([state['x'],state['xc'],state['n']])
    I = float(state.get('inertia', 0.)); previous = bool(state.get('sleep_mode', False))
    episode_start = dt(state['current_episode_start_utc']) if state.get('current_episode_start_utc') else None
    last_wake = state.get('last_wake_utc')
    output = []
    for left, right in zip(times, times[1:]):
        midpoint = left + (right-left)/2
        actual = [e for e in sleeps if dt(e['start_utc']) <= midpoint < dt(e['end_utc'])]
        wake_observed = any(e['kind'] in ('wake_actual','wake_inferred') and dt(e['start_utc']) <= midpoint < dt(e['end_utc']) for e in events)
        asleep = bool(actual) or (history and not wake_observed and habit_asleep(midpoint, profile))
        source = actual[0].get('source', 'observed') if actual else 'inferred_or_observed_awake' if wake_observed else 'assumed_history' if history else 'predicted_awake'
        if asleep and not previous:
            episode_start = left
        if previous and not asleep:
            duration = max(0., (left-(episode_start or left)).total_seconds()/3600)
            I = 1-math.exp(-duration/config['inertia_hours'])
            last_wake = stamp(left)
            episode_start = None
        if asleep:
            I = 0.
        light, light_source = light_at(midpoint, asleep, events, profile, config['light_factor'])
        duration = (right-left).total_seconds()/3600
        # Global UTC minute grid plus every event boundary, including sub-minute observations.
        first_minute = left.replace(second=0, microsecond=0)
        if first_minute < left:
            first_minute += timedelta(minutes=1)
        sample_times = [left]
        cursor = first_minute
        while cursor < right:
            if cursor != left:
                sample_times.append(cursor)
            cursor += timedelta(minutes=1)
        offsets = np.array([(t-left).total_seconds()/3600 for t in sample_times])
        osc_end, values = circadian.integrate(osc, duration, light, config, offsets)
        for t, offset, xyz in zip(sample_times, offsets, values.T):
            s = advance(S, offset, asleep, config['tau_w'],config['tau_s'])
            inertia = I*math.exp(-offset/config['inertia_hours']) if not asleep else 0.
            x, xc, n = map(float, xyz)
            amplitude = math.hypot(x,xc)
            phase = math.atan2(-xc,x)
            c = (1-math.cos(phase))/2 if amplitude >= 1e-3 else None
            w = config['burden_weights']
            burden = w[0]*s+w[1]*c+w[2]*inertia if c is not None else None
            output.append({'time_utc':stamp(t), 'S':s, 'x':x,'xc':xc,'n':n,
                           'phase_rad':phase,'amplitude':amplitude,'C':c,'inertia':inertia,
                           'burden':burden,'asleep':asleep,'sleep_source':source,
                           'light_lux':light,'light_source':light_source})
        S = advance(S,duration,asleep,config['tau_w'],config['tau_s'])
        I = 0. if asleep else I*math.exp(-duration/config['inertia_hours'])
        osc = osc_end
        previous = asleep
    final = {'at_utc':stamp(end),'S':S,'x':float(osc[0]),'xc':float(osc[1]),'n':float(osc[2]),
             'inertia':I,'sleep_mode':previous,'current_episode_start_utc':stamp(episode_start) if episode_start else None,
             'last_wake_utc':last_wake}
    # End is retained to integrate the final segment; metrics treat it as the left-hand limit.
    row = dict(output[-1]); amplitude = math.hypot(osc[0],osc[1]); phase = math.atan2(-osc[1],osc[0])
    c = (1-math.cos(phase))/2 if amplitude >= 1e-3 else None
    w = config['burden_weights']
    row.update(time_utc=stamp(end),S=S,x=float(osc[0]),xc=float(osc[1]),n=float(osc[2]),
               phase_rad=phase,amplitude=amplitude,C=c,inertia=I,
               burden=w[0]*S+w[1]*c+w[2]*I if c is not None else None)
    output.append(row)
    return output, final


def phase_markers(trace):
    if len(trace) < 3:
        return []
    # Irregular event points are deliberately handled as time values, not peak-index distance.
    points, _ = find_peaks(-np.array([p['x'] for p in trace]))
    found = []
    for index in points:
        a, m, b = trace[index-1:index+2]
        # Quadratic interpolation of the one-minute bracket, bounded to that bracket.
        x = np.array([(dt(p['time_utc'])-dt(m['time_utc'])).total_seconds()/3600 for p in (a,m,b)])
        coef = np.polyfit(x,[p['x'] for p in (a,m,b)],2)
        offset = max(x[0],min(x[-1],-coef[1]/(2*coef[0]))) if coef[0] > 0 else 0.
        instant = dt(m['time_utc'])+timedelta(hours=float(offset))
        ambiguous = bool(found and (instant-dt(found[-1]['cbtmin_utc'])).total_seconds()<6*3600)
        if ambiguous:
            found[-1]['ambiguous'] = True
        found.append({'cbtmin_utc':stamp(instant),'dlmo_proxy_utc':stamp(instant-timedelta(hours=7)),
                      'ambiguous':ambiguous,'method':'bounded quadratic, 1-minute bracket'})
    return found
