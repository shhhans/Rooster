"""Descriptive metrics and fixed, separately versioned engineering burden loss."""
import math
from datetime import timedelta
import numpy as np
from .normalize import dt, interval, merge, stamp
from .sleep_forecast import sleep_hours
from .simulation import phase_markers


def penalty(z, config):
    k, threshold = config['softness'],config['threshold']
    def q(x):
        v = (x-threshold)/k
        return k*(max(v,0)+math.log1p(math.exp(-abs(v))))
    return (q(z)-q(0))/(q(1)-q(0))


def measure(trace, snapshot, sleeps, start, end, config, initial_state):
    events = snapshot['events']
    duties = merge((max(start,a),min(end,b)) for e in events if e['kind']=='duty'
                   for a,b in [interval(e)] if a<end and b>start)
    markers = [m for m in snapshot.get('task_markers',[]) if start<=dt(m['time_utc'])<end]
    selected = []
    area=0.; threshold_hours=0.; excess_area=0.; mean_loss=0.; duration=0.
    has_duties = bool(duties)
    for left,right in zip(trace,trace[1:]):
        a,b=dt(left['time_utc']),dt(right['time_utc'])
        midpoint=a+(b-a)/2
        included = any(x<=midpoint<y for x,y in duties) if has_duties else not left['asleep']
        if not included or left['burden'] is None or right['burden'] is None:
            continue
        hours=(b-a).total_seconds()/3600
        # Right at a waking boundary has a jump; use the left-hand limit for integration.
        right_b = right['burden']
        if left['asleep'] != right['asleep'] and not left['asleep']:
            right_b = config['burden_weights'][0]*right['S']+config['burden_weights'][1]*right['C'] + config['burden_weights'][2]*left['inertia']*math.exp(-hours/config['inertia_hours'])
        z=(left['burden']+right_b)/2
        selected.extend([left['burden'],right_b]);duration+=hours;area+=z*hours
        mean_loss+=(penalty(left['burden'],config)+penalty(right_b,config))/2*hours
        threshold=config['threshold']
        lo,hi=sorted((left['burden'],right_b))
        fraction=1. if lo>=threshold else 0. if hi<=threshold else (hi-threshold)/(hi-lo)
        threshold_hours+=hours*fraction
        excess_area+=hours*((lo+hi)/2-threshold) if lo>=threshold else hours*fraction*(hi-threshold)/2 if hi>threshold else 0.
    by_time={p['time_utc']:p for p in trace}
    task_values=[]
    for m in markers:
        p=by_time.get(stamp(dt(m['time_utc'])))
        task_values.append(dict(m,**{k:p[k] if p else None for k in ('burden','S','C','inertia')}))
    available=[]
    if duration:
        available.extend([(.60,mean_loss/duration),(.25,penalty(max(selected),config))])
    keys=[m['burden'] for m in task_values if m['burden'] is not None]
    if keys:
        available.append((.15,penalty(max(keys),config)))
    phys=sum(w*x for w,x in available)/sum(w for w,x in available) if available else None
    if any(p['burden'] is None for p in trace):
        phys=None
    actual=[e for e in events if e['kind']=='sleep_actual']
    rolling={str(h):sleep_hours(actual,start-timedelta(hours=h),start) for h in (24,48,72)}
    actual_wake=[dt(e['end_utc']) for e in actual if dt(e['end_utc'])<=start]
    explicit=(snapshot.get('initial_state') or {}).get('last_wake_utc')
    if explicit and dt(explicit)<=start:
        actual_wake.append(dt(explicit))
    anchor=max(actual_wake) if actual_wake else start
    cursor=anchor;longest=0.;wake_margins=[]
    future_sleep=merge((max(start,a),min(end,b)) for e in sleeps for a,b in [interval(e)] if a<end and b>start)
    # A sleep already in progress at start is not a new pre-sleep wake gap.
    for a,b in future_sleep:
        longest=max(longest,max(0.,(a-cursor).total_seconds()/3600));cursor=b
        upcoming=[dt(m['time_utc']) for m in markers if dt(m['time_utc'])>=b]
        wake_margins.append({'wake_utc':stamp(b),'next_task_minutes':(min(upcoming)-b).total_seconds()/60 if upcoming else None})
    longest=max(longest,(end-cursor).total_seconds()/3600)
    # Fragmentation is evaluated only where explicit sleep-episode identity exists.
    episodes={}
    for e in (actual if actual and start<dt(snapshot.get('now_utc',stamp(end))) else sleeps):
        if e.get('episode_id'):
            episodes.setdefault(e['episode_id'],[]).append(e)
    interruptions=0;waso=0.;episode_coverage=False
    for group in episodes.values():
        pieces=merge(interval(e) for e in group)
        episode_coverage |= any(a<end and b>start for a,b in pieces)
        for left,right in zip(pieces,pieces[1:]):
            a,b=left[1],right[0]
            if a<end and b>start:
                episode_coverage=True
                interruptions+=int(start<=a<end)
                waso+=(min(end,b)-max(start,a)).total_seconds()/3600
    if not episode_coverage:
        interruptions=waso=None
    coverage=merge((max(start-timedelta(hours=72),a),min(start,b)) for e in events if e['kind'] in ('sleep_actual','wake_actual') for a,b in [interval(e)] if a<start and b>start-timedelta(hours=72))
    coverage_hours=sum((b-a).total_seconds()/3600 for a,b in coverage)
    return {'sleep_hours':sleep_hours(sleeps,start,end),'actual_sleep_hours':sleep_hours(actual,start,end) if actual else None,
            'rolling_actual_sleep_hours':rolling,'history_observation_coverage':coverage_hours/72,
            'longest_wake_hours':longest,'left_censored':not bool(actual_wake),'right_censored':not bool(future_sleep and future_sleep[-1][1]==end),
            'domain':'duty' if has_duties else 'awake','duty_hours':sum((b-a).total_seconds()/3600 for a,b in duties),
            'duty_burden':area/duration if has_duties and duration else None,
            'mean_burden':area/duration if duration else None,'peak_burden':max(selected) if selected else None,
            'minimum_burden':min(selected) if selected else None,
            'above_threshold_hours':threshold_hours if duration else None,
            'excess_area':excess_area if duration else None,
            'task_markers':task_values,'task_marker_coverage':sum(m['burden'] is not None for m in task_values)/len(markers) if markers else None,
            'sleep_piece_count':len(future_sleep),'main_count':sum(e.get('role')=='main' for e in sleeps),
            'nap_count':sum(e.get('role')=='nap' for e in sleeps),'fragmentation_interruptions':interruptions,
            'waso_hours':waso,
            'wake_task_margins':wake_margins,'phase_markers':phase_markers(trace),
            'assumed_light_fraction':sum(p['light_source']!='observed' for p in trace)/len(trace)}, phys
