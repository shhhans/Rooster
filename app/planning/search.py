"""Budgeted deterministic beam search; approximate search costs are not physiology."""
from datetime import timedelta
from zoneinfo import ZoneInfo
from app.contracts.time import dt, stamp
from app.sleep_model.api import advance_inertia, advance_pressure, burden, penalty
from app.sleep_model.contracts import physiological_configuration

from .policy import POLICY_DEFAULTS

VERSION = 'sleep-beam-v2'


def generate(snapshot, windows, initial_s=.5, *, width=64, reference_trace=(), sleep_count=None):
    """Bounded beam search over the planning windows.

    The beam cost is a search-owned approximation: the model contributes its
    public pressure/inertia/burden/penalty primitives and the reference C
    trajectory, while the grid, the beam and the fixed need/excess buckets are
    planning's. ``sleep_count`` is the explicit sleep-count compensation
    switch: ``None`` (default) keeps the legacy cost, ``False`` drops the
    episode-count term from the objective.
    """
    start,end=dt(snapshot['now_utc']),dt(snapshot['horizon_end_utc'])
    total=(end-start).total_seconds()/3600
    # Bound memory and work explicitly rather than silently shortening the horizon.
    if total>24*35:
        return [],dict(version=VERSION,status='horizon_budget_exceeded',expanded=0)
    def offset(value):return (dt(value)-start).total_seconds()/3600
    resources=[(offset(w['start_utc']),offset(w['end_utc']),w['timezone']) for w in windows]
    cuts={0.,total}
    for a,b,_ in resources:cuts.update((max(0.,a),min(total,b)))
    for i in range(int(total*2)+1):cuts.add(i/2)
    duties=[]
    for e in snapshot['events']:
        if e['kind']=='duty':duties.append((offset(e['start_utc']),offset(e['end_utc'])))
    times=sorted(t for t in cuts if 0<=t<=total)
    need=float(snapshot.get('profile',{}).get('sleep_need_hours',8))
    # The public search primitives resolve the physiological configuration
    # internally; resolve it once here and hand it to every call.
    config=physiological_configuration()
    latency_hours=config['latency_minutes']/60
    count_enabled=True if sleep_count is None else bool(sleep_count)
    reference={round(offset(p['time_utc'])*60):p['C'] for p in reference_trace}
    duty_hours=sum(max(0.,min(total,b)-max(0.,a)) for a,b in duties)
    actual=[e for e in snapshot['events'] if e['kind']=='sleep_actual' and dt(e['end_utc'])<=start]
    history=tuple((offset(e['start_utc']),offset(e['end_utc'])) for e in actual)
    anchor=max((b for a,b in history),default=0.)
    explicit=snapshot.get('initial_state',{}).get('last_wake_utc')
    if explicit and dt(explicit)<=start:anchor=max(anchor,offset(explicit)) if history else offset(explicit)
    # cost, S, intervals, last wake, asleep, inertia
    beam=[(0.,initial_s,(),anchor,False,0.)]
    expanded=0
    for left,right in zip(times,times[1:]):
        h=right-left;mid=(left+right)/2
        resource=next((w for w in resources if w[0]<=left+1e-8 and w[1]>=right-1e-8),None)
        duty=any(a<=mid<b for a,b in duties)
        C=reference.get(round(mid*60),.5)
        trials=[]
        for cost,S,blocks,wake,asleep,I in beam:
            for sleep in (False,True) if resource else (False,):
                expanded+=1
                if asleep and not sleep and blocks[-1][1]-blocks[-1][0]<.5-1e-8:continue
                if sleep and not asleep and resource[1]-left<.5-1e-8:continue
                if sleep and asleep and not any(w[0]<=blocks[-1][0]+1e-8 and w[1]>=right-1e-8 for w in resources):continue
                if sleep:
                    new=blocks[:-1]+((blocks[-1][0],right),) if asleep else blocks+((left,right),)
                    latency=max(0.,min(h,latency_hours-(left-new[-1][0])))
                    nextS=advance_pressure(advance_pressure(S,latency,asleep=False,config=config),
                                           h-latency,asleep=True,config=config)
                    nextwake=wake
                else:
                    new=blocks;nextS=advance_pressure(S,h,asleep=False,config=config);nextwake=left if asleep else wake
                if sleep:
                    nextI=advance_inertia(I,h,asleep=True,config=config)
                elif asleep:
                    # Wake boundary: the onset of the episode that just ended,
                    # latency excluded exactly as the forecast does.
                    nextI=advance_inertia(I,h,asleep=False,
                                          woken_from_hours=max(0.,blocks[-1][1]-blocks[-1][0]-latency_hours),
                                          config=config)
                else:
                    nextI=advance_inertia(I,h,asleep=False,config=config)
                # The reference C trajectory is a search-owned approximation; a
                # missing reference value falls back to the neutral 0.5.
                C_value=C if C is not None else .5
                B=burden(nextS,C_value,nextI,config=config)
                step=.65*h*penalty(B,config=config)/max(1.,duty_hours or total) if duty or (not duties and not sleep) else 0.
                # Same rolling episode cost as the scorer, approximated on the beam grid.
                # New intervals are opportunities, so exclude the fixed onset latency.
                if count_enabled:
                    episodes=sum(a<mid and b>mid-24 for a,b in history)
                    episodes+=sum(a+latency_hours<min(b,mid) and b>mid-24 for a,b in new)
                    step+=POLICY_DEFAULTS['sleep_count_coefficient']/100*h/total*max(0,episodes-POLICY_DEFAULTS['sleep_count_threshold'])**2
                # Same fixed 24h need/excess buckets as the current model, not user heuristics.
                if abs(right%24)<1e-8 or right==total:
                    bucket_start=max(0.,right-24) if abs(right%24)<1e-8 else right-right%24
                    length=right-bucket_start;target=need*length/24
                    amount=sum(max(0.,min(b,right)-max(a+.25,bucket_start)) for a,b in new)
                    step+=(length/total)*(.2*max(0.,target-amount)/target+.05*min(1.,max(0.,amount-target-2*length/24)/target))
                trials.append((cost+step,nextS,new,nextwake,sleep,nextI))
        trials.sort(key=lambda v:(v[0],v[2]))
        beam=[];bins={}
        for value in trials:
            _,S,blocks,wake,sleep,I=value
            amount=sum(max(0.,min(b,right)-max(a,right-24)) for a,b in blocks)
            key=(sleep,int(S*12),int(amount),int((right-wake)/3),min(len(blocks),6))
            if bins.get(key,0)>=3:continue
            bins[key]=bins.get(key,0)+1;beam.append(value)
            if len(beam)>=width:break
        if not beam:break
    unique={}
    for cost,S,blocks,wake,sleep,I in beam:
        if blocks and (not sleep or blocks[-1][1]-blocks[-1][0]>=.5):unique.setdefault(blocks,cost)
    result=[]
    for blocks,cost in sorted(unique.items(),key=lambda v:(v[1],v[0]))[:12]:
        values=[]
        for i,(a,b) in enumerate(blocks):
            zone=next(w[2] for w in resources if w[0]<=a+1e-8 and w[1]>=b-1e-8)
            values.append(dict(id=f'sleep-{i+1}',start_utc=stamp(start+timedelta(hours=a)),
                               end_utc=stamp(start+timedelta(hours=b)),timezone=zone,
                               role='main' if b-a>=2 else 'nap'))
        result.append(values)
    return result,dict(version=VERSION,status='complete' if result else 'no_candidate_found',
                       expanded=expanded,beam_width=width,grid_minutes=30,
                       surrogate='model S/C/I burden + existing fixed-bucket need/excess; reference C approximation; full model reranks finalists')
