"""Pure physiological pipeline shared by the legacy evaluation orchestration
and the public model API (module decoupling contract step 2).

No planning, preference, product-ranking or sleep-count policy lives here:
the legacy scoring orchestration composes policy on top of :func:`run`, and
the public API (``app.sleep_model.api``) exposes the same pipeline as the
pure model. The code is a verbatim move of the physiological section of the
legacy evaluation path so behavior, hashes and error strings are unchanged.
"""
from copy import deepcopy
from datetime import timedelta
from .normalize import normalize, dt, stamp, interval, merge
from .initialization import build_state
from .simulation import simulate
from .sleep_forecast import forecast_sleep
from .metrics import measure


def run(snapshot, blocks, *, start_utc=None, end_utc=None, mode='forecast', config=None):
    """Run the deterministic physiological pipeline over an evaluation window.

    ``snapshot`` and ``blocks`` must be private copies owned by the caller;
    ``config`` must be a fully resolved configuration dict. Returns every
    physiological intermediate and output needed by the legacy orchestration
    and the public API; policy composition stays with the callers. Raises the
    same ``ValueError``/``KeyError``/``TypeError``/``ArithmeticError`` as the
    legacy path.
    """
    now=dt(snapshot['now_utc']);start=dt(start_utc or now);end=dt(end_utc or snapshot['horizon_end_utc'])
    if end<=start or mode not in ('forecast','retrospective'):
        raise ValueError('invalid evaluation bounds or mode')
    if mode=='forecast' and start<now:
        raise ValueError('forecast window precedes knowledge cutoff')
    if mode=='retrospective' and end>now:
        raise ValueError('retrospective end exceeds knowledge cutoff')
    snap,blocks,assumptions=normalize(snapshot,blocks,now)
    actual=[dict(e,source='observed') for e in snap['events'] if e['kind']=='sleep_actual']
    state,initialization=build_state(snap,now if mode=='forecast' else start,config,checkpoint=snapshot.get('model_checkpoint'))
    if mode=='forecast':
        # Current sessions extend from observed sleep, without charging another latency.
        effective=deepcopy(blocks)
        for session in snap.get('sessions',{}).values():
            if session.get('actual_end_utc') is None:
                bound=dt(session['expected_end_utc'])
                effective=[b for b in effective if b['id']!=session.get('planned_block_id')]
                if bound>now:
                    effective.append({'id':'ongoing-'+session['id'],'start_utc':stamp(now-timedelta(minutes=config['latency_minutes'])),
                                      'end_utc':stamp(min(bound,end)),'role':'main','parent_sleep_episode_id':session['id']})
        predicted=forecast_sleep(effective,config)
        sleeps=predicted
        simstart=now
        history=False
    else:
        predicted=[];sleeps=actual;simstart=start;history=True
    markers=[dt(m['time_utc']) for m in snap.get('task_markers',[]) if simstart<=dt(m['time_utc'])<end]
    trace,final=simulate(state,simstart,end,sleeps,snap['events'],snap['profile'],config,history=history,markers=markers+[start])
    trace=[p for p in trace if dt(p['time_utc'])>=start]
    # Retrospective metrics use assumed-history segments too, but expose actual separately.
    measured_sleeps=sleeps
    if mode=='retrospective':
        pieces=[]
        for p,q in zip(trace,trace[1:]):
            if p['asleep']:pieces.append({'id':'effective-'+p['time_utc'],'start_utc':p['time_utc'],'end_utc':q['time_utc']})
        measured_sleeps=[{'id':f'effective-{i}','start_utc':stamp(a),'end_utc':stamp(b)} for i,(a,b) in enumerate(merge(interval(e) for e in pieces))]
    metrics,phys=measure(trace,snap,measured_sleeps,start,end,config,state)
    return {'now':now,'start':start,'end':end,'mode':mode,'snapshot':snap,'blocks':blocks,
            'assumptions':assumptions,'actual':actual,'predicted':predicted,'sleeps':sleeps,
            'measured_sleeps':measured_sleeps,'state':state,'initialization':initialization,
            'trace':trace,'final':final,'metrics':metrics,'physiological_loss':phys}
