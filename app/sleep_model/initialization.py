"""Cached population prior; actual corrections replay a content-addressed prefix."""
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from copy import deepcopy
from .contracts import VERSION, digest
from .normalize import dt, stamp
from .simulation import simulate


@lru_cache(maxsize=32)
def prior(bed, wake, factor, step, rtol, atol, tau_w, tau_s, inertia_hours):
    config = {'light_factor':factor,'max_step_minutes':step,'rtol':rtol,'atol':atol,
              'tau_w':tau_w,'tau_s':tau_s,'inertia_hours':inertia_hours,'burden_weights':[.6,.25,.15]}
    profile = {'home_timezone':'UTC','_bed':bed,'_wake':wake}
    start = datetime(2000,1,1,tzinfo=timezone.utc)
    state = {'S':.5,'x':.5,'xc':.5,'n':.5,'inertia':0,'sleep_mode':False}
    stable = 0; converged = False
    for day in range(120):
        a = start + timedelta(days=day)
        environment = [{'id':'prior-home','kind':'environment','location':'home','timezone':'UTC',
                        'start_utc':stamp(a),'end_utc':stamp(a+timedelta(days=1))}]
        trace, end = simulate(state,a,a+timedelta(days=1),[],environment,profile,config,history=True)
        delta = max(abs(end[k]-state[k]) for k in ('S','x','xc','n','inertia'))
        stable = stable+1 if delta < 1e-6 else 0
        state = end
        if day >= 29 and stable >= 3:
            converged = True
            break
    return trace, converged, day+1


_CACHE = {}


def visible_prefix(snapshot, at):
    """Visible events starting before ``at``, clipped to ``at``.

    Candidate/future data is excluded by construction; this is the exact
    prefix the initialization and the checkpoint identity are built from.
    """
    prefix = []
    for raw in snapshot['events']:
        if dt(raw['start_utc']) < at:
            e = deepcopy(raw)
            e['end_utc'] = stamp(min(at,dt(e['end_utc'])))
            prefix.append(e)
    return prefix


def checkpoint_key(snapshot, at, config):
    """Content-addressed identity of the visible prefix at ``at``.

    Excludes candidate/future data by construction (only events starting
    before ``at`` are hashed, clipped to ``at``). The hashed fields and their
    serialization must stay byte-identical: legacy checkpoints and public
    API states (``app.sleep_model.api``) interoperate through this key.
    """
    return digest({'at':stamp(at),'prefix':visible_prefix(snapshot, at),'profile':snapshot['profile'],
                  'config':config,'model':VERSION})


def build_state(snapshot, at, config, checkpoint=None):
    profile = snapshot['profile']
    # Exclude candidate/future data from the initialization key.
    prefix = visible_prefix(snapshot, at)
    key = checkpoint_key(snapshot, at, config)
    if checkpoint:
        if checkpoint.get('input_prefix_hash') != key or checkpoint.get('model_version') != VERSION:
            raise ValueError('stale or incompatible checkpoint')
        return deepcopy(checkpoint), {'checkpoint_reused':True,'input_prefix_hash':key}
    if key in _CACHE:
        return deepcopy(_CACHE[key])
    observations = [e for e in prefix if e['kind'] in ('sleep_actual','wake_actual')]
    earliest = min((dt(e['start_utc']) for e in observations),default=at-timedelta(hours=72))
    start = max(at-timedelta(days=14),min(earliest,at-timedelta(hours=72)))
    cycle, converged, days = prior(profile['_bed'],profile['_wake'],config['light_factor'],
        config['max_step_minutes'],config['rtol'],config['atol'],config['tau_w'],config['tau_s'],config['inertia_hours'])
    from zoneinfo import ZoneInfo
    # Reconstruct the cached cycle state at the exact fixed-offset wall time.
    local = start.astimezone(ZoneInfo(profile.get('home_timezone','UTC')))
    hour = local.hour+local.minute/60+local.second/3600+local.microsecond/3.6e9
    midnight = dt(cycle[0]['time_utc'])
    target = midnight+timedelta(hours=hour)
    chosen = max((p for p in cycle if dt(p['time_utc'])<=target),key=lambda p:p['time_utc'])
    state = {k:chosen[k] for k in ('S','x','xc','n','inertia')}
    state.update(sleep_mode=chosen['asleep'])
    # Prior episode and wake anchors are assumptions, never called confirmed observations.
    from .simulation import habit_asleep
    if habit_asleep(start,profile):
        since = (hour-profile['_bed'])%24
        state['current_episode_start_utc'] = stamp(start-timedelta(hours=since))
    else:
        state['last_wake_utc'] = stamp(start-timedelta(hours=(hour-profile['_wake'])%24))
    if target > dt(chosen['time_utc']):
        base = dt(chosen['time_utc']); delta = target-base
        env = [{'id':'prior','kind':'environment','location':'home','timezone':'UTC',
                'start_utc':stamp(base),'end_utc':stamp(target)}]
        _, state = simulate(state,base,target,[],env,{'home_timezone':'UTC', '_bed':profile['_bed'],'_wake':profile['_wake']},config,history=True)
        # The synthetic UTC clock is only a phase selector; anchors remain real instants.
        if state.get('current_episode_start_utc'):
            state['current_episode_start_utc'] = stamp(start-timedelta(hours=(hour-profile['_bed'])%24))
        if state.get('last_wake_utc'):
            state['last_wake_utc'] = stamp(start-timedelta(hours=(hour-profile['_wake'])%24))
    sleeps = [dict(e,source='observed') for e in prefix if e['kind']=='sleep_actual']
    _, state = simulate(state,start,at,sleeps,prefix,profile,config,history=True)
    state.update(input_prefix_hash=key,model_version=VERSION,configuration_hash=digest(config),
                 initialization_id=digest({'bed':profile['_bed'],'wake':profile['_wake'],'config':config}))
    info = {'input_prefix_hash':key,'history_start_utc':stamp(start),'prior_converged':converged,
            'prior_days':days,'checkpoint_reused':False}
    if len(_CACHE)>256:
        _CACHE.clear()
    _CACHE[key] = deepcopy((state,info))
    return state,info
