"""Public pure physiological model API (module decoupling contract step 2).

The formal model entry points live here: ``compute`` (full computation),
``initialize`` (initial state / checkpoint validation), ``advance_state``
(observation-assimilation increment with explicit full visible history), the
standalone search components ``advance_pressure``/``advance_inertia``/
``burden``/``penalty``, and ``concatenate_traces`` (explicit seam rule).

No planning, preference, product-ranking, constraint or sleep-count policy is
imported or computed here; composition stays with the callers. The legacy
``app.sleep_model.evaluate``/``adapter`` paths remain the compatibility
computation and are behavior-identical for the same physiological parameters:
this module resolves physiological parameters only and pairs them with the
current policy defaults via ``compatibility_configuration`` so identity hashes
(``input_prefix_hash``, ``initialization_id``, ``configuration_hash``) and
checkpoints stay byte-compatible with the legacy path.

Invalid inputs raise explicit ``ValueError``/``TypeError``; nothing silently
returns fake scores. Units for every numeric output are declared in ``UNITS``.
"""
import math
from copy import deepcopy
from datetime import timedelta

from . import pipeline
from .contracts import (PHYSIOLOGICAL_DEFAULTS, VERSION, compatibility_configuration,
                        digest, physiological_configuration)
from .homeostat import advance as _homeostat_advance
from .initialization import build_state, checkpoint_key
from .melatonin import compute_melatonin
from .metrics import penalty as _softplus_penalty
from .normalize import dt, normalize, stamp
from .simulation import simulate
from .sleep_forecast import forecast_sleep as _forecast_sleep

API_VERSION = 'rooster-sleep-model-api-v1'


def forecast_sleep_opportunities(opportunities, *, config=None):
    """Return forecast sleep intervals for the supplied planned opportunities."""
    return _forecast_sleep(opportunities, physiological_configuration(config))

# Component graph of the physiological model. Selections are names only;
# parameters travel exclusively through the physiological config.
COMPONENT_DEPENDENCIES = {
    'pressure': (),
    'circadian': (),
    'inertia': (),
    'burden': ('pressure', 'circadian', 'inertia'),
    'metrics': ('burden',),
    # Opt-in endogenous-only Breslow13 component (independent oscillator).
    'melatonin': (),
}

# The frozen default selection: the five legacy components. ``melatonin`` is
# reachable only by explicit naming, so an omitted/None selection keeps the
# legacy outputs, hashes and checkpoints byte-identical.
DEFAULT_COMPONENTS = ('pressure', 'circadian', 'inertia', 'burden', 'metrics')

# Units of every numeric output surface (dimensionless unless stated).
UNITS = {
    'S': 'dimensionless homeostatic sleep pressure in [0, 1]',
    'x': 'dimensionless circadian oscillator state',
    'xc': 'dimensionless circadian oscillator state',
    'n': 'dimensionless circadian light adaptation state',
    'inertia': 'dimensionless sleep inertia in [0, 1]',
    'C': 'dimensionless circadian burden factor in [0, 1] (None while oscillator amplitude < 1e-3)',
    'burden': 'dimensionless weighted composite (defaults 0.60*S + 0.25*C + 0.15*inertia)',
    'penalty': 'dimensionless softplus threshold loss in [0, 1] (defaults threshold 0.60, softness 0.05)',
    'physiological_loss': 'dimensionless weighted penalty composite',
    'mean_burden': 'dimensionless', 'peak_burden': 'dimensionless', 'minimum_burden': 'dimensionless',
    'duty_burden': 'dimensionless mean burden over duty segments',
    'sleep_hours': 'hours', 'actual_sleep_hours': 'hours', 'rolling_actual_sleep_hours': 'hours',
    'longest_wake_hours': 'hours', 'waso_hours': 'hours', 'above_threshold_hours': 'hours',
    'duty_hours': 'hours', 'history_observation_coverage': 'fraction of the last 72 hours',
    'excess_area': 'burden-hours', 'sleep_piece_count': 'count', 'main_count': 'count',
    'nap_count': 'count', 'fragmentation_interruptions': 'count',
    'phase_rad': 'radians', 'amplitude': 'dimensionless', 'light_lux': 'photopic lux',
    'time_utc': 'UTC ISO-8601 instant',
}

_PHYSIOLOGY_ASSUMPTIONS = [
    'unobserved history filled with habit prior',
    'unmeasured light uses declared photopic scenario',
    'latency and inertia are engineering assumptions; outputs are not fitness-for-duty judgments',
]


def resolve_components(selection=None):
    """Resolve a component selection to its dependency closure.

    Returns ``(requested, resolved)``, both deterministic sorted name lists.
    Unknown names, bare strings and per-component parameter dicts are explicit
    errors: components select equations by name only, parameters travel
    through the physiological config.

    ``selection=None`` resolves to the frozen legacy default selection
    (:data:`DEFAULT_COMPONENTS`); the opt-in ``melatonin`` component is never
    added implicitly.
    """
    if selection is None:
        requested = set(DEFAULT_COMPONENTS)
    else:
        if isinstance(selection, str):
            raise ValueError('components must be an iterable of names, not a string')
        if isinstance(selection, dict):
            raise ValueError('components must be an iterable of names; parameters belong in config')
        try:
            requested = set(selection)
        except TypeError:
            raise ValueError('components must be an iterable of names') from None
        unknown = sorted(requested - set(COMPONENT_DEPENDENCIES))
        if unknown:
            raise ValueError('unknown components: ' + ','.join(unknown))
    resolved = set(requested)
    pending = True
    while pending:
        pending = False
        for name in sorted(resolved):
            for dependency in COMPONENT_DEPENDENCIES[name]:
                if dependency not in resolved:
                    resolved.add(dependency)
                    pending = True
    return sorted(requested), sorted(resolved)


def _validate_snapshot(snapshot):
    if not isinstance(snapshot, dict):
        raise TypeError('snapshot must be a dict')
    if 'now_utc' not in snapshot:
        raise ValueError('snapshot missing required field: now_utc')
    dt(snapshot['now_utc'])


def _validate_state(state):
    """Finite/bounded physiological state validation (documented definition):
    ``S`` in [0, 1], ``inertia`` in [0, 1], ``x``/``xc``/``n`` finite,
    ``sleep_mode`` boolean, and any present anchor (``at_utc``,
    ``current_episode_start_utc``, ``last_wake_utc``) parseable."""
    if not isinstance(state, dict):
        raise TypeError('model state must be a dict')
    missing = [key for key in ('S', 'x', 'xc', 'n', 'inertia', 'sleep_mode') if key not in state]
    if missing:
        raise ValueError('incomplete model state: ' + ','.join(missing))
    for key in ('S', 'x', 'xc', 'n', 'inertia'):
        value = state[key]
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError('non-finite model state: ' + key)
    if not 0 <= state['S'] <= 1:
        raise ValueError('process S outside [0, 1]')
    if not 0 <= state['inertia'] <= 1:
        raise ValueError('inertia outside [0, 1]')
    if not isinstance(state['sleep_mode'], bool):
        raise ValueError('sleep_mode must be boolean')
    for key in ('at_utc', 'current_episode_start_utc', 'last_wake_utc'):
        if state.get(key) is not None:
            try:
                dt(state[key])
            except (ValueError, TypeError, KeyError):
                raise ValueError('invalid state anchor: ' + key) from None


def compute(snapshot, sleep_blocks=None, *, start_utc=None, end_utc=None, mode='forecast',
            config=None, components=None, melatonin_config=None, melatonin_summaries=None,
            melatonin_crossings=False):
    """Full pure physiological computation over an evaluation window.

    Returns normalized inputs, actual/effective/predicted sleeps, the UTC
    minute-grid trajectory, initial/final states, physiological metrics and
    loss, units, validity, assumptions and provenance. No product utility,
    preferences, constraints, need/excess buckets, sleep-count policy or rank
    weights are computed or returned. In retrospective mode ``sleep_blocks``
    are validated and normalized but do not become predicted sleep (legacy
    parity). Invalid inputs raise ``ValueError`` (the legacy ``evaluate``
    path answers with an ``invalid_input`` dict instead).

    ``components`` selects the physiological components; an omitted/None
    selection keeps the five legacy components, so the result is
    byte-identical to the pre-melatonin behavior and carries no ``melatonin``
    key. The opt-in ``'melatonin'`` component adds ``result['melatonin']``:
    the independent endogenous-only Breslow13 states, units, validity,
    assumptions, provenance and checkpoints. ``melatonin_config``,
    ``melatonin_summaries`` and ``melatonin_crossings`` are read only when
    that component is selected.
    """
    physio = physiological_configuration(config)
    compat = compatibility_configuration(physio)
    requested, resolved = resolve_components(components)
    _validate_snapshot(snapshot)
    if end_utc is None and 'horizon_end_utc' not in snapshot:
        raise ValueError('snapshot missing required field: horizon_end_utc (or pass end_utc)')
    core = pipeline.run(deepcopy(snapshot), deepcopy(sleep_blocks or []),
                        start_utc=start_utc, end_utc=end_utc, mode=mode, config=compat)
    snap = core['snapshot']
    blocks = core['blocks']
    initialization = core['initialization']
    prior_converged = bool(initialization.get('prior_converged', True))
    loss_available = core['physiological_loss'] is not None
    reasons = []
    if not prior_converged:
        reasons.append('prior_not_converged')
    if not loss_available:
        reasons.append('physiological_loss_unavailable')
    status = 'partial' if reasons else 'complete'
    assumptions = list(core['assumptions']) + list(_PHYSIOLOGY_ASSUMPTIONS)
    if not prior_converged:
        assumptions.append('prior_not_converged')
    result = {
        'api_version': API_VERSION, 'model_version': VERSION, 'mode': core['mode'],
        'status': status,
        'validity': {'status': status, 'physiological_loss_available': loss_available,
                     'prior_converged': prior_converged, 'partial_reasons': reasons},
        'components': {'requested': list(requested), 'resolved': list(resolved),
                       'dependencies': {name: list(deps) for name, deps in COMPONENT_DEPENDENCIES.items()},
                       'execution': 'the full physiological solver executes every component; the selection '
                                   'governs reported component coverage, not executed equations (documented '
                                   'limitation of the current implementation)'},
        'knowledge_cutoff_utc': stamp(core['now']),
        'evaluation_start_utc': stamp(core['start']), 'evaluation_end_utc': stamp(core['end']),
        'normalized': {'snapshot': snap, 'sleep_blocks': blocks},
        'actual_sleep': core['actual'], 'effective_sleep': core['measured_sleeps'],
        'predicted_sleep': core['predicted'],
        'trace': core['trace'], 'initial_state': core['state'], 'final_state': core['final'],
        'physiological_metrics': core['metrics'], 'physiological_loss': core['physiological_loss'],
        'units': dict(UNITS), 'assumptions': assumptions,
        'provenance': {
            'model_version': VERSION, 'api_version': API_VERSION,
            'configuration': physio, 'model_config_hash': digest(physio),
            'compatibility_configuration': compat, 'compatibility_configuration_hash': digest(compat),
            'snapshot_hash': digest(snapshot),
            'opportunity_hash': digest(blocks),
            'input_hash': digest({'snapshot': snapshot, 'opportunities': blocks, 'mode': core['mode'],
                                  'start_utc': stamp(core['start']), 'end_utc': stamp(core['end'])}),
            'trace_hash': digest(core['trace']),
            'initialization': initialization,
            'equation_source': 'sources/circadian/manifest.json',
        },
    }
    if 'melatonin' in resolved:
        result['melatonin'] = _melatonin_block(core, snapshot, physio, melatonin_config,
                                                melatonin_summaries, melatonin_crossings)
    return result


def _melatonin_block(core, snapshot, physio, melatonin_config, summaries, crossings):
    """Integrated endogenous-only melatonin run over the evaluation window.

    The sampling grid is the legacy trajectory's instants (strictly
    increasing, with the window endpoints included); the per-interval light
    and its provenance label come from the interval's left-edge trajectory row
    and are propagated verbatim (sleep-gated dark rows keep their own label).
    The checkpoint identity series is the snapshot's declared photopic
    scenario (sleep-independent), so the same scenario hashes identically in
    the run that produced a checkpoint and in the run that resumes it.
    """
    start = core['start']
    end = core['end']
    trace = core['trace']
    instants = [dt(point['time_utc']) for point in trace]
    for previous, current in zip(instants, instants[1:]):
        if current <= previous:
            raise ValueError('legacy trajectory grid is not strictly increasing')
    grid = [start] + [instant for instant in instants if start < instant < end] + [end]
    luxes = []
    labels = []
    chosen = None
    position = 0
    for index in range(len(grid) - 1):
        left = grid[index]
        while position < len(trace) and instants[position] <= left:
            chosen = trace[position]
            position += 1
        if chosen is None:
            chosen = trace[0]
        luxes.append(float(chosen['light_lux']))
        labels.append(chosen['light_source'])
    identity = _scenario_light_rows(core['snapshot'], start, end, physio['light_factor'])
    consumed = None
    checkpoint = (snapshot or {}).get('melatonin_checkpoint')
    if checkpoint is not None:
        consumed = _scenario_light_rows(core['snapshot'], dt(checkpoint['window_start_utc']),
                                        dt(checkpoint['at_utc']), physio['light_factor'])
    return compute_melatonin(start_utc=stamp(start), end_utc=stamp(end),
                              grid_utc=[stamp(instant) for instant in grid],
                              light_lux=luxes, light_provenance=labels,
                              config=melatonin_config, summaries=summaries, crossings=crossings,
                              checkpoint=checkpoint, light_prefix=consumed, identity_light=identity)


def _scenario_light_rows(snapshot, start, end, light_factor):
    """Canonical minute series of the declared photopic scenario light.

    Derived from the snapshot's own events (sleep-independent), so the same
    scenario hashes identically in the run that produced a checkpoint and in
    the run that resumes it.
    """
    from .light import light_at
    events = snapshot['events']
    profile = snapshot['profile']
    rows = []
    cursor = start.replace(second=0, microsecond=0)
    if cursor < start:
        cursor += timedelta(minutes=1)
    while cursor < end:
        lux, source = light_at(cursor, False, events, profile, light_factor)
        rows.append({'start_utc': stamp(cursor), 'light_lux': float(lux),
                     'light_provenance': source})
        cursor += timedelta(minutes=1)
    return rows


def initialize(snapshot, *, cutoff_utc=None, config=None, checkpoint=None):
    """Normalize and validate the initial state at a knowledge cutoff.

    ``cutoff_utc`` defaults to ``snapshot['now_utc']`` and must not exceed it.
    A ``checkpoint`` (legacy ``model_checkpoint`` or a public-API state) is
    reused only when its identity matches the visible prefix
    (``checkpoint_key``); derived position/identity keys are always recomputed
    for the returned state, which is itself a valid legacy checkpoint.
    """
    physio = physiological_configuration(config)
    compat = compatibility_configuration(physio)
    _validate_snapshot(snapshot)
    now = dt(snapshot['now_utc'])
    cutoff = dt(cutoff_utc) if cutoff_utc is not None else now
    if cutoff > now:
        raise ValueError('cutoff exceeds knowledge cutoff')
    snap, _, assumptions = normalize(snapshot, [], cutoff)
    state, info = build_state(snap, cutoff, compat, checkpoint=checkpoint)
    state = dict(state)
    state.update(at_utc=stamp(cutoff), api_version=API_VERSION,
                 model_config_hash=digest(physio), configuration_hash=digest(compat))
    if state['input_prefix_hash'] != checkpoint_key(snap, cutoff, compat):
        raise ValueError('stale or incompatible checkpoint')
    _validate_state(state)
    prior_converged = bool(info.get('prior_converged', True))
    return {
        'api_version': API_VERSION, 'model_version': VERSION,
        'status': 'initialized',
        'validity': {'state_valid': True, 'prior_converged': prior_converged,
                     'checkpoint_reused': bool(info.get('checkpoint_reused', False)),
                     'partial_reasons': [] if prior_converged else ['prior_not_converged']},
        'cutoff_utc': stamp(cutoff),
        'state': state,
        'initialization': info,
        'normalized': {'snapshot': snap},
        'assumptions': list(assumptions) + list(_PHYSIOLOGY_ASSUMPTIONS),
        'provenance': {
            'model_version': VERSION, 'api_version': API_VERSION,
            'configuration': physio, 'model_config_hash': digest(physio),
            'compatibility_configuration': compat,
            'compatibility_configuration_hash': digest(compat),
            'snapshot_hash': digest(snapshot),
            'initialization': info,
            'equation_source': 'sources/circadian/manifest.json',
        },
    }


def advance_state(state, snapshot, *, start_utc, end_utc, config=None):
    """Assimilate the explicit full visible history over ``[start, end]``.

    Contract: the state must sit exactly at the window start (``cutoff=start``
    — its ``at_utc`` for public-API states, and the ``at`` embedded in the
    checkpoint identity for legacy checkpoints), and ``snapshot`` is the FULL
    visible history as of the window end (never an implicit delta). The
    already-consumed prefix must match the checkpoint identity byte-for-byte;
    stale or revised history (including late reports covering consumed spans,
    or config/profile changes) is rejected and the state must be re-initialized.
    Unobserved spans follow the tagged habit prior; full sleep episode
    continuity is preserved across the seam. The returned state is itself a
    valid legacy ``model_checkpoint`` at the new position.
    """
    physio = physiological_configuration(config)
    compat = compatibility_configuration(physio)
    _validate_snapshot(snapshot)
    now = dt(snapshot['now_utc'])
    start = dt(start_utc)
    end = dt(end_utc)
    if end <= start:
        raise ValueError('invalid advance window')
    if end > now:
        raise ValueError('advance window exceeds knowledge cutoff')
    if not isinstance(state, dict):
        raise TypeError('state must be a dict')
    if 'input_prefix_hash' not in state:
        raise ValueError('not a model checkpoint; call initialize first')
    if state.get('model_version') != VERSION:
        raise ValueError('stale or incompatible state: model version')
    if state.get('model_config_hash') is not None:
        if state['model_config_hash'] != digest(physio):
            raise ValueError('stale or incompatible state: configuration')
    elif state.get('configuration_hash') is not None:
        if state['configuration_hash'] != digest(compat):
            raise ValueError('stale or incompatible state: configuration')
    if 'at_utc' in state and stamp(start) != state['at_utc']:
        raise ValueError('advance start does not match the state position (cutoff=start): expected '
                         + str(state['at_utc']))
    _validate_state(state)
    snap, _, _ = normalize(snapshot, [], end)
    if checkpoint_key(snap, start, compat) != state['input_prefix_hash']:
        raise ValueError('stale or revised history: the visible prefix at the window start no longer '
                         'matches the checkpoint identity; re-initialize')
    actual = [dict(e, source='observed') for e in snap['events']
              if e['kind'] == 'sleep_actual' and dt(e['start_utc']) < end and dt(e['end_utc']) > start]
    trace, final = simulate(state, start, end, actual, snap['events'], snap['profile'], compat, history=True)
    advanced = dict(final)
    advanced.update(model_version=VERSION, api_version=API_VERSION,
                    model_config_hash=digest(physio), configuration_hash=digest(compat),
                    input_prefix_hash=checkpoint_key(snap, end, compat))
    if 'initialization_id' in state:
        advanced['initialization_id'] = state['initialization_id']
    _validate_state(advanced)
    return {
        'api_version': API_VERSION, 'model_version': VERSION, 'status': 'advanced',
        'validity': {'state_valid': True, 'partial_reasons': []},
        'start_utc': stamp(start), 'end_utc': stamp(end), 'knowledge_cutoff_utc': stamp(end),
        'state': advanced, 'trace': trace,
        'seam': {'convention': 'drop the previous chunk terminal endpoint row at the seam; keep the next '
                               'chunk opening row (concatenate_traces applies this rule)',
                 'note': 'solver restarts at chunk boundaries can introduce tiny circadian differences vs a '
                         'single window; quantified in docs/module-model-api.md'},
        'assumptions': ['unobserved history filled with habit prior',
                        'advance-window observations are visible as of the window end'],
        'provenance': {
            'model_version': VERSION, 'api_version': API_VERSION,
            'configuration': physio, 'model_config_hash': digest(physio),
            'compatibility_configuration_hash': digest(compat),
            'input_prefix_hash': advanced['input_prefix_hash'],
            'trace_hash': digest(trace),
            'equation_source': 'sources/circadian/manifest.json',
        },
    }


def concatenate_traces(*chunks):
    """Explicit seam rule for chunked traces: discard the earlier chunk's
    terminal endpoint row at the seam and keep the next chunk's opening row.
    Chunks must be adjacent (previous end == next start)."""
    prepared = [list(chunk) for chunk in chunks if chunk]
    if not prepared:
        return []
    if len(prepared) == 1:
        return list(prepared[0])
    result = list(prepared[0])
    for chunk in prepared[1:]:
        if dt(result[-1]['time_utc']) != dt(chunk[0]['time_utc']):
            raise ValueError('trace chunks are not adjacent at the seam')
        result = result[:-1] + chunk
    return result


def advance_pressure(S, hours, *, asleep, config=None):
    """Exact Process S update over a constant sleep/wake interval (search
    component; identical to the simulation's homeostatic update)."""
    physio = physiological_configuration(config)
    return _homeostat_advance(S, hours, asleep, physio['tau_w'], physio['tau_s'])


def advance_inertia(inertia, hours, *, asleep=False, woken_from_hours=None, config=None):
    """Sleep-inertia component for search.

    Zero while asleep. At a wake boundary the onset is
    ``1 - exp(-episode_sleep_hours / inertia_hours)`` — pass the just-ended
    episode's asleep duration as ``woken_from_hours`` (exclude onset latency
    exactly as the forecast does; read it from the config's ``latency_minutes``).
    Otherwise inertia decays as ``inertia * exp(-hours / inertia_hours)``.
    """
    physio = physiological_configuration(config)
    if not 0 <= inertia <= 1:
        raise ValueError('invalid inertia state')
    if not math.isfinite(hours) or hours < 0:
        raise ValueError('invalid duration')
    if asleep:
        return 0.
    decay = math.exp(-hours / physio['inertia_hours'])
    if woken_from_hours is not None:
        if not math.isfinite(woken_from_hours) or woken_from_hours < 0:
            raise ValueError('invalid episode duration')
        return (1 - math.exp(-woken_from_hours / physio['inertia_hours'])) * decay
    return inertia * decay


def burden(S, C, inertia, config=None):
    """Burden composite ``w0*S + w1*C + w2*inertia`` (defaults ``[.6, .25, .15]``).

    Returns ``None`` when the circadian factor is unavailable (oscillator
    amplitude < 1e-3), mirroring the trajectory semantics."""
    physio = physiological_configuration(config)
    if C is None:
        return None
    for value in (S, C, inertia):
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError('non-finite burden input')
    weights = physio['burden_weights']
    return weights[0] * S + weights[1] * C + weights[2] * inertia


def penalty(z, config=None):
    """Softplus threshold penalty in [0, 1] (defaults threshold 0.60, softness 0.05)."""
    if not isinstance(z, (int, float)) or not math.isfinite(z):
        raise ValueError('non-finite penalty input')
    physio = physiological_configuration(config)
    return _softplus_penalty(z, physio)
