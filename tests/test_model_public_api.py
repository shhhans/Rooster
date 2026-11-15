"""Public model API tests for the module decoupling contract step 2.

Covers: neutral time contract re-exports, no-planning-import guarantees,
byte-identical physiology vs the legacy evaluate path, component selection,
config separation (policy keys rejected), checkpoint/state validation,
incremental advance (identity anchors, late-report discrimination, seam
convention and measured solver-restart tolerance), public search helpers,
experiments (explicit identity, closed registry, default isolation).

All facts are synthetic UTC; no wall clock. Existing tests and fixtures are
not modified.
"""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import math
from pathlib import Path
import subprocess
import sys

import pytest

import app.contracts.time as contracts_time
from app.sleep_model import api, evaluate, experiments
from app.sleep_model import homeostat, metrics as model_metrics, normalize as legacy_normalize
from app.sleep_model.contracts import DEFAULTS, PHYSIOLOGICAL_DEFAULTS, digest, physiological_configuration

UTC = timezone.utc
C0 = datetime(2027, 1, 5, 12, tzinfo=UTC)
T1 = datetime(2027, 1, 6, 0, tzinfo=UTC)
T2 = datetime(2027, 1, 6, 12, tzinfo=UTC)
T1H = datetime(2027, 1, 6, 0, 30, tzinfo=UTC)  # non-hour-aligned seam (solver-restart case)


def iso(value):
    return value.astimezone(UTC).isoformat().replace('+00:00', 'Z')


def history_events():
    events = []
    for days_ago in (3, 2, 1):
        wake = C0 - timedelta(days=days_ago) + timedelta(hours=19)
        events.append({'id': f'sleep-{days_ago}', 'kind': 'sleep_actual',
                       'start_utc': iso(wake - timedelta(hours=8)), 'end_utc': iso(wake),
                       'source': 'synthetic_observation', 'episode_id': f'episode-{days_ago}'})
    return events


def snapshot(now, *, events=None, **extra):
    snap = {'snapshot_id': 'public-api-test', 'revision': 1, 'now_utc': iso(now),
            'horizon_end_utc': iso(now + timedelta(days=1)),
            'profile': {'home_timezone': 'UTC', 'habit_bed_local': '23:00',
                        'habit_wake_local': '07:00', 'sleep_need_hours': 8},
            'initial_state': {},
            'events': (history_events() if events is None else events)}
    snap.update(extra)
    return snap


def home_environment(now):
    return {'id': 'home', 'kind': 'environment', 'location': 'home', 'availability': 'available',
            'confirmed': True, 'timezone': 'UTC',
            'start_utc': iso(C0 - timedelta(days=4)), 'end_utc': iso(now + timedelta(days=1))}


def sleep_event(identifier, start, end, **extra):
    event = {'id': identifier, 'kind': 'sleep_actual', 'start_utc': iso(start), 'end_utc': iso(end),
             'source': 'synthetic_observation'}
    event.update(extra)
    return event


def block(identifier, start, end, *, role='main'):
    return {'id': identifier, 'start_utc': iso(start), 'end_utc': iso(end), 'timezone': 'UTC', 'role': role}


# ---------------------------------------------------------------------------
# Neutral time contract (Lead-delegated app/contracts).
# ---------------------------------------------------------------------------

def test_neutral_time_functions_are_reexported_unchanged():
    for name in ('dt', 'stamp', 'interval', 'overlap', 'merge', 'local_instant', 'clock_hours'):
        assert getattr(legacy_normalize, name) is getattr(contracts_time, name), name
        assert getattr(contracts_time, name) is getattr(__import__('app.contracts', fromlist=[name]), name)


def test_neutral_time_function_behavior_is_preserved():
    with pytest.raises(ValueError, match='nonexistent'):
        contracts_time.local_instant('2027-03-14', '02:30', 'America/New_York')
    with pytest.raises(ValueError, match='ambiguous'):
        contracts_time.local_instant('2027-11-07', '01:30', 'America/New_York')
    early = contracts_time.local_instant('2027-11-07', '01:30', 'America/New_York', 0)
    late = contracts_time.local_instant('2027-11-07', '01:30', 'America/New_York', 1)
    assert (late - early).total_seconds() == 3600
    assert contracts_time.clock_hours('07:30') == 7.5
    assert contracts_time.clock_hours(23.5) == 23.5
    with pytest.raises(ValueError, match='habit clock'):
        contracts_time.clock_hours('24:00')
    base = datetime(2027, 1, 5, 12, tzinfo=UTC)
    assert contracts_time.stamp(base) == '2027-01-05T12:00:00Z'
    assert contracts_time.dt('2027-01-05T12:00:00Z') == base
    with pytest.raises(ValueError, match='timezone-aware'):
        contracts_time.dt('2027-01-05T12:00:00')
    with pytest.raises(ValueError, match='invalid interval'):
        contracts_time.interval({'id': 'x', 'start_utc': iso(base), 'end_utc': iso(base)})
    merged = contracts_time.merge([(base, base + timedelta(hours=1)),
                                   (base + timedelta(hours=1), base + timedelta(hours=2))])
    assert merged == [(base, base + timedelta(hours=2))]


# ---------------------------------------------------------------------------
# New compute graph independence (no planning, no evaluation routing).
# ---------------------------------------------------------------------------

def test_new_compute_graph_sources_have_no_planning_tokens():
    root = Path(__file__).resolve().parents[1]
    for relative in ('app/contracts/__init__.py', 'app/contracts/time.py',
                     'app/sleep_model/pipeline.py', 'app/sleep_model/api.py',
                     'app/sleep_model/experiments.py'):
        text = (root / relative).read_text(encoding='utf-8')
        for token in ('app.planning', 'rooster_bench', 'app.sleep_model.evaluation', 'from .evaluation'):
            assert token not in text, f'{token} found in {relative}'


def test_new_compute_graph_imports_no_planning_modules():
    root = Path(__file__).resolve().parents[1]
    code = ("import sys; "
            "import app.sleep_model.api, app.sleep_model.experiments; "
            "bad=[m for m in sys.modules if m.startswith(('app.planning','rooster_bench'))]; "
            "sys.exit(1 if bad else 0)")
    completed = subprocess.run([sys.executable, '-c', code], cwd=str(root),
                               capture_output=True, text=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr


# ---------------------------------------------------------------------------
# Legacy physiology identity (pure code motion must be byte-identical).
# ---------------------------------------------------------------------------

def _legacy_case(now, blocks, window_end, *, mode='forecast'):
    snap = snapshot(now)
    snap['events'].append(home_environment(now))
    return snap, blocks, iso(window_end)


def test_compute_matches_legacy_forecast_physiology_exactly():
    now = C0
    snap = snapshot(now)
    snap['events'].append(home_environment(now))
    blocks = [block('main', now + timedelta(hours=2), now + timedelta(hours=10))]
    legacy = evaluate(snap, blocks, start_utc=iso(now), end_utc=iso(now + timedelta(days=1)), scenarios=False)
    public = api.compute(snap, blocks, start_utc=iso(now), end_utc=iso(now + timedelta(days=1)))
    assert public['trace'] == legacy['trace']
    assert public['initial_state'] == legacy['initial_state']
    assert public['final_state'] == legacy['final_state']
    metrics_only = {k: v for k, v in legacy['physiological_metrics'].items()
                    if k not in ('sleep_target_buckets', 'sleep_count_cost')}
    assert public['physiological_metrics'] == metrics_only
    assert public['physiological_loss'] == legacy['losses']['physiological']
    assert public['predicted_sleep'] == legacy['predicted_sleep']
    assert public['effective_sleep'] == legacy['predicted_sleep']
    assert public['actual_sleep'] == [dict(e, source='observed') for e in public['normalized']['snapshot']['events']
                                      if e['kind'] == 'sleep_actual']
    assert public['provenance']['trace_hash'] == legacy['provenance']['trace_hash']
    assert public['provenance']['compatibility_configuration_hash'] == legacy['provenance']['configuration_hash']
    assert public['provenance']['snapshot_hash'] == legacy['provenance']['snapshot_hash']
    assert public['provenance']['opportunity_hash'] == legacy['provenance']['opportunity_hash']
    assert public['initial_state']['input_prefix_hash'] == legacy['initial_state']['input_prefix_hash']
    assert public['initial_state']['initialization_id'] == legacy['initial_state']['initialization_id']
    # C4: the public input_hash is a different, documented formula (no preferences key).
    assert public['provenance']['input_hash'] != legacy['provenance']['input_hash']


def test_compute_matches_legacy_retrospective_and_ignores_blocks_like_legacy():
    snap = snapshot(C0)
    snap['events'].append(home_environment(C0))
    blocks = [block('main', C0 - timedelta(hours=10), C0 - timedelta(hours=2))]
    kwargs = dict(start_utc=iso(C0 - timedelta(hours=24)), end_utc=iso(C0), mode='retrospective')
    legacy = evaluate(snap, blocks, scenarios=False, **kwargs)
    public = api.compute(snap, blocks, **kwargs)
    assert public['predicted_sleep'] == [] == legacy['predicted_sleep']
    assert public['trace'] == legacy['trace']
    assert public['initial_state'] == legacy['initial_state']
    metrics_only = {k: v for k, v in legacy['physiological_metrics'].items()
                    if k not in ('sleep_target_buckets', 'sleep_count_cost')}
    assert public['physiological_metrics'] == metrics_only
    with pytest.raises(ValueError, match='overlapping sleep opportunities'):
        api.compute(snap, [block('a', C0 - timedelta(hours=10), C0 - timedelta(hours=4)),
                           block('b', C0 - timedelta(hours=5), C0 - timedelta(hours=1))], **kwargs)


def test_compute_does_not_mutate_caller_inputs():
    snap = snapshot(C0)
    snap['events'].append(home_environment(C0))
    blocks = [block('main', C0 + timedelta(hours=2), C0 + timedelta(hours=10))]
    snap_before = deepcopy(snap)
    blocks_before = deepcopy(blocks)
    api.compute(snap, blocks, end_utc=iso(C0 + timedelta(days=1)))
    assert snap == snap_before and blocks == blocks_before


# ---------------------------------------------------------------------------
# Component selection and config separation.
# ---------------------------------------------------------------------------

def test_component_dependency_resolution_and_rejection():
    requested, resolved = api.resolve_components(['metrics'])
    assert requested == ['metrics']
    assert resolved == ['burden', 'circadian', 'inertia', 'metrics', 'pressure']
    requested, resolved = api.resolve_components(None)
    assert requested == resolved == ['burden', 'circadian', 'inertia', 'metrics', 'pressure']
    with pytest.raises(ValueError, match='unknown components: warp'):
        api.resolve_components(['pressure', 'warp'])
    with pytest.raises(ValueError, match='not a string'):
        api.resolve_components('pressure')
    with pytest.raises(ValueError, match='parameters belong in config'):
        api.resolve_components({'pressure': {'tau_w': 1.0}})


def test_compute_reports_requested_and_resolved_components():
    snap = snapshot(C0)
    snap['events'].append(home_environment(C0))
    result = api.compute(snap, components=['metrics'], end_utc=iso(C0 + timedelta(hours=6)))
    assert result['components']['requested'] == ['metrics']
    assert result['components']['resolved'] == ['burden', 'circadian', 'inertia', 'metrics', 'pressure']
    assert result['components']['execution'].startswith('the full physiological solver executes')


def test_compute_rejects_policy_and_unknown_config_keys():
    snap = snapshot(C0)
    with pytest.raises(ValueError, match='unknown physiological parameters: rank_weights'):
        api.compute(snap, config={'rank_weights': [.25, .25, .25, .25]})
    with pytest.raises(ValueError, match='unknown physiological parameters: sleep_count_threshold'):
        api.compute(snap, config={'sleep_count_threshold': 3})
    with pytest.raises(ValueError, match='unknown physiological parameters: bogus'):
        api.compute(snap, config={'bogus': 1.0})
    with pytest.raises(ValueError, match='invalid positive parameter'):
        api.compute(snap, config={'tau_w': 0.0})


def test_compute_validation_errors_are_explicit():
    snap = snapshot(C0)
    snap['events'].append(home_environment(C0))
    with pytest.raises(ValueError, match='invalid evaluation bounds or mode'):
        api.compute(snap, mode='backtest', end_utc=iso(C0 + timedelta(hours=6)))
    with pytest.raises(ValueError, match='invalid evaluation bounds or mode'):
        api.compute(snap, start_utc=iso(C0), end_utc=iso(C0))
    with pytest.raises(ValueError, match='forecast window precedes knowledge cutoff'):
        api.compute(snap, start_utc=iso(C0 - timedelta(hours=6)), end_utc=iso(C0 + timedelta(hours=6)))
    with pytest.raises(ValueError, match='retrospective end exceeds knowledge cutoff'):
        api.compute(snap, start_utc=iso(C0 - timedelta(hours=6)), end_utc=iso(C0 + timedelta(hours=6)),
                    mode='retrospective')
    future = deepcopy(snap)
    future['events'].append(sleep_event('future', C0 + timedelta(hours=2), C0 + timedelta(hours=8)))
    with pytest.raises(ValueError, match='future actual observation'):
        api.compute(future, end_utc=iso(C0 + timedelta(days=1)))
    with pytest.raises(ValueError, match='missing required field: now_utc'):
        api.compute({'horizon_end_utc': iso(C0)})


def test_compute_output_contract_has_no_product_policy():
    snap = snapshot(C0)
    snap['events'].append(home_environment(C0))
    result = api.compute(snap, [block('main', C0 + timedelta(hours=2), C0 + timedelta(hours=10))],
                         end_utc=iso(C0 + timedelta(days=1)))
    for key in ('api_version', 'model_version', 'mode', 'status', 'validity', 'components',
                'knowledge_cutoff_utc', 'evaluation_start_utc', 'evaluation_end_utc', 'normalized',
                'actual_sleep', 'effective_sleep', 'predicted_sleep', 'trace', 'initial_state',
                'final_state', 'physiological_metrics', 'physiological_loss', 'units', 'assumptions',
                'provenance'):
        assert key in result, key
    for forbidden in ('utility', 'publishability', 'preference_metrics', 'constraint_report',
                      'resolved_preferences', 'missed_opportunities', 'losses', 'sensitivity_range'):
        assert forbidden not in result, forbidden
    assert 'sleep_target_buckets' not in result['physiological_metrics']
    assert 'sleep_count_cost' not in result['physiological_metrics']
    for unit_key in ('S', 'inertia', 'burden', 'C', 'penalty', 'sleep_hours', 'excess_area',
                     'phase_rad', 'light_lux', 'time_utc'):
        assert unit_key in result['units'], unit_key
    assert 'hours' in result['units']['sleep_hours']
    assert 'dimensionless' in result['units']['burden']
    assert result['validity']['status'] == result['status'] == 'complete'
    assert result['provenance']['model_config_hash'] == digest(physiological_configuration())
    assert result['provenance']['equation_source'] == 'sources/circadian/manifest.json'
    assert set(physiological_configuration()) == set(PHYSIOLOGICAL_DEFAULTS)


# ---------------------------------------------------------------------------
# initialize: cutoff, checkpoint identity, finite/bounded state.
# ---------------------------------------------------------------------------

def test_initialize_validates_cutoff_and_reuses_checkpoints():
    snap = snapshot(C0)
    snap['events'].append(home_environment(C0))
    with pytest.raises(ValueError, match='cutoff exceeds knowledge cutoff'):
        api.initialize(snap, cutoff_utc=iso(C0 + timedelta(hours=1)))
    first = api.initialize(snap)
    second = api.initialize(snap, checkpoint=first['state'])
    assert second['validity']['checkpoint_reused'] is True
    assert second['state'] == first['state']
    assert second['state']['at_utc'] == iso(C0)
    assert second['state']['api_version'] == api.API_VERSION
    # C3: a checkpoint with a stale/garbage position still reports the true cutoff.
    corrupted = dict(first['state'], at_utc='1999-01-01T00:00:00Z')
    repaired = api.initialize(snap, checkpoint=corrupted)
    assert repaired['state']['at_utc'] == iso(C0)


def test_initialize_rejects_stale_checkpoints_and_corrupt_state():
    snap = snapshot(C0)
    snap['events'].append(home_environment(C0))
    state = api.initialize(snap)['state']
    revised = deepcopy(snap)
    revised['events'][0]['source'] = 'revised-source'
    with pytest.raises(ValueError, match='stale or incompatible checkpoint'):
        api.initialize(revised, checkpoint=state)
    for field, value in (('S', float('nan')), ('x', float('inf')), ('inertia', -0.5), ('sleep_mode', 'yes')):
        corrupt = dict(state, **{field: value})
        with pytest.raises(ValueError):
            api.initialize(snap, checkpoint=corrupt)
    with pytest.raises(ValueError, match='invalid state anchor'):
        api.initialize(snap, checkpoint=dict(state, last_wake_utc='not-a-time'))


def test_initialize_cutoff_hides_unobserved_events_and_ignores_future_candidates():
    base = api.initialize(snapshot(C0))
    late = snapshot(C0)
    late['events'].append(sleep_event('late', C0 - timedelta(hours=3), C0 - timedelta(hours=1),
                                      observed_at_utc=iso(T1)))
    assert api.initialize(late)['state']['input_prefix_hash'] == base['state']['input_prefix_hash']
    future_duty = snapshot(C0)
    future_duty['events'].append({'id': 'duty-future', 'kind': 'duty', 'start_utc': iso(C0 + timedelta(hours=2)),
                                  'end_utc': iso(C0 + timedelta(hours=4)), 'timezone': 'UTC'})
    assert api.initialize(future_duty)['state']['input_prefix_hash'] == base['state']['input_prefix_hash']


def test_initialize_agrees_with_compute_initial_state():
    snap = snapshot(C0)
    snap['events'].append(home_environment(C0))
    state = api.initialize(snap)['state']
    computed = api.compute(snap, [block('main', C0 + timedelta(hours=2), C0 + timedelta(hours=10))],
                           end_utc=iso(C0 + timedelta(days=1)))['initial_state']
    for key in ('S', 'x', 'xc', 'n', 'inertia', 'sleep_mode', 'input_prefix_hash', 'initialization_id',
                'configuration_hash'):
        assert state[key] == computed[key], key


# ---------------------------------------------------------------------------
# advance_state: identity anchors, late reports, seam, continuity.
# ---------------------------------------------------------------------------

def _advance_fixture():
    """Two-chunk assimilation with a wake before each seam (solver restart at T1)."""
    snap0 = snapshot(C0)
    snap0['events'].append(home_environment(C0))
    snap1 = snapshot(T1)
    snap1['events'].append(home_environment(T1))
    snap1['events'].append(sleep_event('day-sleep-1', C0 + timedelta(hours=2), C0 + timedelta(hours=10)))
    snap2 = snapshot(T2)
    snap2['events'].append(home_environment(T2))
    snap2['events'].append(sleep_event('day-sleep-1', C0 + timedelta(hours=2), C0 + timedelta(hours=10)))
    snap2['events'].append(sleep_event('day-sleep-2', T1 + timedelta(hours=2), T1 + timedelta(hours=8)))
    return snap0, snap1, snap2


def test_advance_state_two_chunks_match_single_window_within_measured_tolerance():
    snap0, snap1, snap2 = _advance_fixture()
    init = api.initialize(snap0)
    first = api.advance_state(init['state'], snap1, start_utc=iso(C0), end_utc=iso(T1))
    second = api.advance_state(first['state'], snap2, start_utc=iso(T1), end_utc=iso(T2))
    single = api.advance_state(init['state'], snap2, start_utc=iso(C0), end_utc=iso(T2))
    assert first['state']['at_utc'] == iso(T1) and second['state']['at_utc'] == iso(T2)
    assert first['state']['input_prefix_hash'] != init['state']['input_prefix_hash']
    assert second['state']['input_prefix_hash'] != first['state']['input_prefix_hash']
    # Measured solver-restart tolerance (documented in docs/module-model-api.md).
    for key in ('S', 'x', 'xc', 'n', 'inertia'):
        delta = abs(second['state'][key] - single['state'][key])
        assert delta < (1e-12 if key == 'S' else 1e-6), (key, delta)
    assert second['state']['sleep_mode'] == single['state']['sleep_mode']
    assert second['state']['last_wake_utc'] == single['state']['last_wake_utc']
    assert second['state']['current_episode_start_utc'] == single['state']['current_episode_start_utc']
    joined = api.concatenate_traces(first['trace'], second['trace'])
    assert len(joined) == len(first['trace']) + len(second['trace']) - 1
    times = [p['time_utc'] for p in joined]
    assert len(set(times)) == len(times)
    assert joined[len(first['trace']) - 1] == second['trace'][0]
    assert len(joined) == len(single['trace'])
    worst = 0.0
    for left, right in zip(joined, single['trace']):
        assert left['time_utc'] == right['time_utc']
        assert left['asleep'] == right['asleep']
        for key in ('S', 'x', 'xc', 'n', 'inertia', 'burden'):
            if left[key] is None or right[key] is None:
                assert left[key] is right[key]
            else:
                worst = max(worst, abs(left[key] - right[key]))
    assert worst < 1e-6, worst


def test_advance_state_non_hour_aligned_seam_solver_restart_tolerance():
    """Worst case for seam solver restarts: the chunk boundary does not coincide
    with a wall-clock hour boundary of the simulation grid. Measured deviations
    stay at machine-epsilon level (documented in docs/module-model-api.md)."""
    snap0, snap1, snap2 = _advance_fixture()
    init = api.initialize(snap0)['state']
    snap_mid = snapshot(T1H)
    snap_mid['events'].append(home_environment(T1H))
    snap_mid['events'].append(sleep_event('day-sleep-1', C0 + timedelta(hours=2), C0 + timedelta(hours=10)))
    snap_full = snapshot(T2)
    snap_full['events'].append(home_environment(T2))
    snap_full['events'].append(sleep_event('day-sleep-1', C0 + timedelta(hours=2), C0 + timedelta(hours=10)))
    # The window [C0, T1H] ends 30 minutes off the hour grid.
    first = api.advance_state(init, snap_mid, start_utc=iso(C0), end_utc=iso(T1H))
    second = api.advance_state(first['state'], snap_full, start_utc=iso(T1H), end_utc=iso(T2))
    single = api.advance_state(init, snap_full, start_utc=iso(C0), end_utc=iso(T2))
    for key in ('S', 'x', 'xc', 'n', 'inertia'):
        delta = abs(second['state'][key] - single['state'][key])
        assert delta < 1e-9, (key, delta)
    joined = api.concatenate_traces(first['trace'], second['trace'])
    worst = 0.0
    for left, right in zip(joined, single['trace']):
        assert left['time_utc'] == right['time_utc']
        for key in ('S', 'x', 'xc', 'n', 'inertia', 'burden', 'C'):
            if left[key] is None or right[key] is None:
                assert left[key] is right[key]
            else:
                worst = max(worst, abs(left[key] - right[key]))
    assert worst < 1e-9, worst


def test_advance_state_preserves_episode_continuity_across_the_seam():
    snap0 = snapshot(C0)
    snap0['events'].append(home_environment(C0))
    snap1 = snapshot(T1)
    snap1['events'].append(home_environment(T1))
    snap1['events'].append(sleep_event('late-evening', C0 + timedelta(hours=5), T1))
    snap2 = snapshot(T2)
    snap2['events'].append(home_environment(T2))
    snap2['events'].append(sleep_event('late-evening', C0 + timedelta(hours=5), T1))
    snap2['events'].append(sleep_event('early-morning', T1, T1 + timedelta(hours=2)))
    init = api.initialize(snap0)
    first = api.advance_state(init['state'], snap1, start_utc=iso(C0), end_utc=iso(T1))
    second = api.advance_state(first['state'], snap2, start_utc=iso(T1), end_utc=iso(T2))
    single = api.advance_state(init['state'], snap2, start_utc=iso(C0), end_utc=iso(T2))
    # The person is asleep at the seam; the episode started at 17:00 in chunk 1.
    assert first['state']['sleep_mode'] is True
    assert first['state']['current_episode_start_utc'] == iso(C0 + timedelta(hours=5))
    wake = T1 + timedelta(hours=2)
    assert second['state']['last_wake_utc'] == iso(wake) == single['state']['last_wake_utc']
    # Onset uses the pre-seam episode start (17:00 -> 02:00 = 9h episode), not the chunk start.
    expected_onset = 1 - math.exp(-9 / .5)
    row = next(p for p in second['trace'] if p['time_utc'] == iso(wake + timedelta(minutes=1)))
    single_row = next(p for p in single['trace'] if p['time_utc'] == iso(wake + timedelta(minutes=1)))
    assert abs(row['inertia'] - expected_onset * math.exp(-(1 / 60) / .5)) < 1e-9
    assert abs(row['inertia'] - single_row['inertia']) < 1e-6
    assert second['state']['current_episode_start_utc'] == single['state']['current_episode_start_utc'] is None


def test_advance_state_rejects_wrong_start_config_and_nonfinite_state():
    snap0, snap1, snap2 = _advance_fixture()
    state = api.initialize(snap0)['state']
    with pytest.raises(ValueError, match='cutoff=start'):
        api.advance_state(state, snap1, start_utc=iso(C0 + timedelta(hours=1)), end_utc=iso(T1))
    with pytest.raises(ValueError, match='stale or incompatible state: configuration'):
        api.advance_state(state, snap1, start_utc=iso(C0), end_utc=iso(T1), config={'tau_w': 12.0})
    with pytest.raises(ValueError, match='non-finite model state'):
        api.advance_state(dict(state, S=float('nan')), snap1, start_utc=iso(C0), end_utc=iso(T1))
    with pytest.raises(ValueError, match='advance window exceeds knowledge cutoff'):
        api.advance_state(state, snap0, start_utc=iso(C0), end_utc=iso(C0 + timedelta(hours=6)))
    with pytest.raises(ValueError, match='invalid advance window'):
        api.advance_state(state, snap1, start_utc=iso(T1), end_utc=iso(T1))
    with pytest.raises(ValueError, match='not a model checkpoint'):
        api.advance_state({'S': .5}, snap1, start_utc=iso(C0), end_utc=iso(T1))


def test_advance_state_late_report_discrimination():
    snap0, snap1, _ = _advance_fixture()
    state = api.initialize(snap0)['state']
    # (a) late report covering a consumed span -> reject.
    late_before = deepcopy(snap1)
    late_before['events'].append(sleep_event('late-before', C0 - timedelta(hours=3),
                                             C0 - timedelta(hours=1), observed_at_utc=iso(C0 + timedelta(hours=1))))
    with pytest.raises(ValueError, match='stale or revised history'):
        api.advance_state(state, late_before, start_utc=iso(C0), end_utc=iso(T1))
    # (b) late report covering only the in-window span -> consumable.
    late_inside = deepcopy(snap1)
    late_inside['events'].append(sleep_event('late-inside', C0 + timedelta(minutes=30),
                                             C0 + timedelta(hours=1, minutes=30),
                                             observed_at_utc=iso(C0 + timedelta(hours=2))))
    accepted = api.advance_state(state, late_inside, start_utc=iso(C0), end_utc=iso(T1))
    plain = api.advance_state(state, snap1, start_utc=iso(C0), end_utc=iso(T1))
    asleep_late = sum(p['asleep'] for p in accepted['trace'])
    asleep_plain = sum(p['asleep'] for p in plain['trace'])
    assert asleep_late > asleep_plain
    # (c) a report observed after the window end is invisible to this advance.
    invisible = deepcopy(snap1)
    invisible['events'].append(sleep_event('invisible', C0 + timedelta(hours=16),
                                           C0 + timedelta(hours=18), observed_at_utc=iso(T2)))
    hidden = api.advance_state(state, invisible, start_utc=iso(C0), end_utc=iso(T1))
    assert hidden['trace'] == plain['trace']


def test_advance_state_ongoing_session_appearing_late():
    snap0, snap1, _ = _advance_fixture()
    state = api.initialize(snap0)['state']
    before_start = deepcopy(snap1)
    before_start['sessions'] = {'s': {'id': 's', 'planned_block_id': None,
                                      'actual_start_utc': iso(C0 - timedelta(hours=2)),
                                      'actual_end_utc': None, 'expected_end_utc': iso(T1)}}
    with pytest.raises(ValueError, match='stale or revised history'):
        api.advance_state(state, before_start, start_utc=iso(C0), end_utc=iso(T1))
    inside_window = deepcopy(snap1)
    inside_window['sessions'] = {'s': {'id': 's', 'planned_block_id': None,
                                       'actual_start_utc': iso(C0 + timedelta(hours=1)),
                                       'actual_end_utc': None, 'expected_end_utc': iso(T1)}}
    result = api.advance_state(state, inside_window, start_utc=iso(C0), end_utc=iso(T1))
    assert any(p['asleep'] for p in result['trace'])


def test_advance_state_interoperates_with_legacy_checkpoints_both_directions():
    snap0, snap1, snap2 = _advance_fixture()
    # API state -> legacy model_checkpoint.
    api_state = api.initialize(snap0)['state']
    with_legacy = deepcopy(snap0)
    with_legacy['model_checkpoint'] = api_state
    legacy = evaluate(with_legacy, [], end_utc=iso(C0 + timedelta(hours=6)), scenarios=False)
    assert legacy['initialization']['checkpoint_reused'] is True
    assert legacy['initial_state'] == api_state
    # Legacy state -> API advance.
    legacy_state = evaluate(snap0, [], end_utc=iso(C0 + timedelta(hours=6)), scenarios=False)['initial_state']
    advanced = api.advance_state(legacy_state, snap1, start_utc=iso(C0), end_utc=iso(T1))
    assert advanced['status'] == 'advanced'
    # Advanced API state -> legacy checkpoint at the new position.
    carried = deepcopy(snap1)
    carried['model_checkpoint'] = advanced['state']
    resumed = evaluate(carried, [], end_utc=iso(T1 + timedelta(hours=6)), scenarios=False)
    assert resumed['initialization']['checkpoint_reused'] is True
    assert resumed['initial_state'] == advanced['state']


def test_advance_state_rejects_non_adjacent_windows_and_gaps():
    snap0, snap1, snap2 = _advance_fixture()
    state = api.initialize(snap0)['state']
    first = api.advance_state(state, snap1, start_utc=iso(C0), end_utc=iso(T1))
    with pytest.raises(ValueError, match='cutoff=start'):
        api.advance_state(first['state'], snap2, start_utc=iso(T1 + timedelta(hours=1)), end_utc=iso(T2))
    with pytest.raises(ValueError, match='cutoff=start'):
        api.advance_state(first['state'], snap2, start_utc=iso(C0), end_utc=iso(T2))


def test_concatenate_traces_requires_adjacent_chunks():
    snap0, snap1, snap2 = _advance_fixture()
    state = api.initialize(snap0)['state']
    first = api.advance_state(state, snap1, start_utc=iso(C0), end_utc=iso(T1))
    second = api.advance_state(first['state'], snap2, start_utc=iso(T1), end_utc=iso(T2))
    assert api.concatenate_traces() == []
    assert api.concatenate_traces(first['trace']) == first['trace']
    with pytest.raises(ValueError, match='not adjacent'):
        api.concatenate_traces(first['trace'][1::2], second['trace'])
    joined = api.concatenate_traces(first['trace'], second['trace'])
    assert joined[-1] == second['trace'][-1]
    assert sum(1 for p in joined if p['time_utc'] == iso(T1)) == 1


# ---------------------------------------------------------------------------
# Public search helpers (defaults preserved, no retuning).
# ---------------------------------------------------------------------------

def test_search_helpers_preserve_default_constants_and_formulas():
    assert physiological_configuration()['latency_minutes'] == 15.0  # 0.25 h onset latency
    assert physiological_configuration()['inertia_hours'] == .5
    assert physiological_configuration()['burden_weights'] == [.60, .25, .15]
    assert physiological_configuration()['threshold'] == .60
    assert physiological_configuration()['softness'] == .05
    assert api.advance_pressure(.5, 18.2, asleep=False) == homeostat.advance(.5, 18.2, False, 18.2, 4.2)
    assert api.advance_pressure(.5, 4.2, asleep=True) == homeostat.advance(.5, 4.2, True, 18.2, 4.2)
    assert api.advance_pressure(.5, 2.0, asleep=True, config={'tau_s': 2.0}) == \
        homeostat.advance(.5, 2.0, True, 18.2, 2.0)
    assert api.advance_inertia(.8, 1.0) == pytest.approx(.8 * math.exp(-1.0 / .5))
    assert api.advance_inertia(.5, 0.0, woken_from_hours=1.0) == pytest.approx(1 - math.exp(-1.0 / .5))
    assert api.advance_inertia(.5, 1.0, woken_from_hours=1.0) == \
        pytest.approx((1 - math.exp(-1.0 / .5)) * math.exp(-1.0 / .5))
    assert api.advance_inertia(.9, 1.0, asleep=True) == 0.0
    assert api.advance_inertia(.9, 1.0, config={'inertia_hours': 2.0}) == pytest.approx(.9 * math.exp(-.5))
    with pytest.raises(ValueError):
        api.advance_inertia(1.5, 1.0)
    with pytest.raises(ValueError):
        api.advance_inertia(.5, -1.0)
    assert api.burden(.4, .5, .2) == pytest.approx(.6 * .4 + .25 * .5 + .15 * .2)
    assert api.burden(.4, None, .2) is None
    assert api.burden(.4, .5, .2, config={'burden_weights': [1.0, 0.0, 0.0]}) == pytest.approx(.4)
    with pytest.raises(ValueError):
        api.burden(.4, float('nan'), .2)
    assert api.penalty(.61) == model_metrics.penalty(.61, physiological_configuration())
    values = [api.penalty(z) for z in (0.0, .59, .60, .61, 1.0)]
    assert values == sorted(values) and all(0 <= v <= 1 for v in values)
    with pytest.raises(ValueError):
        api.penalty(float('nan'))


def test_search_burden_matches_simulation_trace_rows():
    snap = snapshot(C0)
    snap['events'].append(home_environment(C0))
    result = api.compute(snap, [block('main', C0 + timedelta(hours=2), C0 + timedelta(hours=10))],
                         end_utc=iso(C0 + timedelta(hours=12)))
    for point in result['trace']:
        if point['C'] is None:
            assert api.burden(point['S'], point['C'], point['inertia']) is None
        else:
            assert api.burden(point['S'], point['C'], point['inertia']) == point['burden']


# ---------------------------------------------------------------------------
# Experiments: explicit identity, closed registry, default isolation.
# ---------------------------------------------------------------------------

def test_experiments_require_explicit_identity_and_known_implementation():
    snap = snapshot(C0)
    snap['events'].append(home_environment(C0))
    kwargs = dict(sleep_blocks=[block('main', C0 + timedelta(hours=2), C0 + timedelta(hours=10))],
                  end_utc=iso(C0 + timedelta(hours=12)))
    with pytest.raises(ValueError, match='experiment_id'):
        experiments.run('', snap, config={}, components=None, **kwargs)
    with pytest.raises(ValueError, match='experiment_id'):
        experiments.run(42, snap, config={}, components=None, **kwargs)
    with pytest.raises(ValueError, match='explicit experiment config'):
        experiments.run('exp', snap, config=None, components=['pressure'], **kwargs)
    with pytest.raises(ValueError, match='explicit experiment components'):
        experiments.run('exp', snap, config={}, components=None, **kwargs)
    with pytest.raises(ValueError, match='unknown experiment implementation'):
        experiments.run('exp', snap, implementation='future', config={}, components=['pressure'], **kwargs)
    with pytest.raises(ValueError, match='unknown components'):
        experiments.run('exp', snap, config={}, components=['warp'], **kwargs)


def test_experiments_reuse_compute_with_traceable_identity_and_no_policy_keys():
    snap = snapshot(C0)
    snap['events'].append(home_environment(C0))
    blocks = [block('main', C0 + timedelta(hours=2), C0 + timedelta(hours=10))]
    kwargs = dict(sleep_blocks=blocks, end_utc=iso(C0 + timedelta(hours=12)),
                  config={}, components=['pressure', 'burden', 'metrics'])
    first = experiments.run('exp-alpha', snap, **kwargs)
    second = experiments.run('exp-alpha', snap, **kwargs)
    direct = api.compute(snap, blocks, end_utc=iso(C0 + timedelta(hours=12)),
                         components=['pressure', 'burden', 'metrics'])
    assert first == second
    assert first['experiment']['experiment_hash'] == second['experiment']['experiment_hash']
    assert {k: v for k, v in first.items() if k != 'experiment'} == direct
    assert first['experiment']['requested_components'] == ['burden', 'metrics', 'pressure']
    assert first['experiment']['model_config_hash'] == direct['provenance']['model_config_hash']
    other = experiments.run('exp-beta', snap, **kwargs)
    assert other['experiment']['experiment_hash'] != first['experiment']['experiment_hash']
    assert other['experiment']['input_hash'] == first['experiment']['input_hash']
    for forbidden in ('utility', 'publishability', 'constraint_report', 'preference_metrics'):
        assert forbidden not in first


def test_experiments_never_mutate_production_defaults():
    snap = snapshot(C0)
    snap['events'].append(home_environment(C0))
    blocks = [block('main', C0 + timedelta(hours=2), C0 + timedelta(hours=10))]
    before_defaults = digest(DEFAULTS)
    plain_before = api.compute(snap, blocks, end_utc=iso(C0 + timedelta(hours=12)))
    experiments.run('override-test', snap, blocks, config={'tau_w': 30.0},
                    components=['pressure', 'metrics'], end_utc=iso(C0 + timedelta(hours=12)))
    assert digest(DEFAULTS) == before_defaults
    plain_after = api.compute(snap, blocks, end_utc=iso(C0 + timedelta(hours=12)))
    assert plain_after == plain_before
