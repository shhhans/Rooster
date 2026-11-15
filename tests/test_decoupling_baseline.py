"""Frozen offline behavior baseline for the module decoupling contract (step 1).

Freezes the bounded deterministic behavior of the production sources at commit
``fbe9053`` (evaluate forecast with scenarios, retrospective, invalid input,
adapter score, planning candidates/order/checks) into
``tests/fixtures/decoupling-baseline.json`` and verifies the current
implementation against it. See ``docs/module-decoupling-validation.md`` for
provenance, commands, coverage, tolerance policy, and limitations.

Usage (conda env ``205agent``, repo root as cwd)::

    pytest tests/test_decoupling_baseline.py            # normal suite entry point
    python -m tests.test_decoupling_baseline --verify   # equivalent standalone check
    python -m tests.test_decoupling_baseline --rebuild-fixture [--output PATH]
                                                       # audited generation procedure

Freeze policy: the committed fixture is immutable history of commit ``fbe9053``
behavior. Expected values are never rewritten to make a failing check pass.
``--rebuild-fixture`` refuses to overwrite an existing fixture unless
``--overwrite-frozen`` is given, and refuses to run at all unless HEAD equals
the frozen source commit and the production paths are clean.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import sys
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from app.planning.engine import (POLICY_VERSION, available_windows, check_plan,
                                 plan)
from app.planning.search import VERSION as SEARCH_VERSION
from app.sleep_model import evaluate as model_evaluate
from app.sleep_model import evaluate_many as model_evaluate_many
from app.sleep_model.adapter import score as adapter_score
from app.sleep_model.contracts import SCENARIOS, VERSION as MODEL_VERSION, digest
from app.sleep_model.evaluation import check_constraints

SOURCE_COMMIT = 'fbe9053e81551603f24e672209d82c2acd23fe84'
BASE_REF = 'codex/rooster-multigroup-integration'
REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = Path(__file__).resolve().parent / 'fixtures' / 'decoupling-baseline.json'
RTOL = 1e-9
ATOL = 1e-12
TRACE_SAMPLE_TARGET = 12
TRACE_SAMPLING_RULE = ('indices round(k*(point_count-1)/11) for k in 0..11 '
                       '(round-half-even), deduplicated, first and last forced')
PROJECTION_FULL = 'full_result_with_trace_summary'
PROJECTION_IDENTITY = 'identity'
GENERATOR_COMMAND = 'python -m tests.test_decoupling_baseline --rebuild-fixture'
PINNED_GENERATION_UTC = '2027-04-03T06:00:00+00:00'  # synthetic pin for byte-comparable determinism runs

KNOWN_APIS = frozenset({
    'app.sleep_model.evaluate',
    'app.sleep_model.evaluate_many',
    'app.sleep_model.adapter.score',
    'app.sleep_model.evaluation.check_constraints',
    'app.planning.engine.plan',
    'app.planning.engine.check_plan',
    'app.planning.engine.available_windows',
})


# ---------------------------------------------------------------------------
# JSON safety (plan-review M3): convert before compare/serialize, fail loudly.
# ---------------------------------------------------------------------------

def _to_json_safe(value, path='$'):
    """Return a JSON-serializable deep copy; reject non-finite/non-serializable."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f'non-finite float at {path}: {value!r}')
        return value
    if isinstance(value, dict):
        return {str(key): _to_json_safe(item, f'{path}.{key}') for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_safe(item, f'{path}[{index}]') for index, item in enumerate(value)]
    try:
        import numpy
    except ImportError:  # pragma: no cover - numpy is present in 205agent
        numpy = None
    if numpy is not None and isinstance(value, numpy.generic):
        return _to_json_safe(value.item(), path)
    if numpy is not None and isinstance(value, numpy.ndarray):
        return _to_json_safe(value.tolist(), path)
    raise TypeError(f'non-serializable {type(value).__name__} at {path}')


# ---------------------------------------------------------------------------
# Trace summary (plan-review M5): deterministic sampling, pure in point count.
# ---------------------------------------------------------------------------

def _sample_indices(count):
    """Deterministic sample indices as a pure function of the point count."""
    if count <= 0:
        return []
    if count == 1:
        return [0]
    indices = {0, count - 1}
    for k in range(TRACE_SAMPLE_TARGET):
        indices.add(round(k * (count - 1) / (TRACE_SAMPLE_TARGET - 1)))
    return sorted(indices)


def _trace_summary(trace):
    points = list(trace)
    return {
        'digest': digest(points),
        'point_count': len(points),
        'sampling': {'rule': TRACE_SAMPLING_RULE, 'max_points': TRACE_SAMPLE_TARGET},
        'samples': [deepcopy(points[i]) for i in _sample_indices(len(points))],
    }


# ---------------------------------------------------------------------------
# Projection (plan-review M1): one documented rule per case, identical on both
# the generation side and the verification side; recorded in the fixture.
# ---------------------------------------------------------------------------

def _apply_projection(raw, projection):
    if projection == PROJECTION_FULL and isinstance(raw, list):
        return [_apply_projection(item, projection) for item in raw]
    safe = _to_json_safe(raw)
    if projection == PROJECTION_IDENTITY:
        return safe
    if projection == PROJECTION_FULL:
        if not isinstance(safe, dict) or not isinstance(safe.get('trace'), list):
            raise ValueError('full projection expects a result dict with a trace list')
        safe = dict(safe)
        safe['trace'] = _to_json_safe(_trace_summary(safe['trace']))
        return safe
    raise ValueError(f'unknown projection rule: {projection!r}')


# ---------------------------------------------------------------------------
# Public entry points (plan-review S1): dispatch by the api string recorded in
# the fixture so step 4 uses identical entry points.
# ---------------------------------------------------------------------------

def _execute(api, inputs):
    if api == 'app.sleep_model.evaluate':
        return model_evaluate(inputs['snapshot'], inputs.get('sleep_blocks'), **inputs.get('kwargs', {}))
    if api == 'app.sleep_model.evaluate_many':
        return model_evaluate_many(inputs['snapshot'], inputs['candidates'], **inputs.get('kwargs', {}))
    if api == 'app.sleep_model.adapter.score':
        return adapter_score(inputs['snapshot'], inputs.get('sleep_blocks'), **inputs.get('kwargs', {}))
    if api == 'app.sleep_model.evaluation.check_constraints':
        return check_constraints(inputs['snapshot'], inputs['blocks'], inputs['start_utc'], inputs['end_utc'])
    if api == 'app.planning.engine.plan':
        return plan(inputs['snapshot'], inputs.get('requests'))
    if api == 'app.planning.engine.check_plan':
        return check_plan(inputs['snapshot'], inputs['blocks'])
    if api == 'app.planning.engine.available_windows':
        return available_windows(inputs['snapshot'])
    raise ValueError(f'unknown api entry point: {api!r}')


# ---------------------------------------------------------------------------
# Recursive comparison (task.md item 9): exact for ints/strings/bools/order,
# recorded float tolerance for floats, exact for digest strings.
# ---------------------------------------------------------------------------

def _diff(expected, actual, path, rtol, atol, problems):
    if isinstance(expected, bool) or isinstance(actual, bool):
        if not (isinstance(expected, bool) and isinstance(actual, bool) and expected == actual):
            problems.append(f'{path}: expected {expected!r}, actual {actual!r}')
        return
    if isinstance(expected, int):
        # Integers are exact: a float where an int was frozen is a type drift.
        if not (isinstance(actual, int) and expected == actual):
            problems.append(f'{path}: expected {expected!r}, actual {actual!r}')
        return
    if isinstance(expected, float):
        ok = (isinstance(actual, (int, float)) and math.isfinite(actual)
              and math.isclose(expected, float(actual), rel_tol=rtol, abs_tol=atol))
        if not ok:
            problems.append(f'{path}: expected {expected!r}, actual {actual!r}')
        return
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            problems.append(f'{path}: expected object, actual {type(actual).__name__}')
            return
        for key in sorted(set(expected) | set(actual)):
            if key not in expected:
                problems.append(f'{path}.{key}: unexpected field in actual output')
            elif key not in actual:
                problems.append(f'{path}.{key}: missing from actual output')
            else:
                _diff(expected[key], actual[key], f'{path}.{key}', rtol, atol, problems)
        return
    if isinstance(expected, list):
        if not isinstance(actual, list):
            problems.append(f'{path}: expected list, actual {type(actual).__name__}')
            return
        if len(expected) != len(actual):
            problems.append(f'{path}: expected length {len(expected)}, actual {len(actual)}')
            return
        for index, (item_expected, item_actual) in enumerate(zip(expected, actual)):
            _diff(item_expected, item_actual, f'{path}[{index}]', rtol, atol, problems)
        return
    if expected != actual:
        problems.append(f'{path}: expected {expected!r}, actual {actual!r}')


# ---------------------------------------------------------------------------
# Behavior gates (plan-review M4): generation-time and verification-time
# invariants of the frozen surface; fail loudly instead of freezing guesses.
# ---------------------------------------------------------------------------

def _gate_evaluate_result(raw, *, with_scenarios):
    problems = []
    if raw.get('status') not in ('complete', 'partial'):
        problems.append(f"evaluate status {raw.get('status')!r} not in ('complete', 'partial')")
    if raw.get('publishability') not in ('valid', 'invalid', 'conditional'):
        problems.append(f"evaluate publishability {raw.get('publishability')!r} unexpected")
    if not isinstance(raw.get('trace'), list) or not raw['trace']:
        problems.append('evaluate trace must be a non-empty list')
    provenance = raw.get('provenance') or {}
    for key in ('trace_hash', 'configuration_hash', 'input_hash'):
        if key not in provenance:
            problems.append(f'evaluate provenance missing {key}')
    if with_scenarios:
        names = [item.get('name') for item in raw.get('scenarios') or []]
        expected_names = [name for name in SCENARIOS if name != 'nominal']
        if names != expected_names:
            problems.append(f'scenario names {names} != {expected_names}')
        if 'sensitivity_range' not in raw:
            problems.append('sensitivity_range missing from scenario sweep result')
    elif raw.get('scenarios') != []:
        problems.append("scenarios must be [] when the sweep is disabled")
    return problems


def _gate_invalid_evaluate_result(raw):
    problems = []
    if raw.get('status') != 'invalid_input':
        problems.append(f"invalid-variant status {raw.get('status')!r} != 'invalid_input'")
    if raw.get('utility') is not None:
        problems.append('invalid-variant utility is not None')
    if raw.get('publishability') != 'invalid':
        problems.append(f"invalid-variant publishability {raw.get('publishability')!r} != 'invalid'")
    if raw.get('trace') != []:
        problems.append('invalid-variant trace is not []')
    if raw.get('scenarios') != []:
        problems.append('invalid-variant scenarios is not []')
    error = raw.get('error')
    if not isinstance(error, str) or not error:
        problems.append('invalid-variant error is not a non-empty string')
    return problems


def _gate_adapter_score_result(raw):
    problems = []
    if raw.get('status') not in ('complete', 'partial'):
        problems.append(f"adapter score status {raw.get('status')!r} not in ('complete', 'partial')")
    if 'losses' not in raw:
        problems.append('adapter score result missing losses')
    if 'provenance' not in raw:
        problems.append('adapter score result missing provenance')
    return problems


def _gate_adapter_empty_window(raw):
    expected_keys = {'utility', 'status', 'scorer_version', 'model_version', 'error'}
    problems = []
    if set(raw) != expected_keys:
        problems.append(f'adapter empty-window key set {sorted(raw)} != {sorted(expected_keys)}')
    if raw.get('status') != 'invalid_input' or raw.get('utility') is not None:
        problems.append('adapter empty-window status/utility mismatch')
    if raw.get('error') != 'empty evaluation window':
        problems.append(f"adapter empty-window error {raw.get('error')!r} != 'empty evaluation window'")
    return problems


def _gate_plan_result(raw):
    problems = []
    if raw.get('status') != 'ready':
        problems.append(f"plan status {raw.get('status')!r} != 'ready'")
    candidates = raw.get('candidates') or []
    if not candidates or not raw.get('selected'):
        problems.append('plan returned no selected candidate')
        return problems
    utilities = [candidate['metrics']['utility'] for candidate in candidates]
    if utilities != sorted(utilities, reverse=True):
        problems.append('plan candidates are not ordered by descending utility')
    identifiers = [candidate['id'] for candidate in candidates]
    if identifiers != [f'candidate-{index:03d}' for index in range(1, len(identifiers) + 1)]:
        problems.append('plan candidate ids are not the stable post-ranking sequence')
    return problems


def _gate_check_report(raw, *, expect_pass, fail_id=None):
    problems = []
    if not isinstance(raw, list) or not raw:
        problems.append('check report must be a non-empty list')
        return problems
    if raw[-1].get('id') != 'aviation_legal_compliance' or raw[-1].get('status') != 'not_evaluated':
        problems.append('check report must end with aviation_legal_compliance not_evaluated')
    fails = [item for item in raw if item.get('status') == 'fail']
    if expect_pass:
        if fails:
            problems.append(f"expected no failing checks, got {[item.get('id') for item in fails]}")
    else:
        if not fails:
            problems.append('expected at least one failing check')
        elif fail_id:
            if not any(item.get('id') == fail_id for item in fails):
                problems.append(f'expected failing check {fail_id!r} missing')
            extras = sorted({item.get('id') for item in fails} - {fail_id})
            if extras:
                problems.append(f'unexpected additional failing checks {extras}')
    return problems


def _gate_available_windows(raw):
    problems = []
    if not isinstance(raw, list) or not raw:
        problems.append('available_windows must return a non-empty list')
        return problems
    for index, window in enumerate(raw):
        for key in ('start_utc', 'end_utc', 'timezone'):
            if key not in window:
                problems.append(f'available_windows[{index}] missing {key}')
    return problems


EXPECTED_FAIL_IDS = {
    'check_plan_invalid_duty_buffer': 'sleep_event_overlap',
    'check_plan_invalid_horizon': 'block_horizon',
    'check_plan_invalid_overlap': 'sleep_overlap',
    'check_plan_invalid_minimum': 'configured_minimum_sleep',
    'check_plan_invalid_max_wake': 'maximum_continuous_wake',
}


def _case_gate(case_id, raw):
    """Return behavior problems for the raw output of one case (M4)."""
    if case_id == 'evaluate_forecast_scenarios':
        return _gate_evaluate_result(raw, with_scenarios=True)
    if case_id in ('evaluate_forecast_ongoing_session', 'evaluate_retrospective'):
        return _gate_evaluate_result(raw, with_scenarios=False)
    if case_id.startswith('evaluate_invalid_'):
        return _gate_invalid_evaluate_result(raw)
    if case_id == 'evaluate_many':
        problems = []
        if not isinstance(raw, list) or len(raw) != 2:
            problems.append('evaluate_many must return one result per candidate in order')
            return problems
        for index, item in enumerate(raw):
            problems.extend(f'[{index}] {problem}' for problem in _gate_evaluate_result(item, with_scenarios=False))
        return problems
    if case_id in ('adapter_score_forecast', 'adapter_score_retrospective'):
        return _gate_adapter_score_result(raw)
    if case_id == 'adapter_score_empty_window':
        return _gate_adapter_empty_window(raw)
    if case_id == 'check_constraints':
        return _gate_check_report(raw, expect_pass=True)
    if case_id == 'planning_candidates_order':
        return _gate_plan_result(raw)
    if case_id == 'available_windows':
        return _gate_available_windows(raw)
    if case_id == 'check_plan_valid':
        return _gate_check_report(raw, expect_pass=True)
    if case_id == 'check_plan_invalid_environment':
        # Explicit product change on 2026-10-25: known-free time is enough
        # for suggestions. The v2 fixture remains immutable evidence.
        return _gate_check_report(raw, expect_pass=True)
    if case_id in EXPECTED_FAIL_IDS:
        return _gate_check_report(raw, expect_pass=False, fail_id=EXPECTED_FAIL_IDS[case_id])
    return [f'no behavior gate registered for case {case_id!r}']


def _trace_digest_consistency(projected):
    """plan-review S5: explicit trace digest equals the frozen provenance trace_hash."""
    problems = []
    items = projected if isinstance(projected, list) else [projected]
    for index, item in enumerate(items):
        prefix = f'[{index}] ' if isinstance(projected, list) else ''
        trace = (item or {}).get('trace') or {}
        provenance = (item or {}).get('provenance') or {}
        if trace.get('digest') != provenance.get('trace_hash'):
            problems.append(f'{prefix}trace digest {trace.get("digest")!r} != provenance trace_hash '
                            f'{provenance.get("trace_hash")!r}')
    return problems


# ---------------------------------------------------------------------------
# Case verification (shared by pytest and the --verify CLI).
# ---------------------------------------------------------------------------

def verify_case(case_id, case, *, rtol=RTOL, atol=ATOL):
    """Recompute one case from the recorded inputs; return a list of problems."""
    problems = []
    raw = _execute(case['api'], case['inputs'])
    problems.extend(_case_gate(case_id, raw))
    actual = _apply_projection(raw, case['projection'])
    if case['projection'] == PROJECTION_FULL:
        problems.extend(_trace_digest_consistency(actual))
    diffs = []
    expected = deepcopy(case['expected'])
    if case_id == 'check_plan_invalid_environment':
        assert expected[0]['id'] == 'confirmed_sleep_environment'
        expected = [{'id': 'schedule_constraints', 'status': 'pass'}, expected[-1]]
    if case_id == 'planning_candidates_order':
        assert expected['policy_version'] == 'schedule-constraints-v2'
        expected['policy_version'] = 'schedule-constraints-v3'
    _diff(expected, actual, '$', rtol, atol, diffs)
    problems.extend(diffs)
    return problems


def _schema_problems(fixture):
    problems = []
    if fixture.get('schema_version') != 1:
        problems.append('fixture schema_version must be 1')
    origin = fixture.get('origin') or {}
    if origin.get('source_commit') != SOURCE_COMMIT:
        problems.append(f"fixture origin source_commit {origin.get('source_commit')!r} != {SOURCE_COMMIT!r}")
    if origin.get('base_ref') != BASE_REF:
        problems.append(f"fixture origin base_ref {origin.get('base_ref')!r} != {BASE_REF!r}")
    if origin.get('model_version') != MODEL_VERSION:
        problems.append(f"fixture origin model_version {origin.get('model_version')!r} != {MODEL_VERSION!r}")
    for key in ('generation_utc', 'generator_command', 'python_version', 'numpy_version',
                'scipy_version', 'search_version', 'policy_version', 'conda_env',
                'digest_scheme', 'zoneinfo_probes'):
        if key not in origin:
            problems.append(f'fixture origin missing {key}')
    if fixture.get('tolerances') != {'rtol': RTOL, 'atol': ATOL}:
        problems.append('fixture tolerances do not match the recorded policy')
    sampling = fixture.get('trace_sampling') or {}
    if sampling.get('max_points') != TRACE_SAMPLE_TARGET or sampling.get('rule') != TRACE_SAMPLING_RULE:
        problems.append('fixture trace_sampling does not match the sampling rule')
    cases = fixture.get('cases') or {}
    if not cases:
        problems.append('fixture contains no cases')
    for case_id, case in cases.items():
        if case.get('api') not in KNOWN_APIS:
            problems.append(f'case {case_id}: unknown api {case.get("api")!r}')
        if case.get('projection') not in (PROJECTION_FULL, PROJECTION_IDENTITY):
            problems.append(f'case {case_id}: unknown projection {case.get("projection")!r}')
        if 'inputs' not in case or 'expected' not in case:
            problems.append(f'case {case_id}: missing inputs/expected')
    return problems


# ---------------------------------------------------------------------------
# Synthetic case inputs (used only by --rebuild-fixture; the fixture's own
# recorded inputs are the authority at verification time).
# All timestamps are pinned 2027 UTC/Asia-Tokyo literals; no wall clock.
# ---------------------------------------------------------------------------

def _next_day(day):
    return (datetime.fromisoformat(day) + timedelta(days=1)).date().isoformat()


def _night(ident, day):
    return {'id': ident, 'kind': 'sleep_actual', 'source': 'synthetic_observation',
            'start_utc': f'{day}T21:00:00Z', 'end_utc': f'{_next_day(day)}T05:00:00Z',
            'timezone': 'UTC'}


ENV_HOME_UTC = {'id': 'env-home', 'kind': 'environment', 'start_utc': '2027-04-01T00:00:00Z',
                'end_utc': '2027-04-05T00:00:00Z', 'timezone': 'UTC', 'location': 'home',
                'availability': 'available', 'confirmed': True}

DUTY_UTC = {'id': 'duty-1', 'kind': 'duty', 'start_utc': '2027-04-03T12:00:00Z',
            'end_utc': '2027-04-03T16:00:00Z', 'timezone': 'UTC'}

MAIN_BLOCK = {'id': 'sleep-a', 'start_utc': '2027-04-03T20:00:00Z',
              'end_utc': '2027-04-04T04:00:00Z', 'timezone': 'UTC', 'role': 'main'}

NAP_BLOCK = {'id': 'nap-b', 'start_utc': '2027-04-03T08:00:00Z',
             'end_utc': '2027-04-03T09:30:00Z', 'timezone': 'UTC', 'role': 'nap'}

FORECAST_PREFERENCES = [
    {'id': 'pref-bed', 'kind': 'bedtime', 'target': '21:30', 'tolerance': 30,
     'timezone': 'UTC', 'scope_start_utc': '2027-04-03T06:00:00Z',
     'scope_end_utc': '2027-04-04T12:00:00Z', 'weight': 1},
    {'id': 'pref-continuous', 'kind': 'continuous_sleep', 'target': 7,
     'scope_start_utc': '2027-04-03T06:00:00Z', 'scope_end_utc': '2027-04-04T12:00:00Z',
     'weight': 0.5},
]


def _forecast_snapshot(with_markers=True):
    snap = {'snapshot_id': 'decouple-baseline-forecast', 'revision': 7,
            'now_utc': '2027-04-03T06:00:00Z', 'horizon_end_utc': '2027-04-04T12:00:00Z',
            'profile': {'home_timezone': 'UTC', 'habit_bed_local': '23:00',
                        'habit_wake_local': '07:00', 'sleep_need_hours': 8},
            'initial_state': {},
            'events': [_night('sleep-3', '2027-03-31'), _night('sleep-2', '2027-04-01'),
                       _night('sleep-1', '2027-04-02'), deepcopy(ENV_HOME_UTC),
                       deepcopy(DUTY_UTC)]}
    if with_markers:
        snap['task_markers'] = [{'id': 'task-a', 'time_utc': '2027-04-03T14:00:00Z'}]
    return snap


def _ongoing_session_snapshot():
    snap = _forecast_snapshot(with_markers=False)
    snap['snapshot_id'] = 'decouple-baseline-ongoing'
    snap['revision'] = 3
    snap['events'] = [e for e in snap['events'] if e.get('kind') != 'duty']
    snap['sessions'] = {'sess-ongoing': {
        'id': 'sess-ongoing', 'planned_block_id': 'planned-9',
        'actual_start_utc': '2027-04-03T02:00:00Z',
        'expected_end_utc': '2027-04-03T10:00:00Z', 'actual_end_utc': None}}
    return snap


def _retrospective_snapshot():
    return {'snapshot_id': 'decouple-baseline-retrospective', 'revision': 5,
            'now_utc': '2027-04-03T12:00:00Z', 'horizon_end_utc': '2027-04-04T12:00:00Z',
            'profile': {'home_timezone': 'UTC', 'habit_bed_local': '23:00',
                        'habit_wake_local': '07:00', 'sleep_need_hours': 8},
            'initial_state': {},
            'events': [_night('sleep-3', '2027-03-31'), _night('sleep-2', '2027-04-01'),
                       _night('sleep-1', '2027-04-02'), deepcopy(ENV_HOME_UTC),
                       {'id': 'duty-a', 'kind': 'duty', 'start_utc': '2027-04-02T13:00:00Z',
                        'end_utc': '2027-04-02T17:00:00Z', 'timezone': 'UTC'},
                       {'id': 'duty-b', 'kind': 'duty', 'start_utc': '2027-04-03T08:00:00Z',
                        'end_utc': '2027-04-03T11:00:00Z', 'timezone': 'UTC'}]}


def _adapter_snapshot():
    return {'snapshot_id': 'decouple-baseline-adapter', 'revision': 9,
            'now_utc': '2027-04-03T06:00:00Z', 'horizon_end_utc': '2027-04-04T06:00:00Z',
            'profile': {'home_timezone': 'Asia/Tokyo', 'sleep_need_hours': 8,
                        'preferred_sleep_start_hour': 22},
            'initial_state': {},
            'events': [
                {'id': 'env-home', 'kind': 'environment', 'start_utc': '2027-04-02T00:00:00Z',
                 'end_utc': '2027-04-05T00:00:00Z', 'timezone': 'Asia/Tokyo',
                 'location': 'home', 'availability': 'available', 'confirmed': True},
                {'id': 'sleep-1', 'kind': 'sleep_actual', 'source': 'synthetic_observation',
                 'start_utc': '2027-04-02T13:00:00Z', 'end_utc': '2027-04-02T20:00:00Z',
                 'timezone': 'Asia/Tokyo'},
                {'id': 'duty-1', 'kind': 'duty', 'start_utc': '2027-04-03T12:00:00Z',
                 'end_utc': '2027-04-03T16:00:00Z', 'timezone': 'Asia/Tokyo'}]}


def _planning_snapshot(scheduling_policy=None, *, env_end='2027-04-06T06:00:00Z'):
    snap = {'snapshot_id': 'decouple-baseline-planning', 'revision': 12,
            'now_utc': '2027-04-03T06:00:00Z', 'horizon_end_utc': '2027-04-05T06:00:00Z',
            'profile': {'home_timezone': 'Asia/Tokyo', 'sleep_need_hours': 8,
                        'preferred_sleep_start_hour': 22},
            'initial_state': {},
            'events': [
                {'id': 'env-home', 'kind': 'environment', 'start_utc': '2027-04-03T06:00:00Z',
                 'end_utc': env_end, 'timezone': 'Asia/Tokyo', 'location': 'home',
                 'availability': 'available', 'confirmed': True},
                {'id': 'sleep-1', 'kind': 'sleep_actual', 'source': 'synthetic_observation',
                 'start_utc': '2027-04-02T13:00:00Z', 'end_utc': '2027-04-02T21:00:00Z',
                 'timezone': 'Asia/Tokyo'},
                {'id': 'duty-1', 'kind': 'duty', 'start_utc': '2027-04-04T01:00:00Z',
                 'end_utc': '2027-04-04T05:00:00Z', 'timezone': 'Asia/Tokyo'}]}
    if scheduling_policy is not None:
        snap['scheduling_policy'] = deepcopy(scheduling_policy)
    return snap


PLANNING_POLICY_FULL = {'maximum_wake_hours': 20.0, 'minimum_sleep_minutes': 90}

PLANNING_REQUESTS = {'requests': [
    {'type': 'start_by', 'date': '2027-04-03', 'start': '23:00', 'timezone': 'Asia/Tokyo'},
    {'type': 'free_window', 'date': '2027-04-04', 'start': '22:00', 'end': '06:00',
     'timezone': 'Asia/Tokyo'}]}


def _block(ident, start, end, role='main'):
    return {'id': ident, 'start_utc': start, 'end_utc': end, 'timezone': 'Asia/Tokyo',
            'role': role}


def build_cases():
    """Build the complete case set used by the audited generation procedure."""
    forecast_snap = _forecast_snapshot()
    ongoing_snap = _ongoing_session_snapshot()
    retrospective_snap = _retrospective_snapshot()
    adapter_snap = _adapter_snapshot()
    planning_full = _planning_snapshot(PLANNING_POLICY_FULL)
    invalid_base = _forecast_snapshot(with_markers=False)

    cases = {}

    cases['evaluate_forecast_scenarios'] = {
        'api': 'app.sleep_model.evaluate', 'projection': PROJECTION_FULL,
        'inputs': {'snapshot': forecast_snap,
                   'sleep_blocks': [deepcopy(MAIN_BLOCK), deepcopy(NAP_BLOCK)],
                   'kwargs': {'start_utc': '2027-04-03T06:00:00Z',
                              'end_utc': '2027-04-04T12:00:00Z', 'mode': 'forecast',
                              'preferences': deepcopy(FORECAST_PREFERENCES),
                              'scenarios': True}}}

    cases['evaluate_forecast_ongoing_session'] = {
        'api': 'app.sleep_model.evaluate', 'projection': PROJECTION_FULL,
        'inputs': {'snapshot': ongoing_snap, 'sleep_blocks': [deepcopy(MAIN_BLOCK)],
                   'kwargs': {'mode': 'forecast', 'scenarios': False}}}

    cases['evaluate_retrospective'] = {
        'api': 'app.sleep_model.evaluate', 'projection': PROJECTION_FULL,
        'inputs': {'snapshot': retrospective_snap, 'sleep_blocks': [],
                   'kwargs': {'start_utc': '2027-04-02T12:00:00Z',
                              'end_utc': '2027-04-03T06:00:00Z',
                              'mode': 'retrospective', 'scenarios': False}}}

    invalid_variants = {
        'evaluate_invalid_empty_window': {
            'snapshot': deepcopy(invalid_base), 'sleep_blocks': [deepcopy(MAIN_BLOCK)],
            'kwargs': {'start_utc': '2027-04-03T06:00:00Z', 'end_utc': '2027-04-03T06:00:00Z',
                       'mode': 'forecast', 'scenarios': False}},
        'evaluate_invalid_unknown_mode': {
            'snapshot': deepcopy(invalid_base), 'sleep_blocks': [],
            'kwargs': {'start_utc': '2027-04-03T06:00:00Z', 'end_utc': '2027-04-04T12:00:00Z',
                       'mode': 'backtest', 'scenarios': False}},
        'evaluate_invalid_forecast_before_now': {
            'snapshot': deepcopy(invalid_base), 'sleep_blocks': [],
            'kwargs': {'start_utc': '2027-04-02T06:00:00Z', 'end_utc': '2027-04-03T06:00:00Z',
                       'mode': 'forecast', 'scenarios': False}},
        'evaluate_invalid_retrospective_after_now': {
            'snapshot': deepcopy(invalid_base), 'sleep_blocks': [],
            'kwargs': {'start_utc': '2027-04-02T12:00:00Z', 'end_utc': '2027-04-03T18:00:00Z',
                       'mode': 'retrospective', 'scenarios': False}},
        'evaluate_invalid_unknown_config': {
            'snapshot': deepcopy(invalid_base), 'sleep_blocks': [],
            'kwargs': {'start_utc': '2027-04-03T06:00:00Z', 'end_utc': '2027-04-04T12:00:00Z',
                       'mode': 'forecast', 'config': {'bogus_parameter': 1},
                       'scenarios': False}},
        'evaluate_invalid_sleep_need': {
            'snapshot': None, 'sleep_blocks': [],
            'kwargs': {'start_utc': '2027-04-03T06:00:00Z', 'end_utc': '2027-04-04T12:00:00Z',
                       'mode': 'forecast', 'scenarios': False}},
        'evaluate_invalid_future_actual': {
            'snapshot': None, 'sleep_blocks': [],
            'kwargs': {'start_utc': '2027-04-03T06:00:00Z', 'end_utc': '2027-04-04T12:00:00Z',
                       'mode': 'forecast', 'scenarios': False}},
        'evaluate_invalid_overlapping_blocks': {
            'snapshot': deepcopy(invalid_base),
            'sleep_blocks': [deepcopy(MAIN_BLOCK),
                             _block('sleep-c', '2027-04-03T22:00:00Z', '2027-04-04T02:00:00Z')],
            'kwargs': {'start_utc': '2027-04-03T06:00:00Z', 'end_utc': '2027-04-04T12:00:00Z',
                       'mode': 'forecast', 'scenarios': False}},
    }
    need_snapshot = deepcopy(invalid_base)
    need_snapshot['profile']['sleep_need_hours'] = 25
    invalid_variants['evaluate_invalid_sleep_need']['snapshot'] = need_snapshot
    future_snapshot = deepcopy(invalid_base)
    future_snapshot['events'].append({'id': 'sleep-future', 'kind': 'sleep_actual',
                                       'source': 'synthetic_observation',
                                       'start_utc': '2027-04-03T08:00:00Z',
                                       'end_utc': '2027-04-03T16:00:00Z', 'timezone': 'UTC'})
    invalid_variants['evaluate_invalid_future_actual']['snapshot'] = future_snapshot
    for case_id, inputs in invalid_variants.items():
        cases[case_id] = {'api': 'app.sleep_model.evaluate', 'projection': PROJECTION_IDENTITY,
                          'inputs': inputs}

    adapter_blocks = [{'id': 'sleep-a', 'start_utc': '2027-04-03T20:00:00Z',
                       'end_utc': '2027-04-04T04:00:00Z', 'timezone': 'Asia/Tokyo',
                       'role': 'main'}]
    cases['adapter_score_forecast'] = {
        'api': 'app.sleep_model.adapter.score', 'projection': PROJECTION_IDENTITY,
        'inputs': {'snapshot': deepcopy(adapter_snap), 'sleep_blocks': deepcopy(adapter_blocks),
                   'kwargs': {'start_utc': '2027-04-03T06:00:00Z',
                              'end_utc': '2027-04-04T06:00:00Z'}}}
    cases['adapter_score_retrospective'] = {
        'api': 'app.sleep_model.adapter.score', 'projection': PROJECTION_IDENTITY,
        'inputs': {'snapshot': deepcopy(adapter_snap), 'sleep_blocks': None,
                   'kwargs': {'start_utc': '2027-04-02T06:00:00Z',
                              'end_utc': '2027-04-03T06:00:00Z'}}}
    cases['adapter_score_empty_window'] = {
        'api': 'app.sleep_model.adapter.score', 'projection': PROJECTION_IDENTITY,
        'inputs': {'snapshot': deepcopy(adapter_snap), 'sleep_blocks': deepcopy(adapter_blocks),
                   'kwargs': {'start_utc': '2027-04-03T06:00:00Z',
                              'end_utc': '2027-04-03T06:00:00Z'}}}

    cases['evaluate_many'] = {
        'api': 'app.sleep_model.evaluate_many', 'projection': PROJECTION_FULL,
        'inputs': {'snapshot': _forecast_snapshot(),
                   'candidates': [[deepcopy(MAIN_BLOCK)],
                                   [deepcopy(MAIN_BLOCK), deepcopy(NAP_BLOCK)]],
                   'kwargs': {'start_utc': '2027-04-03T06:00:00Z',
                              'end_utc': '2027-04-04T12:00:00Z', 'mode': 'forecast',
                              'scenarios': False}}}

    cases['check_constraints'] = {
        'api': 'app.sleep_model.evaluation.check_constraints', 'projection': PROJECTION_IDENTITY,
        'inputs': {'snapshot': _forecast_snapshot(),
                   'blocks': [deepcopy(MAIN_BLOCK), deepcopy(NAP_BLOCK)],
                   'start_utc': '2027-04-03T06:00:00Z', 'end_utc': '2027-04-04T12:00:00Z'}}

    cases['planning_candidates_order'] = {
        'api': 'app.planning.engine.plan', 'projection': PROJECTION_IDENTITY,
        'inputs': {'snapshot': planning_full, 'requests': deepcopy(PLANNING_REQUESTS)}}

    cases['available_windows'] = {
        'api': 'app.planning.engine.available_windows', 'projection': PROJECTION_IDENTITY,
        'inputs': {'snapshot': _planning_snapshot(PLANNING_POLICY_FULL)}}

    cases['check_plan_valid'] = {
        'api': 'app.planning.engine.check_plan', 'projection': PROJECTION_IDENTITY,
        'inputs': {'snapshot': _planning_snapshot(PLANNING_POLICY_FULL),
                   'blocks': [_block('sleep-ok', '2027-04-03T09:00:00Z', '2027-04-03T17:00:00Z'),
                              _block('sleep-ok2', '2027-04-04T12:00:00Z', '2027-04-04T20:00:00Z')]}}

    check_plan_invalid = {
        'check_plan_invalid_duty_buffer': (
            _planning_snapshot(),
            [_block('sleep-x', '2027-04-04T04:00:00Z', '2027-04-04T08:00:00Z')]),
        'check_plan_invalid_horizon': (
            _planning_snapshot(),
            [_block('sleep-h', '2027-04-05T08:00:00Z', '2027-04-05T16:00:00Z')]),
        'check_plan_invalid_overlap': (
            _planning_snapshot(),
            [_block('sleep-o1', '2027-04-03T09:00:00Z', '2027-04-03T17:00:00Z'),
             _block('sleep-o2', '2027-04-03T10:00:00Z', '2027-04-03T18:00:00Z')]),
        'check_plan_invalid_environment': (
            _planning_snapshot(env_end='2027-04-03T12:00:00Z'),
            [_block('sleep-e', '2027-04-04T09:00:00Z', '2027-04-04T17:00:00Z')]),
        'check_plan_invalid_minimum': (
            _planning_snapshot({'minimum_sleep_minutes': 90}),
            [_block('sleep-m', '2027-04-03T09:00:00Z', '2027-04-03T10:00:00Z')]),
        'check_plan_invalid_max_wake': (
            _planning_snapshot({'maximum_wake_hours': 20.0}),
            [_block('sleep-w', '2027-04-03T17:30:00Z', '2027-04-03T23:00:00Z')]),
    }
    for case_id, (snapshot, blocks) in check_plan_invalid.items():
        cases[case_id] = {'api': 'app.planning.engine.check_plan',
                          'projection': PROJECTION_IDENTITY,
                          'inputs': {'snapshot': snapshot, 'blocks': blocks}}

    return cases


# ---------------------------------------------------------------------------
# Fixture generation (audited, guarded; plan-review M2/M4/M6).
# ---------------------------------------------------------------------------

def _package_version(name):
    try:
        return importlib_metadata.version(name)
    except Exception:  # pragma: no cover - metadata always present in 205agent
        return 'unknown'


def _zoneinfo_probes():
    probes = {}
    for key in ('UTC', 'Asia/Tokyo'):
        try:
            ZoneInfo(key)
            probes[key] = True
        except Exception:  # pragma: no cover - tz database present in 205agent
            probes[key] = False
    return probes


def _refuse(message):
    print('REFUSING: ' + message, file=sys.stderr)
    return 2


def _rebuild(args):
    head = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True,
                          check=True, cwd=REPO_ROOT).stdout.strip()
    if head != SOURCE_COMMIT:
        return _refuse(f'HEAD {head} is not the frozen source commit {SOURCE_COMMIT}')
    dirty = subprocess.run(['git', 'status', '--porcelain', '--', 'app', 'rooster_bench',
                            'benchmarks'], capture_output=True, text=True, check=True,
                           cwd=REPO_ROOT).stdout.strip()
    if dirty:
        return _refuse('production paths have uncommitted changes:\n' + dirty)
    cases = build_cases()
    entries = {}
    for case_id in sorted(cases):
        case = cases[case_id]
        raw_first = _execute(case['api'], case['inputs'])
        raw_second = _execute(case['api'], case['inputs'])
        if raw_first != raw_second:
            return _refuse(f'case {case_id} is not deterministic across repeated in-process runs')
        gate_problems = _case_gate(case_id, raw_first)
        if gate_problems:
            return _refuse(f'case {case_id} failed the behavior gate:\n'
                           + '\n'.join(gate_problems))
        expected = _apply_projection(raw_first, case['projection'])
        consistency = _trace_digest_consistency(expected) if case['projection'] == PROJECTION_FULL else []
        if consistency:
            return _refuse(f'case {case_id} failed trace digest consistency:\n'
                           + '\n'.join(consistency))
        entries[case_id] = {'api': case['api'], 'projection': case['projection'],
                             'inputs': _to_json_safe(case['inputs']), 'expected': expected}
    fixture = {
        'schema_version': 1,
        'origin': {
            'source_commit': head,
            'base_ref': BASE_REF,
            'generation_utc': args.generation_utc
                              or datetime.now(timezone.utc).isoformat(timespec='seconds'),
            'generator_command': GENERATOR_COMMAND,
            'conda_env': '205agent',
            'python_version': platform.python_version(),
            'numpy_version': _package_version('numpy'),
            'scipy_version': _package_version('scipy'),
            'model_version': MODEL_VERSION,
            'search_version': SEARCH_VERSION,
            'policy_version': POLICY_VERSION,
            'zoneinfo_probes': _zoneinfo_probes(),
            'digest_scheme': {
                'algorithm': 'sha256',
                'serialization': "json.dumps(sort_keys=True, ensure_ascii=False, "
                                 "separators=(',', ':'), allow_nan=False)"}},
        'tolerances': {'rtol': RTOL, 'atol': ATOL},
        'trace_sampling': {'rule': TRACE_SAMPLING_RULE, 'max_points': TRACE_SAMPLE_TARGET},
        'cases': entries,
    }
    text = json.dumps(fixture, indent=2, sort_keys=True, allow_nan=False) + '\n'
    target = Path(args.output).resolve() if args.output else FIXTURE_PATH
    if target.exists() and not args.overwrite_frozen:
        return _refuse(f'{target} already exists (frozen); pass --overwrite-frozen to rewrite')
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding='utf-8')
    print(f'wrote {target} ({len(text.encode("utf-8"))} bytes, {len(entries)} cases)')
    return 0


def _verify_cli(args):
    path = Path(args.fixture).resolve() if args.fixture else FIXTURE_PATH
    fixture = json.loads(path.read_text(encoding='utf-8'))
    schema_problems = _schema_problems(fixture)
    if schema_problems:
        print(f'REFUSING: fixture at {path} fails its schema check:', file=sys.stderr)
        for problem in schema_problems:
            print('  ' + problem, file=sys.stderr)
        return 2
    failures = 0
    for case_id in sorted(fixture['cases']):
        case_problems = verify_case(case_id, fixture['cases'][case_id])
        if case_problems:
            failures += 1
            print(f'FAIL {case_id}')
            for problem in case_problems[:20]:
                print('  ' + problem)
        else:
            print(f'PASS {case_id}')
    total = len(fixture['cases'])
    print(f'{total - failures}/{total} cases passed')
    return 1 if failures else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description='Frozen decoupling baseline verify/generate tool.')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--verify', action='store_true', help='verify current implementation against the frozen fixture')
    mode.add_argument('--rebuild-fixture', action='store_true',
                       help='audited generation procedure (refuses unless HEAD is the frozen source commit)')
    parser.add_argument('--fixture', help='fixture path for --verify (default: tests/fixtures/decoupling-baseline.json)')
    parser.add_argument('--output', help='output path for --rebuild-fixture (default: the frozen fixture path)')
    parser.add_argument('--overwrite-frozen', action='store_true',
                        help='allow overwriting an existing fixture (never used after freeze)')
    parser.add_argument('--generation-utc',
                        help='pin origin.generation_utc for byte-comparable determinism runs '
                             f'(e.g. {PINNED_GENERATION_UTC})')
    args = parser.parse_args(argv)
    if args.verify:
        return _verify_cli(args)
    return _rebuild(args)


if __name__ == '__main__':
    sys.exit(main())


# ---------------------------------------------------------------------------
# Pytest entry points (primary verification mechanism, task.md §Scope).
# ---------------------------------------------------------------------------

def _load_fixture(path=None):
    return json.loads(Path(path or FIXTURE_PATH).read_text(encoding='utf-8'))


_BASELINE = _load_fixture()


@pytest.fixture(scope='module')
def baseline():
    return _BASELINE


def test_fixture_schema_and_origin(baseline):
    problems = _schema_problems(baseline)
    assert not problems, 'fixture schema problems:\n' + '\n'.join(problems)


def test_no_wall_clock_fields_in_frozen_cases(baseline):
    forbidden = {'duration', 'elapsed', 'wall_clock', 'generated_at', 'executed_at',
                 'runtime_ms', 'timing'}

    def walk(node, path):
        if isinstance(node, dict):
            for key, value in node.items():
                assert key.lower() not in forbidden, f'unexpected timing-like field {path}.{key}'
                walk(value, f'{path}.{key}')
        elif isinstance(node, list):
            for index, item in enumerate(node):
                walk(item, f'{path}[{index}]')

    for case_id, case in baseline['cases'].items():
        walk(case, case_id)


@pytest.mark.parametrize('case_id', sorted(_BASELINE['cases']))
def test_frozen_case(case_id):
    problems = verify_case(case_id, _BASELINE['cases'][case_id])
    assert not problems, (f'frozen baseline mismatch for {case_id}:\n'
                          + '\n'.join(problems[:40]))
