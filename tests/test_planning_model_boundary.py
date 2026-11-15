"""Boundary tests for the planning/model decoupling (module decoupling contract step 3).

Covered: the lazy legacy compatibility bridges, the policy-vs-physiology split
with the legacy configuration error precedence, the frozen hash/provenance
preservation, the explicit sleep-count compensation ablation (evaluator and
search) and the import boundaries between planning, the model and the
evaluation layer.

Legacy expectations come from the committed frozen fixture
(``tests/fixtures/decoupling-baseline.json``); no new expectation is recorded
for a legacy output.
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import timedelta
from pathlib import Path

import pytest

from app.contracts.time import dt, stamp
from app.planning import adapter, evaluation, preferences, sleep_count
from app.planning.engine import available_windows, plan
from app.planning.evaluation import evaluate
from app.planning.policy import (POLICY_DEFAULTS, POLICY_KEYS, planning_configuration,
                                  policy_configuration, public_planning_configuration)
from app.planning.search import generate
from app.sleep_model.contracts import (DEFAULTS, PHYSIOLOGICAL_DEFAULTS, VERSION,
                                        compatibility_configuration, configuration, digest,
                                        physiological_configuration)

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / 'tests' / 'fixtures' / 'decoupling-baseline.json'

#: Model modules that are private implementation detail: planning must not
#: import them (the public API and the shared time contract are the boundary).
FORBIDDEN_MODEL_MODULES = ('pipeline', 'simulation', 'homeostat', 'metrics', 'initialization',
                            'circadian', 'light', 'sleep_forecast', 'archive', 'sources')
#: The model files that are allowed to route legacy names to planning.
LEGACY_SHIM_FILES = frozenset({'__init__.py', 'evaluation.py', 'adapter.py',
                               'preferences.py', 'sleep_count.py'})
#: Pre-existing production import of the evaluation layer, out of this task's
#: scope (the offline LLM stage archive in app/planning/agent.py).
ROOSTER_BENCH_EXCLUSIONS = ()


def _fixture() -> dict:
    return json.loads(FIXTURE.read_text(encoding='utf-8'))


def _forecast_case() -> dict:
    return _fixture()['cases']['evaluate_forecast_scenarios']


def _planning_case() -> dict:
    return _fixture()['cases']['planning_candidates_order']


def _count_case():
    """A snapshot whose candidate opportunities create more than four episodes."""
    now = dt('2027-01-02T00:00:00Z')
    end = now + timedelta(hours=24)
    snapshot = {'snapshot_id': 'boundary-count', 'revision': 1,
                'now_utc': stamp(now), 'horizon_end_utc': stamp(end),
                'profile': {'home_timezone': 'UTC', 'sleep_need_hours': 8},
                'initial_state': {},
                'events': [{'id': 'env-home', 'kind': 'environment',
                            'start_utc': stamp(now - timedelta(hours=1)),
                            'end_utc': stamp(end), 'timezone': 'UTC', 'location': 'home',
                            'availability': 'available', 'confirmed': True}]}
    blocks = [{'id': f'nap-{hour}', 'start_utc': stamp(now + timedelta(hours=hour)),
               'end_utc': stamp(now + timedelta(hours=hour + 1)), 'timezone': 'UTC',
               'role': 'nap'} for hour in range(0, 16, 2)]
    return snapshot, blocks


# ---------------------------------------------------------------------------
# 1. Lazy legacy compatibility bridges
# ---------------------------------------------------------------------------

def test_legacy_import_paths_resolve_to_the_planning_implementations():
    import app.sleep_model
    import app.sleep_model.adapter as legacy_adapter
    import app.sleep_model.evaluation as legacy_evaluation
    import app.sleep_model.preferences as legacy_preferences
    import app.sleep_model.sleep_count as legacy_sleep_count

    assert app.sleep_model.evaluate is evaluation.evaluate
    assert app.sleep_model.evaluate_many is evaluation.evaluate_many
    assert app.sleep_model.VERSION == VERSION
    assert legacy_evaluation.evaluate is evaluation.evaluate
    assert legacy_evaluation.evaluate_many is evaluation.evaluate_many
    assert legacy_evaluation.check_constraints is evaluation.check_constraints
    assert legacy_adapter.score is adapter.score
    assert legacy_preferences.evaluate_preferences is preferences.evaluate_preferences
    assert legacy_preferences.sleep_targets is preferences.sleep_targets
    assert legacy_sleep_count.sleep_count_cost is sleep_count.sleep_count_cost


def test_importing_the_model_package_never_imports_planning():
    code = ("import sys; import app.sleep_model, app.sleep_model.api; "
            "bad=[m for m in sys.modules if m.startswith(('app.planning', 'rooster_bench'))]; "
            "sys.exit(1 if bad else 0)")
    completed = subprocess.run([sys.executable, '-c', code], cwd=str(ROOT),
                               capture_output=True, text=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr


# ---------------------------------------------------------------------------
# 2. Dependency boundaries
# ---------------------------------------------------------------------------

def test_planning_imports_no_model_private_internal_and_no_evaluation_layer():
    for path in sorted((ROOT / 'app' / 'planning').glob('*.py')):
        text = path.read_text(encoding='utf-8')
        relative = path.relative_to(ROOT).as_posix()
        for module in FORBIDDEN_MODEL_MODULES:
            assert f'app.sleep_model.{module}' not in text, f'{module} imported in {relative}'
        assert 'rooster_bench' not in text or relative in ROOSTER_BENCH_EXCLUSIONS, \
            f'rooster_bench imported in {relative}'
        # the only model surface planning may use is the public API, the
        # documented configuration surface and the shared time contract
        for module in ('app.sleep_model.pipeline', 'app.sleep_model.simulation'):
            assert module not in text


def test_the_model_graph_imports_planning_only_through_the_documented_shims():
    for path in sorted((ROOT / 'app' / 'sleep_model').glob('*.py')):
        if path.name in LEGACY_SHIM_FILES:
            continue
        text = path.read_text(encoding='utf-8')
        assert 'app.planning' not in text, f'app.planning imported in {path.name}'
        assert 'rooster_bench' not in text, f'rooster_bench imported in {path.name}'


def test_only_the_documented_exclusion_imports_the_evaluation_layer():
    offenders = [path.relative_to(ROOT).as_posix() for path in sorted((ROOT / 'app').rglob('*.py'))
                 if 'rooster_bench' in path.read_text(encoding='utf-8')]
    assert offenders == list(ROOSTER_BENCH_EXCLUSIONS)


# ---------------------------------------------------------------------------
# 3. Policy vs physiology
# ---------------------------------------------------------------------------

def test_planning_policy_defaults_equal_the_model_legacy_policy_defaults():
    for key in POLICY_KEYS:
        assert POLICY_DEFAULTS[key] == DEFAULTS[key], key
        assert type(POLICY_DEFAULTS[key]) is type(DEFAULTS[key]), key
    assert policy_configuration() == {key: DEFAULTS[key] for key in POLICY_KEYS}


def test_the_default_full_configuration_is_byte_identical_to_the_legacy_one():
    physio, policy, full = planning_configuration()
    assert full == configuration()
    assert full == compatibility_configuration(physiological_configuration())
    assert physio == {key: value for key, value in full.items() if key in PHYSIOLOGICAL_DEFAULTS}
    assert policy == {key: value for key, value in full.items() if key in POLICY_KEYS}
    assert digest(full) == digest(configuration())


def test_policy_keys_are_rejected_by_the_public_physiological_resolver():
    for key in POLICY_KEYS:
        with pytest.raises(ValueError, match='unknown physiological parameters'):
            physiological_configuration({key: 1})


def test_the_new_planning_api_resolves_physiology_through_the_public_resolver():
    physio, policy, full = public_planning_configuration()
    assert physio == physiological_configuration()
    assert policy == {key: DEFAULTS[key] for key in POLICY_KEYS}
    assert full == {**physio, **policy}
    with pytest.raises(ValueError, match='unknown planning parameters'):
        public_planning_configuration({'bogus_parameter': 1})


# ---------------------------------------------------------------------------
# 4. Legacy configuration error precedence (Lead directive, review finding M10)
# ---------------------------------------------------------------------------

LEGACY_PRECEDENCE_CASES = [
    ({'bogus_parameter': 1, 'tau_w': 0.0}, 'unknown model parameters: bogus_parameter'),
    ({'tau_w': 0.0, 'sleep_count_coefficient': -1.0}, 'invalid positive parameter: tau_w'),
    ({'light_factor': -1.0, 'sleep_count_coefficient': -1.0}, 'negative input assumption'),
    ({'sleep_count_coefficient': -1.0, 'burden_weights': [0.5, 0.25, 0.3]},
     'invalid sleep count parameters'),
    ({'sleep_count_coefficient': -1.0, 'rank_weights': [0.5, 0.5, 0.0, 0.1]},
     'invalid sleep count parameters'),
    ({'sleep_count_coefficient': -1.0}, 'invalid sleep count parameters'),
    ({'rank_weights': [0.5, 0.5, 0.0, 0.1]}, 'weights must be nonnegative and sum to one'),
    ({'burden_weights': [0.5, 0.25, 0.3]}, 'weights must be nonnegative and sum to one'),
]


@pytest.mark.parametrize('overrides,message', LEGACY_PRECEDENCE_CASES)
def test_the_legacy_configuration_error_precedence_is_preserved(overrides, message):
    with pytest.raises(ValueError) as legacy_error:
        configuration(overrides)
    assert str(legacy_error.value) == message
    with pytest.raises(ValueError) as planning_error:
        planning_configuration(overrides)
    assert str(planning_error.value) == message


def test_the_planning_legacy_path_never_surfaces_the_physiological_unknown_key_message():
    with pytest.raises(ValueError) as error:
        planning_configuration({'bogus_parameter': 1})
    assert str(error.value) == 'unknown model parameters: bogus_parameter'
    assert 'unknown physiological parameters' not in str(error.value)
    # a legitimate policy key is accepted by the planning legacy path
    _, policy, _ = planning_configuration({'rank_weights': [.25, .25, .25, .25]})
    assert policy['rank_weights'] == [.25, .25, .25, .25]


def test_the_planning_policy_resolver_never_raises_a_physiological_message():
    for overrides in ({'bogus_parameter': 1}, {'sleep_count_coefficient': -1.0},
                      {'rank_weights': [0.5, 0.5, 0.0, 0.1]}):
        with pytest.raises(ValueError) as error:
            policy_configuration(overrides)
        assert 'invalid positive parameter' not in str(error.value)
        assert 'negative input assumption' not in str(error.value)
        assert 'unknown physiological parameters' not in str(error.value)


# ---------------------------------------------------------------------------
# 5. Frozen hashes, provenance and assumptions
# ---------------------------------------------------------------------------

def test_the_planning_evaluation_matches_the_frozen_fixture():
    case = _forecast_case()
    result = evaluate(case['inputs']['snapshot'], case['inputs']['sleep_blocks'],
                      **case['inputs']['kwargs'])
    expected = case['expected']
    assert set(result['provenance']) == set(expected['provenance'])
    for key in ('configuration_hash', 'input_hash', 'snapshot_hash', 'opportunity_hash', 'trace_hash'):
        assert result['provenance'][key] == expected['provenance'][key], key
    assert result['provenance']['configuration'] == expected['provenance']['configuration']
    assert result['provenance']['initialization'] == expected['provenance']['initialization']
    assert result['initial_state']['input_prefix_hash'] == expected['initial_state']['input_prefix_hash']
    assert result['initial_state']['initialization_id'] == expected['initial_state']['initialization_id']
    assert result['initial_state']['configuration_hash'] == expected['initial_state']['configuration_hash']
    assert result['assumptions'] == expected['assumptions']
    assert result['losses'] == expected['losses']
    assert result['utility'] == expected['utility']


def test_the_frozen_search_invariants_are_preserved():
    case = _planning_case()
    result = plan(case['inputs']['snapshot'], case['inputs'].get('requests'))
    expected = case['expected']
    assert result['search']['expanded'] == expected['search']['expanded']
    assert result['search']['evaluated_candidates'] == expected['search']['evaluated_candidates']
    assert result['search']['feasible_candidates'] == expected['search']['feasible_candidates']
    assert result['search']['ranking'] == expected['search']['ranking']
    assert result['search']['version'] == expected['search']['version']
    assert [item['id'] for item in result['candidates']] == [item['id'] for item in expected['candidates']]


def test_the_public_assumption_strings_are_pinned_and_the_legacy_trio_is_used(monkeypatch):
    import app.sleep_model.pipeline as pipeline
    from app.planning.evaluation import LEGACY_ASSUMPTIONS, PUBLIC_ASSUMPTIONS
    from app.sleep_model import api

    case = _forecast_case()
    public = api.compute(case['inputs']['snapshot'], case['inputs']['sleep_blocks'],
                          start_utc=case['inputs']['kwargs']['start_utc'],
                          end_utc=case['inputs']['kwargs']['end_utc'])
    for text in PUBLIC_ASSUMPTIONS:
        assert text in public['assumptions']

    original = pipeline.build_state

    def non_converged(snapshot, at, config, checkpoint=None):
        state, info = original(snapshot, at, config, checkpoint=checkpoint)
        info = dict(info)
        info['prior_converged'] = False
        return state, info

    monkeypatch.setattr(pipeline, 'build_state', non_converged)
    result = evaluate(case['inputs']['snapshot'], case['inputs']['sleep_blocks'],
                      start_utc=case['inputs']['kwargs']['start_utc'],
                      end_utc=case['inputs']['kwargs']['end_utc'], mode='forecast', scenarios=False)
    assert result['status'] == 'partial'
    assert result['utility'] is None
    assert result['assumptions'][-4] == 'prior_not_converged'
    assert result['assumptions'][-3:] == list(LEGACY_ASSUMPTIONS)
    assert 'latency and inertia are engineering assumptions; outputs are not fitness-for-duty judgments' \
        not in result['assumptions']


def test_the_policy_override_path_keeps_the_legacy_configuration_hash_and_the_documented_identity():
    case = _forecast_case()
    kwargs = dict(case['inputs']['kwargs'])
    kwargs['scenarios'] = False
    default = evaluate(case['inputs']['snapshot'], case['inputs']['sleep_blocks'], **kwargs)
    override = evaluate(case['inputs']['snapshot'], case['inputs']['sleep_blocks'],
                        config={'rank_weights': [.5, .3, .1, .1]}, **kwargs)
    assert override['provenance']['configuration_hash'] != default['provenance']['configuration_hash']
    assert override['provenance']['trace_hash'] == default['provenance']['trace_hash']
    # Documented boundary of the compatibility bridge: the public model API
    # builds the checkpoint identity from the compatibility configuration (the
    # current policy defaults), so a policy override changes the reported
    # configuration but not the model's initialization identity.
    assert override['initial_state']['input_prefix_hash'] == default['initial_state']['input_prefix_hash']


# ---------------------------------------------------------------------------
# 6. Sleep-count compensation ablation
# ---------------------------------------------------------------------------

def test_the_sleep_count_compensation_is_enabled_by_default_and_reported_separately():
    snapshot, blocks = _count_case()
    result = evaluate(snapshot, blocks, scenarios=False)
    count = result['physiological_metrics']['sleep_count_cost']
    assert result['losses']['sleep_count'] == count['points'] / 100
    assert result['losses']['sleep_count'] > 0
    assert count['threshold'] == 4
    assert count['coefficient'] == 2.0


def test_the_sleep_count_switch_removes_only_the_count_contribution():
    snapshot, blocks = _count_case()
    default = evaluate(snapshot, blocks, scenarios=False)
    ablated = evaluate(snapshot, blocks, scenarios=False, sleep_count=False)
    routed = evaluate(snapshot, blocks, scenarios=False, config={'sleep_count_coefficient': 0})

    assert ablated['losses']['sleep_count'] == 0.0
    assert ablated['physiological_metrics']['sleep_count_cost']['points'] == 0.0
    assert ablated['utility'] == routed['utility']
    assert ablated['losses'] == routed['losses']
    assert ablated['trace'] == default['trace']
    assert ablated['physiological_metrics']['sleep_target_buckets'] == \
        default['physiological_metrics']['sleep_target_buckets']
    # the switch is an evaluation-time choice, not a configuration change
    assert ablated['provenance']['configuration'] == default['provenance']['configuration']
    assert ablated['provenance']['configuration_hash'] == default['provenance']['configuration_hash']
    assert ablated['initial_state']['input_prefix_hash'] == default['initial_state']['input_prefix_hash']
    # the configuration route differs by design: it changes the configuration
    assert routed['provenance']['configuration_hash'] != default['provenance']['configuration_hash']


def test_the_sleep_count_switch_reaches_the_search_objective():
    snapshot, blocks = _count_case()
    windows = available_windows(snapshot)
    default, default_info = generate(snapshot, windows, .5, reference_trace=())
    ablated, ablated_info = generate(snapshot, windows, .5, reference_trace=(), sleep_count=False)
    # below the episode threshold the count term is zero anyway: the ablated
    # search must be identical, which proves nothing else was touched
    assert default == ablated
    assert default_info == ablated_info


def _count_divergence_case():
    """A snapshot whose rolling-episode history shadows the first window but
    not the second, so the count term and the remaining objective terms prefer
    different windows."""
    now = dt('2027-01-02T00:00:00Z')
    horizon = now + timedelta(hours=26)
    events = [{'id': 'env-a', 'kind': 'environment', 'start_utc': stamp(now),
               'end_utc': stamp(now + timedelta(hours=2)), 'timezone': 'UTC', 'location': 'home',
               'availability': 'available', 'confirmed': True},
              {'id': 'env-b', 'kind': 'environment', 'start_utc': stamp(now + timedelta(hours=25)),
               'end_utc': stamp(horizon), 'timezone': 'UTC', 'location': 'home',
               'availability': 'available', 'confirmed': True},
              {'id': 'occupied-gap', 'kind': 'environment',
               'start_utc': stamp(now + timedelta(hours=2)),
               'end_utc': stamp(now + timedelta(hours=25)), 'timezone': 'UTC',
               'availability': 'unavailable', 'confirmed': True}]
    for index in range(10):
        start = now - timedelta(hours=2) + timedelta(minutes=12 * index)
        events.append({'id': f'history-{index}', 'kind': 'sleep_actual',
                       'source': 'synthetic_observation', 'start_utc': stamp(start),
                       'end_utc': stamp(start + timedelta(minutes=10)), 'timezone': 'UTC'})
    return {'snapshot_id': 'boundary-count-divergence', 'revision': 1,
            'now_utc': stamp(now), 'horizon_end_utc': stamp(horizon),
            'profile': {'home_timezone': 'UTC', 'sleep_need_hours': 8},
            'initial_state': {}, 'events': events}


def test_the_sleep_count_term_participates_in_the_search_objective():
    """With more than four rolling episodes shadowing the first window, the
    count term is large enough to change the beam's choice, while the
    below-threshold case above proves that nothing else changed."""
    snapshot = _count_divergence_case()
    windows = available_windows(snapshot)
    assert len(windows) == 2
    default, default_info = generate(snapshot, windows, .5, reference_trace=())
    ablated, ablated_info = generate(snapshot, windows, .5, reference_trace=(), sleep_count=False)
    assert default_info['status'] == 'complete'
    assert ablated_info['status'] == 'complete'
    assert default != ablated


def test_the_planner_reports_the_ablated_compensation():
    case = _planning_case()
    snapshot = case['inputs']['snapshot']
    default = plan(snapshot, case['inputs'].get('requests'))
    ablated = plan(snapshot, case['inputs'].get('requests'), sleep_count=False)
    assert default['search']['ranking'] != ablated['search']['ranking']
    for candidate in ablated['candidates']:
        assert candidate['metrics']['losses']['sleep_count'] == 0.0
    # the default path keeps the legacy ranking label
    assert default['search']['ranking'] == \
        'sleep model utility including rolling sleep-count cost; external heuristics excluded'
