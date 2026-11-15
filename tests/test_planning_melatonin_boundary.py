"""Boundary tests for the explicit opt-in melatonin diagnostic.

The diagnostic lives behind its own explicit selection: the absent path must
leave the legacy dictionaries, scores, ordering and hashes exactly as they are
(the frozen fixture is the reference), and the enabled path must call the
public model API with explicitly requested, real candidate intervals and
propagate the model's plasma/phase/synthesis summaries verbatim, including
their units, validity and provenance.

No physiological policy, threshold, dosing logic or planning-quality claim is
introduced, and the diagnostic never influences the planning output.
"""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from app.planning import melatonin_diagnostics
from app.planning.engine import plan
from app.planning.evaluation import evaluate
from app.sleep_model import api

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / 'tests' / 'fixtures' / 'decoupling-baseline.json'

MELATONIN_COMPONENTS = ['melatonin']


def _fixture() -> dict:
    return json.loads(FIXTURE.read_text(encoding='utf-8'))


def _planning_case() -> dict:
    return _fixture()['cases']['planning_candidates_order']


def _forecast_case() -> dict:
    return _fixture()['cases']['evaluate_forecast_scenarios']


def _candidate_intervals():
    """The candidate intervals of a real plan run, with their block ids."""
    case = _planning_case()
    result = plan(case['inputs']['snapshot'], case['inputs'].get('requests'))
    candidate = result['candidates'][0]
    intervals = [{'id': block['id'], 'start_utc': block['start_utc'], 'end_utc': block['end_utc']}
                 for block in candidate['sleep_blocks']]
    return case['inputs']['snapshot'], candidate['sleep_blocks'], intervals


def _model_block(ids=('candidate-001',)):
    """A D2-shaped melatonin block: per-summary plasma/synthesis/crossings plus
    the additive trajectory-wide crossings list, units, validity, provenance."""
    return {
        'summaries': [{
            'id': identifier,
            'plasma_pM': {'min': 0.0, 'mean': 12.5, 'max': 40.0},
            'synthesis_active_hours': 6.5,
            'crossings': [{'kind': 'SynOn', 'at_utc': '2027-04-03T21:14:00Z', 'direction': 'rising',
                           'phase_rad': -2.1, 'validity': {'status': 'complete'}}],
            'units': {'plasma_pM': 'pmol/L', 'synthesis_active_hours': 'hours',
                      'time_utc': 'UTC ISO-8601 instant'},
            'validity': {'status': 'complete', 'numerical_failures': []},
            'provenance': {'melatonin_model_version': 'breslow13-endogenous-v1',
                           'melatonin_model_config_hash': 'sha256:melatonin',
                           'source_pin': 'db6eb08b6c0b0586cdac7795bf10bbbbfd323c1a',
                           'initialization': 'model default, not a personal estimate'},
        } for identifier in ids],
        'crossings': [{'kind': 'SynOn', 'at_utc': '2027-04-03T21:14:00Z', 'direction': 'rising',
                       'phase_rad': -2.1, 'validity': {'status': 'complete'}}],
        'units': {'H1': 'pmol/L', 'H2': 'pmol/L', 'synthesis': 'pmol/hour', 'phase_rad': 'radians',
                  'light_lux': 'photopic lux', 'time_utc': 'UTC ISO-8601 instant'},
        'validity': {'status': 'complete', 'numerical_failures': [],
                     'suppression_guard_activations': 0},
        'provenance': {'melatonin_model_version': 'breslow13-endogenous-v1',
                       'melatonin_api_version': 'rooster-sleep-model-api-v1',
                       'melatonin_model_config_hash': 'sha256:melatonin',
                       'source_pin': 'db6eb08b6c0b0586cdac7795bf10bbbbfd323c1a',
                       'initialization': 'model default, not a personal estimate'},
    }


def _stub_model(monkeypatch, block=None, error=None, captured=None):
    """Replace the public model call and record what planning asked for.

    Calls that do not select the melatonin component are delegated to the real
    public model, so the planning output stays comparable inside the test.
    """
    original = api.compute

    def fake_compute(snapshot, sleep_blocks=None, **kwargs):
        selected = 'melatonin' in (kwargs.get('components') or [])
        if captured is not None:
            captured.update(kwargs)
            captured['snapshot'] = snapshot
            captured['sleep_blocks'] = sleep_blocks
        if not selected:
            return original(snapshot, sleep_blocks, **kwargs)
        if error is not None:
            raise error
        requested = kwargs.get(melatonin_diagnostics.SUMMARIES_KEYWORD) or []
        return {'api_version': 'rooster-sleep-model-api-v1',
                'model_version': 'rooster-sleep-core-v2.1.0',
                'mode': kwargs.get('mode', 'forecast'), 'status': 'complete',
                'knowledge_cutoff_utc': '2027-04-03T06:00:00Z',
                'evaluation_start_utc': '2027-04-03T06:00:00Z',
                'evaluation_end_utc': '2027-04-04T12:00:00Z',
                'melatonin': block if block is not None
                else _model_block([item['id'] for item in requested])}

    monkeypatch.setattr(api, 'compute', fake_compute)


# ---------------------------------------------------------------------------
# 1. The absent path keeps the legacy output exactly
# ---------------------------------------------------------------------------

def test_the_absent_diagnostic_leaves_the_legacy_evaluation_untouched():
    case = _forecast_case()
    result = evaluate(case['inputs']['snapshot'], case['inputs']['sleep_blocks'],
                      **case['inputs']['kwargs'])
    expected = case['expected']
    assert not any('melatonin' in key for key in result)
    assert not any('melatonin' in key for key in result['provenance'])
    assert result['provenance'] == expected['provenance']
    assert result['losses'] == expected['losses']
    assert result['assumptions'] == expected['assumptions']
    assert result['initial_state'] == expected['initial_state']
    assert result['utility'] == expected['utility']


def test_the_absent_diagnostic_leaves_the_plan_output_untouched():
    case = _planning_case()
    result = plan(case['inputs']['snapshot'], case['inputs'].get('requests'))
    expected = case['expected']
    assert not any('melatonin' in key for key in result)
    assert not any('melatonin' in key for key in result['search'])
    assert [item['id'] for item in result['candidates']] == [item['id'] for item in expected['candidates']]
    assert result['search']['expanded'] == expected['search']['expanded']


def test_the_diagnostic_never_changes_the_planning_output(monkeypatch):
    _stub_model(monkeypatch)
    case = _planning_case()
    before = plan(case['inputs']['snapshot'], case['inputs'].get('requests'))
    snapshot, blocks, intervals = _candidate_intervals()
    melatonin_diagnostics.run(snapshot, sleep_blocks=blocks, melatonin_summaries=intervals)
    after = plan(case['inputs']['snapshot'], case['inputs'].get('requests'))
    assert before == after


# ---------------------------------------------------------------------------
# 2. The enabled path calls the public model API on real candidate intervals
# ---------------------------------------------------------------------------

def test_the_enabled_diagnostic_propagates_the_model_block_verbatim(monkeypatch):
    captured = {}
    _stub_model(monkeypatch, captured=captured)
    snapshot, blocks, intervals = _candidate_intervals()
    result = melatonin_diagnostics.run(snapshot, sleep_blocks=blocks, melatonin_summaries=intervals,
                                       melatonin_crossings=True)

    assert captured['components'] == MELATONIN_COMPONENTS
    assert captured[melatonin_diagnostics.SUMMARIES_KEYWORD] == intervals
    assert captured[melatonin_diagnostics.CROSSINGS_KEYWORD] is True
    # only the frozen request kwargs are sent: no model parameter overrides,
    # no grid, no light samples and no checkpoint
    assert set(captured) - {'snapshot', 'sleep_blocks'} == {
        'components', melatonin_diagnostics.SUMMARIES_KEYWORD,
        melatonin_diagnostics.CROSSINGS_KEYWORD, 'start_utc', 'end_utc', 'mode', 'config'}
    # the ids are echoed back verbatim
    assert [item['id'] for item in captured[melatonin_diagnostics.SUMMARIES_KEYWORD]] == \
        [item['id'] for item in intervals]
    assert result['selection'][melatonin_diagnostics.SUMMARIES_KEYWORD] == intervals
    # the model block is propagated verbatim, crossings included, with units,
    # validity and provenance
    assert result['melatonin'] == _model_block([item['id'] for item in intervals])
    summary = result['melatonin']['summaries'][0]
    assert set(summary['plasma_pM']) == {'min', 'mean', 'max'}
    assert 'synthesis_active_hours' in summary
    assert 'crossings' in summary
    assert summary['units'] and summary['validity'] and summary['provenance']
    assert result['melatonin']['units'] and result['melatonin']['validity']
    assert result['melatonin']['provenance']['melatonin_model_config_hash']
    assert result['crossings_requested'] is True
    assert result['trajectory_crossings'] == result['melatonin']['crossings']
    assert result['crossings_unavailable'] == []


def test_the_crossings_opt_in_is_forwarded_verbatim(monkeypatch):
    captured = {}
    _stub_model(monkeypatch, captured=captured)
    snapshot, blocks, intervals = _candidate_intervals()
    melatonin_diagnostics.run(snapshot, sleep_blocks=blocks, melatonin_summaries=intervals,
                              melatonin_crossings=True)
    assert captured[melatonin_diagnostics.CROSSINGS_KEYWORD] is True


def test_the_disabled_opt_in_presents_no_crossings_field(monkeypatch):
    """With a D2-shaped response the disabled opt-in must not present the
    model's placeholder crossings anywhere: the key is omitted from the
    propagated summaries, the trajectory-wide list is not presented, and the
    envelope states crossings_requested: false explicitly."""
    captured = {}
    _stub_model(monkeypatch, captured=captured)
    snapshot, blocks, intervals = _candidate_intervals()
    result = melatonin_diagnostics.run(snapshot, sleep_blocks=blocks, melatonin_summaries=intervals)

    assert captured[melatonin_diagnostics.CROSSINGS_KEYWORD] is False
    assert result['crossings_requested'] is False
    assert 'trajectory_crossings' not in result
    assert 'crossings_unavailable' not in result
    for summary in result['melatonin']['summaries']:
        assert 'crossings' not in summary
    # the non-crossings content of the model block is untouched
    expected = _model_block([item['id'] for item in intervals])
    for summary, model_summary in zip(result['melatonin']['summaries'], expected['summaries']):
        assert summary == {key: value for key, value in model_summary.items() if key != 'crossings'}
    assert 'crossings' not in result['melatonin']


def test_the_diagnostic_config_never_surfaces_the_physiological_message(monkeypatch):
    """R2: the diagnostic's config is the planning-owned surface; a policy key
    is resolved in planning and only the physiological subset reaches the
    model, so the model's 'unknown physiological parameters' message can never
    surface from this API."""
    captured = {}
    _stub_model(monkeypatch, captured=captured)
    snapshot, blocks, intervals = _candidate_intervals()
    result = melatonin_diagnostics.run(snapshot, sleep_blocks=blocks, melatonin_summaries=intervals,
                                       config={'rank_weights': [.25, .25, .25, .25]})
    # the policy key was resolved in planning and not forwarded to the model
    assert 'rank_weights' not in (captured['config'] or {})
    assert result['crossings_requested'] is False
    # a genuinely unknown key raises the planning-owned message
    with pytest.raises(ValueError) as error:
        melatonin_diagnostics.run(snapshot, sleep_blocks=blocks, melatonin_summaries=intervals,
                                  config={'bogus_parameter': 1})
    assert str(error.value) == 'unknown planning parameters: bogus_parameter'
    # an invalid policy value raises the planning-owned policy message
    with pytest.raises(ValueError) as error:
        melatonin_diagnostics.run(snapshot, sleep_blocks=blocks, melatonin_summaries=intervals,
                                  config={'rank_weights': [0.5, 0.5, 0.0, 0.1]})
    assert 'unknown physiological parameters' not in str(error.value)


def test_missing_per_summary_crossings_are_reported_honestly(monkeypatch):
    """A model response without per-summary crossings must be reported, not
    turned into an empty list that would read as 'no crossings occurred'."""
    snapshot, blocks, intervals = _candidate_intervals()
    block = _model_block([item['id'] for item in intervals])
    for summary in block['summaries']:
        summary.pop('crossings')
    _stub_model(monkeypatch, block=block)
    result = melatonin_diagnostics.run(snapshot, sleep_blocks=blocks, melatonin_summaries=intervals,
                                       melatonin_crossings=True)
    assert result['melatonin'] == block
    assert result['crossings_requested'] is True
    assert result['crossings_unavailable'] == [item['id'] for item in intervals]
    assert any('did not carry per-summary crossings' in note for note in result['notes'])
    # planning never fills the gap from the additive trajectory-wide list
    assert 'trajectory_crossings' not in result or \
        result['trajectory_crossings'] == block.get('crossings')


def test_an_empty_crossings_list_is_a_model_statement_and_is_propagated_verbatim(monkeypatch):
    """Under the opt-in the model always provides the key: an empty list means
    'computed, none inside this interval' and must not be flagged."""
    snapshot, blocks, intervals = _candidate_intervals()
    block = _model_block([item['id'] for item in intervals])
    for summary in block['summaries']:
        summary['crossings'] = []
    _stub_model(monkeypatch, block=block)
    result = melatonin_diagnostics.run(snapshot, sleep_blocks=blocks, melatonin_summaries=intervals,
                                       melatonin_crossings=True)
    assert result['melatonin']['summaries'][0]['crossings'] == []
    assert result['crossings_requested'] is True
    assert result['crossings_unavailable'] == []


def test_the_disabled_opt_in_omits_the_placeholder_key(monkeypatch):
    """Defensive variant of the disabled case: even when the model omits the
    key entirely, the disabled opt-in presents no crossings field."""
    snapshot, blocks, intervals = _candidate_intervals()
    block = _model_block([item['id'] for item in intervals])
    for summary in block['summaries']:
        summary.pop('crossings')
    _stub_model(monkeypatch, block=block)
    result = melatonin_diagnostics.run(snapshot, sleep_blocks=blocks, melatonin_summaries=intervals)
    assert 'trajectory_crossings' not in result
    assert 'crossings_unavailable' not in result
    for summary in result['melatonin']['summaries']:
        assert 'crossings' not in summary


def test_an_explicit_selection_is_required():
    snapshot, blocks, intervals = _candidate_intervals()
    with pytest.raises(ValueError, match='explicit melatonin_summaries selection required'):
        melatonin_diagnostics.run(snapshot, sleep_blocks=blocks)
    with pytest.raises(ValueError, match='explicit melatonin_summaries selection required'):
        melatonin_diagnostics.run(snapshot, sleep_blocks=blocks, melatonin_summaries=[])
    with pytest.raises(ValueError, match='missing: end_utc'):
        melatonin_diagnostics.run(snapshot, sleep_blocks=blocks,
                                  melatonin_summaries=[{'id': 'a', 'start_utc': intervals[0]['start_utc']}])


# ---------------------------------------------------------------------------
# 3. Failures are propagated, validity markers are not reinterpreted
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('error', [ValueError('unknown melatonin parameter keys'),
                                   TypeError('melatonin_summaries must be a list'),
                                   ArithmeticError('solver integration failed')])
def test_model_failures_reach_the_caller_unchanged(monkeypatch, error):
    _stub_model(monkeypatch, error=error)
    snapshot, blocks, intervals = _candidate_intervals()
    with pytest.raises(type(error)) as raised:
        melatonin_diagnostics.run(snapshot, sleep_blocks=blocks, melatonin_summaries=intervals)
    assert str(raised.value) == str(error)


def test_validity_markers_are_propagated_without_clamping(monkeypatch):
    snapshot, blocks, intervals = _candidate_intervals()
    block = _model_block([item['id'] for item in intervals])
    block['validity'] = {'status': 'numerical_validity_failure',
                         'numerical_failures': [{'grid_utc': '2027-04-03T22:00:00Z', 'H2': -0.5}]}
    for summary in block['summaries']:
        summary['validity'] = {'status': 'numerical_validity_failure',
                               'numerical_failures': [{'grid_utc': '2027-04-03T22:00:00Z', 'H2': -0.5}]}
        summary['plasma_pM']['min'] = -0.5
    _stub_model(monkeypatch, block=block)
    result = melatonin_diagnostics.run(snapshot, sleep_blocks=blocks, melatonin_summaries=intervals,
                                       melatonin_crossings=True)
    assert result['melatonin'] == block
    assert result['crossings_requested'] is True
    assert result['melatonin']['summaries'][0]['plasma_pM']['min'] == -0.5
    assert result['melatonin']['validity']['status'] == 'numerical_validity_failure'
    assert result['melatonin']['validity']['numerical_failures'][0]['H2'] == -0.5


def test_a_missing_model_block_is_an_explicit_failure(monkeypatch):
    original = api.compute

    def fake_compute(snapshot, sleep_blocks=None, **kwargs):
        if 'melatonin' not in (kwargs.get('components') or []):
            return original(snapshot, sleep_blocks, **kwargs)
        return {'api_version': 'rooster-sleep-model-api-v1', 'model_version': 'rooster-sleep-core-v2.1.0'}

    monkeypatch.setattr(api, 'compute', fake_compute)
    snapshot, blocks, intervals = _candidate_intervals()
    with pytest.raises(RuntimeError, match='no melatonin diagnostics'):
        melatonin_diagnostics.run(snapshot, sleep_blocks=blocks, melatonin_summaries=intervals)


# ---------------------------------------------------------------------------
# 4. Real public-API probe (before and after the joint integration)
# ---------------------------------------------------------------------------

def test_the_real_public_api_probe_never_fabricates_a_summary():
    """Shape-tolerant, reporting probe: it must work before and after the joint
    integration and must not pass against a fabricated local summary."""
    snapshot, blocks, intervals = _candidate_intervals()
    try:
        api.resolve_components(list(MELATONIN_COMPONENTS))
    except ValueError:
        # the model does not know the component yet: the diagnostic must fail
        # loudly instead of inventing a summary
        with pytest.raises((TypeError, ValueError)):
            melatonin_diagnostics.run(snapshot, sleep_blocks=blocks, melatonin_summaries=intervals)
        return
    result = melatonin_diagnostics.run(snapshot, sleep_blocks=blocks, melatonin_summaries=intervals,
                                      melatonin_crossings=True)
    # the model's own block is propagated verbatim
    assert result['melatonin'] is not None
    summaries = {item.get('id'): item for item in (result['melatonin'].get('summaries') or [])
                 if isinstance(item, dict)}
    reported = [item['id'] for item in intervals if isinstance(summaries.get(item['id']), dict)
                and 'crossings' in summaries[item['id']]]
    # the distinction is reported, not asserted away: the frozen per-summary
    # crossings may still be missing in the model group's current delivery
    if reported:
        assert result['crossings_unavailable'] == []
    else:
        assert result['crossings_unavailable'] == [item['id'] for item in intervals]
    # the additive trajectory-wide list is optional and propagated unchanged
    if 'crossings' in result['melatonin']:
        assert result['trajectory_crossings'] == result['melatonin']['crossings']
    else:
        assert 'trajectory_crossings' not in result


# ---------------------------------------------------------------------------
# 5. Boundary scan of the diagnostic module
# ---------------------------------------------------------------------------

def test_the_diagnostic_module_uses_only_public_model_names():
    text = (ROOT / 'app' / 'planning' / 'melatonin_diagnostics.py').read_text(encoding='utf-8')
    for forbidden in ('app.sleep_model.pipeline', 'app.sleep_model.simulation',
                      'app.sleep_model.homeostat', 'app.sleep_model.metrics',
                      'app.sleep_model.initialization', 'app.sleep_model.circadian',
                      'app.sleep_model.light', 'app.sleep_model.sleep_forecast',
                      'app.sleep_model.sources', 'rooster_bench'):
        assert forbidden not in text, forbidden
    # no physiology and no policy constants are re-implemented here
    for forbidden in ('betaIP', 'betaCP', 'phi_on', 'phi_off', 'Mmax', 'Hsat',
                      'alpha0', 'dH1', 'dH2'):
        assert forbidden not in text, forbidden
    # no DLMO marker is produced (the module only states that the modelled
    # synthesis crossings are not a measured DLMO)
    assert not [name for name in dir(melatonin_diagnostics) if 'dlmo' in name.lower()]
