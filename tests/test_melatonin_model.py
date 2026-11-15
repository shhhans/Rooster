"""Independent tests for the opt-in endogenous-only Breslow13 component.

The expectations here are written from the paper/source equations in this
file; the module's own ``rhs``/``alpha``/``light_drive``/``synthesis_drive``
helpers are never used to build an expected value (the evaluation group's
independent transcription is separate work).

Covers: Process L closed forms, the strict phase boundaries and the
suppression guard, the analytic clearance fixture with ``a = 0``, units/time
validation, oral-zero and checkpoint seams, default-off compatibility, source
attribution/deviations, and light observability in both public paths.
"""
from datetime import datetime, timedelta, timezone
import math
from pathlib import Path

import pytest

from app.sleep_model import api, experiments
from app.sleep_model import melatonin as melatonin_module
from app.sleep_model.contracts import digest
from app.sleep_model.melatonin import (MELATONIN_DEFAULTS, MELATONIN_MODEL_VERSION,
                                       MELATONIN_UNITS, SOURCE_MANIFEST_SHA256,
                                       SOURCE_UPSTREAM_COMMIT, default_grid,
                                       melatonin_configuration, melatonin_model_config_hash)

UTC = timezone.utc
START = datetime(2027, 1, 5, 0, tzinfo=UTC)
END = datetime(2027, 1, 5, 6, tzinfo=UTC)


def iso(value):
    return value.astimezone(UTC).isoformat().replace('+00:00', 'Z')


def dt_of(value):
    return datetime.fromisoformat(value.replace('Z', '+00:00')).astimezone(UTC)


def hourly_grid(start, end, step_hours=1):
    marks = []
    cursor = start
    while cursor < end:
        marks.append(cursor)
        cursor += timedelta(hours=step_hours)
    marks.append(end)
    return marks


def independent_alpha(lux, config):
    """Paper/source algebra for the Process L activation (test-side copy)."""
    return (config['alpha_0'] * (lux / config['I0']) ** config['p']
            * lux / (lux + config['I1']))


def independent_drive(x, xc, n, lux, config):
    """Paper/source algebra for the photopic drive B (test-side copy)."""
    return (config['G'] * independent_alpha(lux, config) * (1.0 - n)
            * (1.0 - config['b'] * x) * (1.0 - config['b'] * xc))


def independent_synthesis(phi, config):
    """Paper/source algebra for A(phi) (test-side copy, strict window)."""
    phi_on, phi_off = config['phi_on'], config['phi_off']
    if phi < phi_on and phi > phi_off:
        activation = 1.0 - math.exp(-config['delta'] * ((phi_on - phi) % (2 * math.pi)))
        normalization = 1.0 - math.exp(-config['delta'] * ((phi_on - phi_off) % (2 * math.pi)))
        return config['a'] * activation / normalization
    return config['a'] * math.exp(-config['r'] * ((phi_on - phi_off) % (2 * math.pi)))


def dark_run(hours=6, step_hours=1, initial=None, config=None):
    """Constant dark public run on an hourly grid."""
    resolved = config or melatonin_configuration()
    grid = hourly_grid(START, START + timedelta(hours=hours), step_hours)
    return api.compute_melatonin(start_utc=iso(START), end_utc=iso(START + timedelta(hours=hours)),
                                  grid_utc=[iso(mark) for mark in grid],
                                  light_lux=[0.0] * (len(grid) - 1),
                                  light_provenance=['assumed_dark'] * (len(grid) - 1),
                                  initial_state=initial, config=resolved)


# ---------------------------------------------------------------------------
# 1. Process L closed forms.
# ---------------------------------------------------------------------------

def test_dark_light_gives_zero_activation_and_drive():
    config = melatonin_configuration()
    assert independent_alpha(0.0, config) == 0.0
    assert independent_drive(-0.219, -1.22, 0.519, 0.0, config) == 0.0
    result = dark_run(hours=4)
    assert all(point['B'] == 0.0 for point in result['trace'])
    assert all(point['A'] == pytest.approx(independent_synthesis(point['phase_rad'], config))
               for point in result['trace'])


def test_photoreceptor_decay_matches_closed_form_in_dark():
    config = melatonin_configuration()
    result = dark_run(hours=6)
    n0 = result['trace'][0]['n']
    for point in result['trace']:
        hours = point['elapsed_hours']
        assert point['n'] == pytest.approx(n0 * math.exp(-60.0 * config['beta'] * hours), abs=1e-9)


def test_photoreceptor_constant_light_matches_closed_form():
    config = melatonin_configuration()
    lux = 1000.0
    grid = hourly_grid(START, END)
    result = api.compute_melatonin(start_utc=iso(START), end_utc=iso(END),
                                   grid_utc=[iso(mark) for mark in grid],
                                   light_lux=[lux] * (len(grid) - 1),
                                   light_provenance=['observed'] * (len(grid) - 1))
    alpha = independent_alpha(lux, config)
    equilibrium = alpha / (alpha + config['beta'])
    n0 = result['trace'][0]['n']
    for point in result['trace']:
        hours = point['elapsed_hours']
        expected = equilibrium + (n0 - equilibrium) * math.exp(-60.0 * (alpha + config['beta']) * hours)
        assert point['n'] == pytest.approx(expected, abs=1e-9)


def test_algebraic_drive_matches_the_exposed_trace_values():
    config = melatonin_configuration()
    lux = 250.0
    grid = hourly_grid(START, END)
    result = api.compute_melatonin(start_utc=iso(START), end_utc=iso(END),
                                   grid_utc=[iso(mark) for mark in grid],
                                   light_lux=[lux] * (len(grid) - 1),
                                   light_provenance=['observed'] * (len(grid) - 1))
    for point in result['trace']:
        expected = independent_drive(point['x'], point['xc'], point['n'], lux, config)
        assert point['B'] == pytest.approx(expected, rel=1e-12, abs=1e-15)


# ---------------------------------------------------------------------------
# 2. Strict phase boundaries, periodicity and the suppression guard.
# ---------------------------------------------------------------------------

def test_synthesis_is_periodic_in_phase():
    config = melatonin_configuration()
    result = dark_run(hours=6)
    for point in result['trace']:
        # The model compares the wrapped phase (atan2 range), so periodicity is
        # checked on the wrapped argument.
        shifted = math.atan2(math.sin(point['phase_rad'] + 2 * math.pi),
                             math.cos(point['phase_rad'] + 2 * math.pi))
        assert point['A'] == pytest.approx(independent_synthesis(shifted, config),
                                           rel=1e-12, abs=1e-18)


def test_phase_boundaries_follow_the_pinned_piecewise_function():
    config = melatonin_configuration()
    phi_on, phi_off = config['phi_on'], config['phi_off']
    complementary = config['a'] * math.exp(-config['r'] * ((phi_on - phi_off) % (2 * math.pi)))
    grid = hourly_grid(START, END)
    result = api.compute_melatonin(start_utc=iso(START), end_utc=iso(END),
                                   grid_utc=[iso(mark) for mark in grid],
                                   light_lux=[0.0] * (len(grid) - 1),
                                   light_provenance=['assumed_dark'] * (len(grid) - 1))
    phases = [point['phase_rad'] for point in result['trace']]
    # Inside limit at phi_off is a; inside limit at phi_on is 0 (no continuity).
    inside = [phi for phi in phases if phi_off < phi < phi_on]
    if inside:
        lowest = min(inside, key=lambda phi: abs(phi - phi_off))
        highest = max(inside, key=lambda phi: abs(phi - phi_on))
        assert independent_synthesis(lowest, config) <= config['a'] + 1e-12
        assert independent_synthesis(highest, config) >= 0.0
    # The boundary values themselves select the small complementary constant.
    assert independent_synthesis(phi_off, config) == pytest.approx(complementary, rel=1e-12)
    assert independent_synthesis(phi_on, config) == pytest.approx(complementary, rel=1e-12)
    assert complementary < config['a'] * 1e-6


def test_suppression_guard_boundaries_are_strict():
    config = melatonin_configuration()
    drive = 1.0  # m * B > 1 for the guard's light-suppression condition
    threshold = config['H1_threshold']
    assert threshold == 0.001
    # S = 0 only when H1 < threshold and (1 - m*B) < 0, both strict.
    assert independent_drive(0.0, 0.0, 0.0, 0.0, config) == 0.0
    strong = config['G'] * independent_alpha(1000.0, config)
    assert strong * config['m'] > 1.0
    assert melatonin_module.suppression_guard(0.0, strong, config) == 0.0
    assert melatonin_module.suppression_guard(threshold - 1e-12, strong, config) == 0.0
    assert melatonin_module.suppression_guard(threshold, strong, config) == 1.0
    assert melatonin_module.suppression_guard(threshold + 1e-12, strong, config) == 1.0
    # Without strong light the guard never switches off.
    assert melatonin_module.suppression_guard(0.0, 0.0, config) == 1.0


def test_dark_run_has_no_light_suppression_and_bright_run_does():
    config = melatonin_configuration()
    grid = hourly_grid(START, END)
    dark = api.compute_melatonin(start_utc=iso(START), end_utc=iso(END),
                                 grid_utc=[iso(mark) for mark in grid],
                                 light_lux=[0.0] * (len(grid) - 1),
                                 light_provenance=['assumed_dark'] * (len(grid) - 1))
    bright = api.compute_melatonin(start_utc=iso(START), end_utc=iso(END),
                                    grid_utc=[iso(mark) for mark in grid],
                                    light_lux=[1000.0] * (len(grid) - 1),
                                    light_provenance=['observed'] * (len(grid) - 1))
    # In darkness the pineal synthesis is not light-suppressed and H1 grows;
    # under bright light the guard blocks production and H1 stays zero.
    assert dark['trace'][-1]['H1'] > 0.0
    assert bright['trace'][-1]['H1'] == 0.0
    assert bright['validity']['guard_activations'] > 0
    assert dark['validity']['guard_activations'] == 0
    assert bright['trace'][-1]['B'] > dark['trace'][-1]['B']


# ---------------------------------------------------------------------------
# 3. Analytic clearance fixture (a = 0 through the declared-experiment channel).
# ---------------------------------------------------------------------------

def test_clearance_matches_the_two_compartment_closed_form():
    config = melatonin_configuration()
    cleared = dict(config)
    cleared['a'] = 0.0
    k1, k2 = cleared['betaIP'], cleared['betaCP']
    initial = [-0.219, -1.22, 0.519, 12.0, 8.0]
    grid = hourly_grid(START, END)
    hours = [(mark - START).total_seconds() / 3600 for mark in grid]
    states = melatonin_module.integrate(hours, [0.0] * (len(grid) - 1), initial, cleared)
    for state, hour in zip(states, hours):
        H1, H2 = state[3], state[4]
        expected_H1 = initial[3] * math.exp(-k1 * hour)
        expected_H2 = (initial[4] * math.exp(-k2 * hour)
                       + k1 * initial[3] * (math.exp(-k1 * hour) - math.exp(-k2 * hour)) / (k2 - k1))
        assert H1 == pytest.approx(expected_H1, rel=1e-7, abs=1e-9)
        assert H2 == pytest.approx(expected_H2, rel=1e-7, abs=1e-9)
    # The coupled oscillator keeps evolving: clearance does not contaminate it.
    assert abs(states[-1][0] - initial[0]) > 1e-6


def test_zero_synthesis_is_reachable_only_through_the_declared_experiment_channel():
    snap = _snapshot_with_observed_light(0.0)
    with pytest.raises(ValueError, match='a=0 is only permitted'):
        api.compute_melatonin(start_utc=iso(START), end_utc=iso(END),
                              grid_utc=[iso(mark) for mark in hourly_grid(START, END)],
                              light_lux=[0.0] * 5, light_provenance=['observed'] * 5,
                              config={'a': 0.0})
    with pytest.raises(ValueError, match='a=0 is only permitted'):
        api.compute(snap, components=['melatonin'], melatonin_config={'a': 0.0},
                    end_utc=iso(END))
    # The declared channel (experiments.run) accepts it.
    result = experiments.run('clearance', snap, config={}, components=['melatonin'],
                              melatonin_config={'a': 0.0}, end_utc=iso(END))
    assert result['melatonin']['status'] == 'complete'
    assert result['experiment']['melatonin_model_config_hash'] == \
        melatonin_model_config_hash(melatonin_configuration({'a': 0.0}))
    # With a = 0 the synthesis drive is identically zero and no summary may
    # report active synthesis.
    assert all(point['A'] == 0.0 for point in result['melatonin']['trace'])
    summary = api.compute(snap, components=['melatonin'],
                          melatonin_config=melatonin_module._declared_experiment_configuration(
                              melatonin_configuration({'a': 0.0})),
                          melatonin_summaries=[{'id': 'cand-1', 'start_utc': iso(START),
                                                'end_utc': iso(END)}],
                          end_utc=iso(END))['melatonin']['summaries'][0]
    assert summary['synthesis_active_hours'] == 0.0
    assert summary['plasma_pM']['max'] == summary['plasma_pM']['min'] == 0.0


def test_declared_experiment_marker_is_not_a_public_bypass():
    source = (Path(__file__).resolve().parents[1] / 'app/sleep_model/melatonin.py').read_text(encoding='utf-8')
    assert '_DeclaredExperimentConfiguration' in source
    api_source = (Path(__file__).resolve().parents[1] / 'app/sleep_model/api.py').read_text(encoding='utf-8')
    assert 'DeclaredExperimentConfiguration' not in api_source
    experiments_source = (Path(__file__).resolve().parents[1]
                          / 'app/sleep_model/experiments.py').read_text(encoding='utf-8')
    assert '_declared_experiment_configuration' in experiments_source
    # The marker never alters the configuration identity.
    resolved = melatonin_configuration({'a': 0.0})
    marked = melatonin_module._declared_experiment_configuration(resolved)
    assert melatonin_model_config_hash(resolved) == melatonin_model_config_hash(marked)


# ---------------------------------------------------------------------------
# 4. Nonnegativity and honest validity reporting.
# ---------------------------------------------------------------------------

def test_caller_side_invalid_initial_state_raises():
    grid = [iso(mark) for mark in hourly_grid(START, END)]
    luxes = [0.0] * (len(grid) - 1)
    labels = ['observed'] * (len(grid) - 1)
    with pytest.raises(ValueError, match='nonnegative H1 and H2'):
        api.compute_melatonin(start_utc=iso(START), end_utc=iso(END), grid_utc=grid,
                              light_lux=luxes, light_provenance=labels,
                              initial_state=[0.0, 0.0, 0.5, -1.0, 0.0])
    with pytest.raises(ValueError, match=r'n in \[0, 1\]'):
        api.compute_melatonin(start_utc=iso(START), end_utc=iso(END), grid_utc=grid,
                              light_lux=luxes, light_provenance=labels,
                              initial_state=[0.0, 0.0, 1.5, 0.0, 0.0])
    with pytest.raises(ValueError, match='non-finite'):
        api.compute_melatonin(start_utc=iso(START), end_utc=iso(END), grid_utc=grid,
                              light_lux=luxes, light_provenance=labels,
                              initial_state=[float('nan'), 0.0, 0.5, 0.0, 0.0])


def test_solver_exposed_negative_state_is_reported_not_clamped():
    base = dark_run(hours=2)
    checkpoint = dict(base['final_checkpoint'])
    checkpoint['H1'] = -0.001  # a solver-produced failure carried into a resume
    resume_start = dt_of(checkpoint['at_utc'])
    grid = hourly_grid(resume_start, resume_start + timedelta(hours=2))
    resumed = api.compute_melatonin(start_utc=iso(resume_start),
                                    end_utc=iso(resume_start + timedelta(hours=2)),
                                    grid_utc=[iso(mark) for mark in grid],
                                    light_lux=[0.0] * (len(grid) - 1),
                                    light_provenance=['assumed_dark'] * (len(grid) - 1),
                                    checkpoint=checkpoint)
    assert resumed['validity']['status'] == 'partial'
    assert any(item['state'] == 'H1' for item in resumed['validity']['numerical_failures'])
    # The value is reported as it is; nothing is clamped to zero.
    assert any(point['H1'] < 0 for point in resumed['trace'])


# ---------------------------------------------------------------------------
# 5. Units, time grid and input validation.
# ---------------------------------------------------------------------------

def test_minute_form_analytic_solution_matches_the_hour_form_module():
    config = melatonin_configuration()
    cleared = dict(config)
    cleared['a'] = 0.0
    k1 = config['betaIP'] / 3600.0  # per second, from the source's table rates
    k2 = config['betaCP'] / 3600.0
    initial = [0.0, 0.0, 0.5, 20.0, 5.0]
    grid = hourly_grid(START, END)
    hours = [(mark - START).total_seconds() / 3600 for mark in grid]
    states = melatonin_module.integrate(hours, [0.0] * (len(grid) - 1), initial, cleared)
    for state, hour in zip(states, hours):
        seconds = hour * 3600.0
        expected_H1 = initial[3] * math.exp(-k1 * seconds)
        expected_H2 = (initial[4] * math.exp(-k2 * seconds)
                       + k1 * initial[3] * (math.exp(-k1 * seconds) - math.exp(-k2 * seconds)) / (k2 - k1))
        assert state[3] == pytest.approx(expected_H1, rel=1e-6, abs=1e-8)
        assert state[4] == pytest.approx(expected_H2, rel=1e-6, abs=1e-8)


def test_grid_endpoints_elapsed_hours_and_utc_offsets():
    offset = timezone(timedelta(hours=2))
    grid = [START.astimezone(offset), START.astimezone(offset) + timedelta(hours=3)]
    result = api.compute_melatonin(start_utc=iso(START), end_utc=iso(START + timedelta(hours=3)),
                                    grid_utc=[iso(mark) for mark in grid],
                                    light_lux=[0.0], light_provenance=['observed'])
    assert result['trace'][0]['elapsed_hours'] == 0.0
    assert result['trace'][-1]['elapsed_hours'] == 3.0
    assert result['trace'][0]['time_utc'] == iso(START)
    assert result['trace'][-1]['time_utc'] == iso(START + timedelta(hours=3))


def test_default_grid_covers_the_window_for_aligned_and_unaligned_cases():
    aligned = default_grid(START, END)
    assert aligned[0] == START and aligned[-1] == END
    unaligned = default_grid(START + timedelta(seconds=30), END - timedelta(seconds=45))
    assert unaligned[0] == START + timedelta(seconds=30)
    assert unaligned[-1] == END - timedelta(seconds=45)
    assert all(current > previous for previous, current in zip(unaligned, unaligned[1:]))
    short = default_grid(START, START + timedelta(seconds=20))
    assert short == [START, START + timedelta(seconds=20)]


def test_invalid_inputs_are_explicit_errors():
    grid = [iso(mark) for mark in hourly_grid(START, END)]
    with pytest.raises(ValueError, match='strictly increasing'):
        api.compute_melatonin(start_utc=iso(START), end_utc=iso(END),
                              grid_utc=[grid[0], grid[0]] + grid[2:],
                              light_lux=[0.0] * 5, light_provenance=['observed'] * 5)
    with pytest.raises(ValueError, match='cover'):
        api.compute_melatonin(start_utc=iso(START), end_utc=iso(END),
                              grid_utc=grid[1:], light_lux=[0.0] * 4,
                              light_provenance=['observed'] * 4)
    with pytest.raises(ValueError, match='nonnegative'):
        api.compute_melatonin(start_utc=iso(START), end_utc=iso(END), grid_utc=grid,
                              light_lux=[-1.0] + [0.0] * 4, light_provenance=['observed'] * 5)
    with pytest.raises(ValueError, match='finite and nonnegative'):
        api.compute_melatonin(start_utc=iso(START), end_utc=iso(END), grid_utc=grid,
                              light_lux=[float('nan')] + [0.0] * 4, light_provenance=['observed'] * 5)
    with pytest.raises(ValueError, match='light_lux length'):
        api.compute_melatonin(start_utc=iso(START), end_utc=iso(END), grid_utc=grid,
                              light_lux=[0.0] * 4, light_provenance=['observed'] * 4)
    with pytest.raises(ValueError, match='provenance'):
        api.compute_melatonin(start_utc=iso(START), end_utc=iso(END), grid_utc=grid,
                              light_lux=[0.0] * 5, light_provenance=['observed'] * 4)
    with pytest.raises(ValueError, match='non-empty strings'):
        api.compute_melatonin(start_utc=iso(START), end_utc=iso(END), grid_utc=grid,
                              light_lux=[0.0] * 5, light_provenance=[''] * 5)
    with pytest.raises(ValueError, match='oral dose events'):
        api.compute_melatonin(start_utc=iso(START), end_utc=iso(END), grid_utc=grid,
                              light_lux=[0.0] * 5, light_provenance=['observed'] * 5,
                              dose_events=[{'id': 'dose-1', 'amount_mg': 3}])
    with pytest.raises(ValueError, match='unknown melatonin parameters'):
        api.compute_melatonin(start_utc=iso(START), end_utc=iso(END), grid_utc=grid,
                              light_lux=[0.0] * 5, light_provenance=['observed'] * 5,
                              config={'bogus': 1.0})


def test_units_and_validity_are_declared():
    result = dark_run(hours=3)
    assert result['units']['H1'] == 'pmol/L (pM) endogenous pineal melatonin concentration'
    assert result['units']['A'] == 'pM/hour endogenous synthesis drive'
    assert result['units']['light_lux'] == 'photopic lux'
    assert result['units']['phase_rad'].startswith('radians')
    assert set(MELATONIN_UNITS) >= {'H1', 'H2', 'A', 'B', 'M', 'light_lux', 'phase_rad'}
    assert result['validity']['status'] == 'complete'
    assert result['validity']['numerical_failures'] == []
    assert result['status'] == 'complete'


# ---------------------------------------------------------------------------
# 6. Checkpoints, oral-zero seams and the light-prefix identity.
# ---------------------------------------------------------------------------

def test_checkpoint_resume_matches_the_single_window_run():
    grid = hourly_grid(START, END)
    whole = api.compute_melatonin(start_utc=iso(START), end_utc=iso(END),
                                   grid_utc=[iso(mark) for mark in grid],
                                   light_lux=[0.0] * (len(grid) - 1),
                                   light_provenance=['assumed_dark'] * (len(grid) - 1))
    midpoint = START + timedelta(hours=3)
    first = api.compute_melatonin(start_utc=iso(START), end_utc=iso(midpoint),
                                  grid_utc=[iso(mark) for mark in hourly_grid(START, midpoint)],
                                  light_lux=[0.0] * 3, light_provenance=['assumed_dark'] * 3)
    second_grid = hourly_grid(midpoint, END)
    second = api.compute_melatonin(start_utc=iso(midpoint), end_utc=iso(END),
                                    grid_utc=[iso(mark) for mark in second_grid],
                                    light_lux=[0.0] * (len(second_grid) - 1),
                                    light_provenance=['assumed_dark'] * (len(second_grid) - 1),
                                    checkpoint=first['final_checkpoint'])
    for key in ('x', 'xc', 'n', 'H1', 'H2'):
        assert second['trace'][-1][key] == pytest.approx(whole['trace'][-1][key], rel=1e-9, abs=1e-12)
    assert second['final_checkpoint']['checkpoint_hash'] != first['final_checkpoint']['checkpoint_hash']


def test_checkpoint_identity_mismatches_are_rejected():
    base = dark_run(hours=2)
    checkpoint = base['final_checkpoint']
    resume_start = dt_of(checkpoint['at_utc'])
    grid = hourly_grid(resume_start, resume_start + timedelta(hours=2))
    with pytest.raises(ValueError, match='model version'):
        api.compute_melatonin(start_utc=iso(resume_start), end_utc=iso(resume_start + timedelta(hours=2)),
                              grid_utc=[iso(mark) for mark in grid], light_lux=[0.0, 0.0],
                              light_provenance=['observed', 'observed'],
                              checkpoint=dict(checkpoint, melatonin_model_version='other-v0'))
    with pytest.raises(ValueError, match='configuration'):
        api.compute_melatonin(start_utc=iso(resume_start), end_utc=iso(resume_start + timedelta(hours=2)),
                              grid_utc=[iso(mark) for mark in grid], light_lux=[0.0, 0.0],
                              light_provenance=['observed', 'observed'],
                              checkpoint=dict(checkpoint, melatonin_config_hash='0' * 64))
    with pytest.raises(ValueError, match='source pin'):
        api.compute_melatonin(start_utc=iso(resume_start), end_utc=iso(resume_start + timedelta(hours=2)),
                              grid_utc=[iso(mark) for mark in grid], light_lux=[0.0, 0.0],
                              light_provenance=['observed', 'observed'],
                              checkpoint=dict(checkpoint, source_sha256='0' * 64))
    with pytest.raises(ValueError, match='checkpoint instant'):
        whole_grid = hourly_grid(START, END)
        api.compute_melatonin(start_utc=iso(START), end_utc=iso(END),
                              grid_utc=[iso(mark) for mark in whole_grid],
                              light_lux=[0.0] * (len(whole_grid) - 1),
                              light_provenance=['observed'] * (len(whole_grid) - 1),
                              checkpoint=checkpoint)


def test_changed_light_prefix_is_rejected_on_resume():
    base = dark_run(hours=2)
    checkpoint = base['final_checkpoint']
    resume_start = dt_of(checkpoint['at_utc'])
    grid = hourly_grid(resume_start, resume_start + timedelta(hours=2))
    consumed = [{'start_utc': iso(START), 'light_lux': 0.0, 'light_provenance': 'assumed_dark'}]
    with pytest.raises(ValueError, match='light prefix'):
        api.compute_melatonin(start_utc=iso(resume_start),
                              end_utc=iso(resume_start + timedelta(hours=2)),
                              grid_utc=[iso(mark) for mark in grid], light_lux=[0.0, 0.0],
                              light_provenance=['observed', 'observed'], checkpoint=checkpoint,
                              light_prefix=[{'start_utc': iso(START), 'light_lux': 900.0,
                                             'light_provenance': 'observed'}])
    # The unchanged prefix is accepted.
    accepted = api.compute_melatonin(start_utc=iso(resume_start),
                                     end_utc=iso(resume_start + timedelta(hours=2)),
                                     grid_utc=[iso(mark) for mark in grid], light_lux=[0.0, 0.0],
                                     light_provenance=['observed', 'observed'], checkpoint=checkpoint,
                                     light_prefix=consumed)
    assert accepted['status'] == 'complete'


def test_zero_h3_checkpoint_is_dropped_and_recorded_on_every_resume():
    base = dark_run(hours=2)
    six_state = dict(base['final_checkpoint'], H3=0.0)
    resume_start = dt_of(base['final_checkpoint']['at_utc'])
    grid = hourly_grid(resume_start, resume_start + timedelta(hours=2))
    resumed = api.compute_melatonin(start_utc=iso(resume_start),
                                    end_utc=iso(resume_start + timedelta(hours=2)),
                                    grid_utc=[iso(mark) for mark in grid], light_lux=[0.0, 0.0],
                                    light_provenance=['observed', 'observed'], checkpoint=six_state)
    assert 'H3' not in resumed['final_checkpoint']
    assert any('H3=0' in assumption for assumption in resumed['assumptions'])
    five_state = dict(base['final_checkpoint'])
    reference = api.compute_melatonin(start_utc=iso(resume_start),
                                      end_utc=iso(resume_start + timedelta(hours=2)),
                                      grid_utc=[iso(mark) for mark in grid], light_lux=[0.0, 0.0],
                                      light_provenance=['observed', 'observed'], checkpoint=five_state)
    for key in ('x', 'xc', 'n', 'H1', 'H2'):
        assert resumed['trace'][-1][key] == reference['trace'][-1][key]
    assert resumed['final_checkpoint']['checkpoint_hash'] == reference['final_checkpoint']['checkpoint_hash']
    with pytest.raises(ValueError, match='H3'):
        api.compute_melatonin(start_utc=iso(resume_start),
                              end_utc=iso(resume_start + timedelta(hours=2)),
                              grid_utc=[iso(mark) for mark in grid], light_lux=[0.0, 0.0],
                              light_provenance=['observed', 'observed'],
                              checkpoint=dict(base['final_checkpoint'], H3=1.0))


# ---------------------------------------------------------------------------
# 7. Default-off compatibility and component selection (M1).
# ---------------------------------------------------------------------------

def test_component_selection_keeps_melatonin_opt_in_only():
    legacy = ['burden', 'circadian', 'inertia', 'metrics', 'pressure']
    assert api.resolve_components(None) == (legacy, legacy)
    assert api.resolve_components(legacy) == (legacy, legacy)
    assert api.resolve_components(['melatonin']) == (['melatonin'], ['melatonin'])
    assert api.resolve_components(['pressure', 'melatonin']) == (['melatonin', 'pressure'],
                                                                 ['melatonin', 'pressure'])
    with pytest.raises(ValueError, match='unknown components'):
        api.resolve_components(['melatonin', 'warp'])


def test_default_selection_keeps_the_legacy_result_byte_identical():
    snap = _snapshot_with_observed_light(100.0)
    default = api.compute(snap, end_utc=iso(END))
    assert 'melatonin' not in default
    explicit = api.compute(snap, components=api.DEFAULT_COMPONENTS, end_utc=iso(END))
    assert explicit == default
    assert default['provenance']['trace_hash'] == explicit['provenance']['trace_hash']


def test_explicit_melatonin_selection_adds_only_the_opt_in_block():
    snap = _snapshot_with_observed_light(100.0)
    default = api.compute(snap, end_utc=iso(END))
    selected = api.compute(snap, components=['melatonin'], end_utc=iso(END))
    assert 'melatonin' in selected
    assert selected['trace'] == default['trace']
    assert selected['initial_state'] == default['initial_state']
    assert selected['provenance']['trace_hash'] == default['provenance']['trace_hash']
    assert selected['melatonin']['model_version'] == MELATONIN_MODEL_VERSION


def test_melatonin_config_unknown_keys_are_rejected_and_defaults_are_pinned():
    with pytest.raises(ValueError, match='unknown melatonin parameters'):
        api.compute_melatonin(start_utc=iso(START), end_utc=iso(END),
                              grid_utc=[iso(mark) for mark in hourly_grid(START, END)],
                              light_lux=[0.0] * 5, light_provenance=['observed'] * 5,
                              config={'rank_weights': [1.0]})
    config = melatonin_configuration()
    assert config['a'] == 0.0010422 * 3600.0
    assert config['betaIP'] == 0.000783 * 3600.0
    assert config['betaCP'] == 0.000335 * 3600.0
    assert config['betaAP'] == 0.000162 * 3600.0
    assert config['K'] == 0.55
    assert config['kappa'] == 12.0 / math.pi
    assert config['phi_on'] == 6.113 - 2 * math.pi
    assert config['phi_off'] == 4.352 - 2 * math.pi
    assert config['H1_threshold'] == 0.001
    assert set(MELATONIN_DEFAULTS) == set(config)


# ---------------------------------------------------------------------------
# 8. Attribution and source/paper deviations.
# ---------------------------------------------------------------------------

def test_module_keeps_the_mit_notice_and_the_pinned_source_identity():
    source = (Path(__file__).resolve().parents[1] / 'app/sleep_model/melatonin.py').read_text(encoding='utf-8')
    assert 'MIT License' in source
    assert 'Copyright (c) 2023 Kevin Hannay' in source
    assert SOURCE_UPSTREAM_COMMIT in source
    assert SOURCE_MANIFEST_SHA256 in source
    assert '10.1177/0748730412468081' in source
    result = dark_run(hours=1)
    provenance = result['provenance']
    assert provenance['source']['manifest_sha256'] == SOURCE_MANIFEST_SHA256
    assert provenance['source']['upstream_commit'] == SOURCE_UPSTREAM_COMMIT
    assert 'MIT' in provenance['source']['license']
    assert any('1.0442e-3' in deviation for deviation in provenance['deviations'])
    assert any('60' in deviation for deviation in provenance['deviations'])
    assert 'saliva_pM' not in json_text(result)
    assert 'DLMO' not in json_text(result).upper().replace('NOT A MEASURED DLMO', '')


def json_text(value):
    import json
    return json.dumps(value, default=str)


# ---------------------------------------------------------------------------
# 9. Light observability and dark/bright expressibility in both paths.
# ---------------------------------------------------------------------------

def test_standalone_trace_exposes_the_exact_consumed_light_samples():
    lux = [0.0, 120.0, 1000.0, 40.0]
    labels = ['assumed_dark', 'observed', 'observed', 'assumed_indoor']
    grid = hourly_grid(START, START + timedelta(hours=4))
    result = api.compute_melatonin(start_utc=iso(START), end_utc=iso(START + timedelta(hours=4)),
                                   grid_utc=[iso(mark) for mark in grid],
                                   light_lux=lux, light_provenance=labels)
    consumed = [(point['light_lux'], point['light_provenance']) for point in result['trace'][:-1]]
    assert consumed == [(float(value), label) for value, label in zip(lux, labels)]


def test_integrated_path_exposes_the_consumed_light_and_supports_dark_and_bright():
    for lux in (0.0, 1000.0):
        snap = _snapshot_with_observed_light(lux)
        result = api.compute(snap, components=['melatonin'], end_utc=iso(END))
        trace = result['melatonin']['trace']
        assert len(trace) >= 2
        assert {point['light_lux'] for point in trace} == {float(lux)}
        labels = {point['light_provenance'] for point in trace}
        assert labels == {'observed'}
        assert all(point['B'] == pytest.approx(independent_drive(point['x'], point['xc'], point['n'],
                                                                 float(lux), melatonin_configuration()),
                                               rel=1e-9, abs=1e-12)
                   for point in trace)


def test_summaries_follow_the_settled_shape_and_interval_semantics():
    snap = _snapshot_with_observed_light(0.0)
    result = api.compute(snap, components=['melatonin'], end_utc=iso(END),
                         melatonin_summaries=[{'id': 'cand-7', 'start_utc': iso(START),
                                               'end_utc': iso(START + timedelta(hours=2))}])
    summary = result['melatonin']['summaries'][0]
    assert summary['id'] == 'cand-7'
    assert set(summary['plasma_pM']) == {'min', 'mean', 'max'}
    assert summary['synthesis_active_hours'] >= 0.0
    assert 'units' in summary and 'validity' in summary and 'provenance' in summary
    assert summary['provenance']['initialization'].startswith('model default')
    # The per-summary crossings key is always present (empty when not requested).
    assert summary['crossings'] == []
    assert 'crossings' in summary['units']
    with pytest.raises(ValueError, match='does not intersect'):
        api.compute(snap, components=['melatonin'], end_utc=iso(END),
                    melatonin_summaries=[{'id': 'x', 'start_utc': iso(END + timedelta(hours=5)),
                                          'end_utc': iso(END + timedelta(hours=6))}])


def test_per_summary_crossings_use_half_open_attribution():
    """D1: per-summary crossings are primary; attribution is half-open.

    A record exactly at an interval start belongs to that interval, a record
    exactly at an interval end belongs to the next interval, and the
    trajectory-wide list stays unchanged and additive.
    """
    grid = hourly_grid(START, START + timedelta(hours=12))
    result = api.compute_melatonin(start_utc=iso(START), end_utc=iso(START + timedelta(hours=12)),
                                   grid_utc=[iso(mark) for mark in grid],
                                   light_lux=[0.0] * (len(grid) - 1),
                                   light_provenance=['assumed_dark'] * (len(grid) - 1),
                                   crossings=True)
    crossings = result['crossings']
    assert crossings, 'the dark limit cycle must produce synthesis crossings'
    instant = dt_of(crossings[0]['at_utc'])
    midpoint = instant + timedelta(minutes=30)

    def summaries(left, right):
        return api.compute_melatonin(
            start_utc=iso(START), end_utc=iso(START + timedelta(hours=12)),
            grid_utc=[iso(mark) for mark in grid], light_lux=[0.0] * (len(grid) - 1),
            light_provenance=['assumed_dark'] * (len(grid) - 1), crossings=True,
            summaries=[{'id': 'left', 'start_utc': iso(left), 'end_utc': iso(midpoint)},
                       {'id': 'right', 'start_utc': iso(midpoint), 'end_utc': iso(START + timedelta(hours=12))}])

    attributed = summaries(START, START + timedelta(hours=12))['summaries']
    left, right = attributed[0], attributed[1]
    assert len(left['crossings']) + len(right['crossings']) == len(crossings)
    for record in left['crossings']:
        assert iso(START) <= record['at_utc'] < iso(midpoint)
    for record in right['crossings']:
        assert iso(midpoint) <= record['at_utc'] < iso(START + timedelta(hours=12))
    # The same record must never be attributed to two summaries.
    stamps = [record['at_utc'] for record in left['crossings']] + \
              [record['at_utc'] for record in right['crossings']]
    assert len(stamps) == len(set(stamps))
    # The trajectory-wide list is unchanged and additive.
    assert result['crossings'] == crossings


def test_per_summary_crossings_boundary_is_half_open():
    """A crossing exactly at an interval end lands in the next interval only."""
    grid = hourly_grid(START, START + timedelta(hours=12))
    result = api.compute_melatonin(start_utc=iso(START), end_utc=iso(START + timedelta(hours=12)),
                                   grid_utc=[iso(mark) for mark in grid],
                                   light_lux=[0.0] * (len(grid) - 1),
                                   light_provenance=['assumed_dark'] * (len(grid) - 1),
                                   crossings=True)
    crossing = result['crossings'][0]
    instant = dt_of(crossing['at_utc'])
    # The interval boundary is placed exactly on the crossing instant: the
    # half-open rule then attributes it to the NEXT interval only.
    boundary = instant
    attributed = api.compute_melatonin(
        start_utc=iso(START), end_utc=iso(START + timedelta(hours=12)),
        grid_utc=[iso(mark) for mark in grid], light_lux=[0.0] * (len(grid) - 1),
        light_provenance=['assumed_dark'] * (len(grid) - 1), crossings=True,
        summaries=[{'id': 'before', 'start_utc': iso(START), 'end_utc': iso(boundary)},
                   {'id': 'after', 'start_utc': iso(boundary), 'end_utc': iso(START + timedelta(hours=12))}])['summaries']
    before, after = attributed[0], attributed[1]
    # The crossing instant itself falls in the second interval only.
    assert crossing['at_utc'] not in [record['at_utc'] for record in before['crossings']]
    assert crossing['at_utc'] in [record['at_utc'] for record in after['crossings']]


def test_initial_checkpoint_resume_is_not_falsely_rejected():
    """D2: the initial checkpoint consumed nothing, so its prefix digest covers
    an empty span and a prefix-verified resume must be accepted."""
    grid = hourly_grid(START, END)
    base = api.compute_melatonin(start_utc=iso(START), end_utc=iso(END),
                                 grid_utc=[iso(mark) for mark in grid],
                                 light_lux=[0.0] * (len(grid) - 1),
                                 light_provenance=['assumed_dark'] * (len(grid) - 1))
    initial = base['initial_checkpoint']
    assert initial['at_utc'] == initial['window_start_utc'] == iso(START)
    # Resuming from the initial checkpoint with a re-supplied prefix is accepted.
    resumed = api.compute_melatonin(start_utc=iso(START), end_utc=iso(END),
                                    grid_utc=[iso(mark) for mark in grid],
                                    light_lux=[0.0] * (len(grid) - 1),
                                    light_provenance=['assumed_dark'] * (len(grid) - 1),
                                    checkpoint=initial,
                                    light_prefix=[{'start_utc': iso(START), 'light_lux': 0.0,
                                                   'light_provenance': 'assumed_dark'}])
    assert resumed['status'] == 'complete'
    assert resumed['final_checkpoint']['checkpoint_hash'] == base['final_checkpoint']['checkpoint_hash']


def test_integrated_initial_checkpoint_resume_is_accepted():
    """D2 in the integrated path: resuming from the exposed initial
    checkpoint must not be rejected as a stale light history."""
    snap = _snapshot_with_observed_light(0.0)
    first = api.compute(snap, components=['melatonin'], end_utc=iso(END))
    initial = first['melatonin']['initial_checkpoint']
    resumed_snapshot = _snapshot_with_observed_light(0.0)
    resumed_snapshot['now_utc'] = iso(END)
    resumed_snapshot['melatonin_checkpoint'] = initial
    # The initial checkpoint sits at the window start, so the resume starts
    # there. The window lies in the past (the legacy forecast rule forbids a
    # window before the knowledge cutoff), so the retrospective mode is used.
    resumed = api.compute(resumed_snapshot, components=['melatonin'], mode='retrospective',
                          start_utc=iso(START), end_utc=iso(END))
    assert resumed['melatonin']['status'] == 'complete'


def test_experiments_forward_the_melatonin_request_kwargs():
    snap = _snapshot_with_observed_light(0.0)
    direct = api.compute(snap, components=['melatonin'], end_utc=iso(END),
                         melatonin_summaries=[{'id': 'cand-3', 'start_utc': iso(START),
                                               'end_utc': iso(START + timedelta(hours=2))}],
                         melatonin_crossings=True)
    experiment = experiments.run('exp-kwargs', snap, config={}, components=['melatonin'],
                                 end_utc=iso(END),
                                 melatonin_summaries=[{'id': 'cand-3', 'start_utc': iso(START),
                                                       'end_utc': iso(START + timedelta(hours=2))}],
                                 melatonin_crossings=True)
    assert experiment['melatonin']['summaries'] == direct['melatonin']['summaries']
    assert experiment['melatonin']['crossings'] == direct['melatonin']['crossings']
    assert experiment['experiment']['melatonin_model_config_hash'] == \
        direct['melatonin']['provenance']['melatonin_config_hash']


def test_crossings_are_opt_in_and_report_direction_and_validity():
    snap = _snapshot_with_observed_light(0.0)
    without = api.compute(snap, components=['melatonin'], end_utc=iso(END))
    assert without['melatonin']['crossings'] == []
    with_crossings = api.compute(snap, components=['melatonin'], end_utc=iso(END),
                                 melatonin_crossings=True)
    crossings = with_crossings['melatonin']['crossings']
    assert crossings
    for crossing in crossings:
        assert crossing['kind'] in ('SynOn', 'SynOff')
        assert crossing['direction'] in ('increasing', 'decreasing')
        assert 'validity' in crossing and 'definition' in crossing


def _snapshot_with_observed_light(lux):
    """Legacy snapshot with an explicit observed light record over the window."""
    return {
        'snapshot_id': 'melatonin-test', 'revision': 1, 'now_utc': iso(START),
        'horizon_end_utc': iso(END),
        'profile': {'home_timezone': 'UTC', 'habit_bed_local': '23:00',
                    'habit_wake_local': '07:00', 'sleep_need_hours': 8},
        'initial_state': {},
        'events': [
            {'id': 'home', 'kind': 'environment', 'location': 'home', 'availability': 'available',
             'confirmed': True, 'timezone': 'UTC', 'start_utc': iso(START - timedelta(hours=1)),
             'end_utc': iso(END + timedelta(hours=1))},
            {'id': 'light-observed', 'kind': 'light', 'metric': 'photopic_lux', 'lux': lux,
             'start_utc': iso(START - timedelta(hours=1)), 'end_utc': iso(END + timedelta(hours=1)),
             'timezone': 'UTC'},
        ],
    }
