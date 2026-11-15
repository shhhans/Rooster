"""Independent reference transcription and analytic fixtures for the Breslow13
endogenous-only melatonin model (decoupling cohort, contract step 1).

This module is the *independent* side of the melatonin validation:

* It imports **no** production module. The equations, parameters and closed
  forms are transcribed from the reviewed contract
  (``docs/melatonin-joint-contract.md`` plus the Lead-retained
  ``melatonin-breslow13-contract.md`` / ``melatonin-contract-review.md``),
  never from a production RHS or parameter helper.  The pinned source file is
  only *hash-read* (SHA-256) to prove the reviewed pin is still the pin; not a
  single line of its code is reused.
* Ground truth is analytic (closed forms).  The numeric integration of the
  reference transcription exists only to cross-check those closed forms
  (double entry), so a transcription error cannot hide behind a fixture.
* The expectations under ``tests/fixtures/melatonin/`` are **contract-derived
  analytic constants**, not frozen implementation behavior.  A disagreement
  with any implementation is adjudicated, never edited into the fixture.
  Regeneration is possible (``--rebuild-fixture``) but leaves an audit trail.
* Recorded deviations from the printed paper are part of the fixtures (see
  ``deviations.json``).  No claim of literal paper reproduction, measured
  DLMO, personal assay or biological validation is made anywhere here.

Usage (conda env ``205agent``, repo root as cwd)::

    pytest tests/test_melatonin_reference.py -v
    python -m tests.test_melatonin_reference --rebuild-fixture [--overwrite]
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import sys
from pathlib import Path

import pytest

FIXTURE_DIR = Path(__file__).resolve().parent / 'fixtures' / 'melatonin'
REPO_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Reference parameters (contract parameter table; per-second -> per-hour).
# ---------------------------------------------------------------------------

PER_SECOND = {'betaIP': 0.000783, 'betaCP': 0.000335, 'betaAP': 0.000162, 'a': 0.0010422}
PER_HOUR = {name: value * 3600.0 for name, value in PER_SECOND.items()}

PAPER_TABLE2_A = 0.0010442  # PMC3733227 Table 2 print; the pinned source uses 1.0422e-3

PARAMETERS = {
    'I0': 9500.0,
    'I1': 100.0,
    'alpha_0': 0.1,
    'beta': 0.007,        # per-minute Process L rate (source convention)
    'p': 0.5,
    'G': 37.0,
    'b': 0.4,
    'gamma': 0.13,
    'K': 0.55,            # inherited from Jewett99/Hilaire07, not a Breslow13 parameter
    'kappa': 12.0 / math.pi,
    'tauc': 24.1,
    'f': 0.99729,
    'm': 7.0,
    'eta': 0.04,
    'xi': 0.54,
    'delta': 600.0,
    'r': 15.36,
    'Mmax': 0.019513,
    'Hsat': 861.0,
    'sigma': 50.0,
    'H1_threshold': 0.001,
}

PHI_ON = 6.113 - 2.0 * math.pi    # wrapped into [-pi, pi]
PHI_OFF = 4.352 - 2.0 * math.pi   # wrapped into [-pi, pi]
PHI_SPAN = (PHI_ON - PHI_OFF) % (2.0 * math.pi)   # ~1.761 rad (not exactly 1.761)

INITIAL_STATE = (-0.219, -1.22, 0.519, 0.0, 0.0, 0.0)  # (x, xc, n, H1, H2, H3)

PINNED_SOURCE_PATH = 'app/sleep_model/sources/circadian/models.py.txt'
PINNED_SOURCE_SHA256 = 'd2635b34eebe9a8e4c59383f10413a452544a4405df6fcc8e3190aa4e7977435'
PINNED_UPSTREAM_COMMIT = 'db6eb08b6c0b0586cdac7795bf10bbbbfd323c1a'
PINNED_EOL_GUARANTEE = 'app/sleep_model/sources/** text eol=lf (added by dfecd5c)'

# Tolerance policy.  The default is rigid; only the near-boundary synthesis
# family carries a wider, documented band.  The dominant error there is the
# phase-realization resolution of the wrapped source phase (~1e-16 rad, which
# delta=600 amplifies); the 1-exp cancellation contributes comparatively little
# (~8.3e-8).  The frozen target is the exact exponential evaluation;
# a*delta*eps is an analytic reference for the attribution probe only, never a
# comparison target.  The global tolerance is never widened.
RTOL_DEFAULT = 1e-9
# The source evaluates the synthesis branch on the *wrapped* atan2 phase, so
# `phi_on - wrap(phi_on - eps)` recovers eps only to about 1e-16 rad.  With
# delta = 600 that phase-realization resolution becomes a relative activation
# error of about 1e-16/eps.  The near-boundary family therefore carries a
# per-case band of max(RTOL_DEFAULT, RTOL_PHASE_RESOLUTION / eps).  This is an
# explicit, scoped exception to the numeric regime - the global rtol is never
# widened and every other family stays rigid.
RTOL_PHASE_RESOLUTION = 1e-15


def near_boundary_band(epsilon):
    """Documented tolerance band for the near-boundary synthesis family."""
    return max(RTOL_DEFAULT, RTOL_PHASE_RESOLUTION / epsilon)


# ---------------------------------------------------------------------------
# Independent transcription of the contract equations.
#
# Evaluation order follows the pinned source so that ulp-level comparisons are
# meaningful:  A = a*activation/normalization  (source order, not
# a*(activation/normalization));  B = G*alpha*(1-n)*(1-b*x)*(1-b*xc).
# The activation itself is evaluated with -expm1(-x) == 1-exp(-x) to avoid the
# cancellation of the near-boundary family; see the tolerance policy above.
# ---------------------------------------------------------------------------

def reference_alpha(light_lux, parameters=None):
    """Photopic drive alpha(I) = alpha0 * (I/I0)^p * I/(I+I1)."""
    p = PARAMETERS if parameters is None else parameters
    return p['alpha_0'] * ((light_lux / p['I0']) ** p['p']) * (light_lux / (light_lux + p['I1']))


def reference_light_drive(x, xc, n, light_lux, parameters=None):
    """B = G * alpha * (1-n) * (1-b*x) * (1-b*xc) (source evaluation order)."""
    p = PARAMETERS if parameters is None else parameters
    alpha = reference_alpha(light_lux, p)
    return p['G'] * alpha * (1.0 - n) * (1.0 - p['b'] * x) * (1.0 - p['b'] * xc)


def reference_phase(x, xc):
    """Source convention: atan2(x, xc) with xc on the horizontal axis."""
    return math.atan2(x, xc)


def wrap_phase(phase):
    """Wrap a phase into the source's [-pi, pi) representation (half-open)."""
    return (phase + math.pi) % (2.0 * math.pi) - math.pi


def reference_synthesis(phase, *, a=None, parameters=None):
    """Piecewise synthesis drive A(phi) on the wrapped source phase.

    Active branch (strict ``phi_off < phi < phi_on``):
        A = a * (1-exp(-delta*mod(phi_on-phi))) / (1-exp(-delta*mod(phi_on-phi_off)))
    Complementary branch:
        A = a * exp(-r*mod(phi_on-phi_off))

    The complementary branch is selected at both boundaries, so the function
    jumps at ``phi_off`` (inside limit ``a`` -> small constant) and the inside
    limit at ``phi_on`` is zero.  No continuity is imposed.
    """
    p = PARAMETERS if parameters is None else parameters
    drive = PER_HOUR['a'] if a is None else a
    phase = wrap_phase(phase)
    span = (PHI_ON - PHI_OFF) % (2.0 * math.pi)
    if phase < PHI_ON and phase > PHI_OFF:
        activation = -math.expm1(-p['delta'] * ((PHI_ON - phase) % (2.0 * math.pi)))
        normalization = -math.expm1(-p['delta'] * span)
        return drive * activation / normalization
    return drive * math.exp(-p['r'] * span)


def reference_synthesis_branch(phase):
    """'inside' for the strict active branch, 'complementary' otherwise."""
    phase = wrap_phase(phase)
    return 'inside' if (phase < PHI_ON and phase > PHI_OFF) else 'complementary'


def reference_suppression_guard(H1, light_drive_B, parameters=None):
    """S = 0 only when H1 < 0.001 pM and (1 - m*B) < 0; otherwise S = 1."""
    p = PARAMETERS if parameters is None else parameters
    if H1 < p['H1_threshold'] and (1.0 - p['m'] * light_drive_B) < 0:
        return 0.0
    return 1.0


def reference_melatonin_drive(H2, parameters=None):
    """M = Mmax / (1 + exp((Hsat - H2)/sigma))."""
    p = PARAMETERS if parameters is None else parameters
    return p['Mmax'] / (1.0 + math.exp((p['Hsat'] - H2) / p['sigma']))


def reference_derv(t, state, light_lux, *, a=None, parameters=None):
    """Independent transcription of the pinned six-state right-hand side."""
    p = PARAMETERS if parameters is None else parameters
    x, xc, n, H1, H2, H3 = (float(value) for value in state)
    alpha = reference_alpha(light_lux, p)
    B = p['G'] * alpha * (1.0 - n) * (1.0 - p['b'] * x) * (1.0 - p['b'] * xc)
    gamma_term = p['gamma'] * (x / 3.0 + 4.0 / 3.0 * x ** 3 - 256.0 / 105.0 * x ** 7)
    tauc_term = (24.0 / (p['f'] * p['tauc'])) ** 2 + p['K'] * B
    M = reference_melatonin_drive(H2, p)
    S = reference_suppression_guard(H1, B, p)
    A = reference_synthesis(reference_phase(x, xc), a=a, parameters=p)
    derivative = [0.0] * 6
    derivative[0] = (1.0 / p['kappa']) * (xc + gamma_term + B - p['eta'] * M)
    derivative[1] = (1.0 / p['kappa']) * (B * xc / 3.0 - x * tauc_term - p['xi'] * M)
    derivative[2] = 60.0 * (alpha * (1.0 - n) - p['beta'] * n)
    derivative[3] = -PER_HOUR['betaIP'] * H1 + A * (1.0 - p['m'] * B) * S
    derivative[4] = PER_HOUR['betaIP'] * H1 - PER_HOUR['betaCP'] * H2 + PER_HOUR['betaAP'] * H3
    derivative[5] = -PER_HOUR['betaAP'] * H3
    return derivative


# ---------------------------------------------------------------------------
# Closed forms (independent ground truth).
# ---------------------------------------------------------------------------

def process_l_dark(n0, t_hours, parameters=None):
    """Dark (I=0): alpha = 0, B = 0 -> n(t) = n0 * exp(-60*beta*t)."""
    p = PARAMETERS if parameters is None else parameters
    return n0 * math.exp(-60.0 * p['beta'] * t_hours)


def process_l_constant_light(n0, light_lux, t_hours, parameters=None):
    """Constant light: n_eq = alpha/(alpha+beta), n(t) = n_eq + (n0-n_eq)*exp(-60(alpha+beta)t)."""
    p = PARAMETERS if parameters is None else parameters
    alpha = reference_alpha(light_lux, p)
    n_eq = alpha / (alpha + p['beta'])
    return n_eq + (n0 - n_eq) * math.exp(-60.0 * (alpha + p['beta']) * t_hours)


def clearance_H1(H10, t_hours):
    """a = 0, B = 0, H3 = 0: dH1/dt = -betaIP*H1."""
    return H10 * math.exp(-PER_HOUR['betaIP'] * t_hours)


def clearance_H2(H10, H20, t_hours):
    """Two-compartment closed form for the oral-zero, synthesis-free case."""
    k1, k2 = PER_HOUR['betaIP'], PER_HOUR['betaCP']
    return (H20 * math.exp(-k2 * t_hours)
            + k1 * H10 * (math.exp(-k1 * t_hours) - math.exp(-k2 * t_hours)) / (k2 - k1))


# ---------------------------------------------------------------------------
# Numeric integration of the reference transcription (cross-check only).
# ---------------------------------------------------------------------------

def reference_integrate(initial_state, light_lux, *, duration_hours, a=None,
                        rtol=1e-10, atol=1e-12, max_step_minutes=15.0, dense=True):
    """Integrate the reference six-state system under constant light."""
    from scipy.integrate import solve_ivp

    def rhs(t, y):
        return reference_derv(t, y, light_lux, a=a)

    solution = solve_ivp(rhs, (0.0, duration_hours), list(initial_state), method='DOP853',
                          rtol=rtol, atol=atol, max_step=max_step_minutes / 60.0,
                          dense_output=dense)
    if not solution.success:
        raise ArithmeticError('reference integration failed: ' + solution.message)
    return solution


def reference_phase_at(solution, t_hours):
    state = solution.sol(t_hours)
    return reference_phase(state[0], state[1])


def unwrap_phases(phases):
    """Unwrap a phase series so continuous rotation is visible (wrap-aware)."""
    unwrapped = []
    offset = 0.0
    previous = None
    for phase in phases:
        if previous is not None:
            delta = phase - previous
            if delta > math.pi:
                offset -= 2.0 * math.pi
            elif delta < -math.pi:
                offset += 2.0 * math.pi
        unwrapped.append(phase + offset)
        previous = phase
    return unwrapped


def _bisect_phase(solution, low, high, threshold, iterations=80):
    """Bisect phase(t) - threshold on a bracket with a sign change."""
    f_low = reference_phase_at(solution, low) - threshold
    f_high = reference_phase_at(solution, high) - threshold
    if f_low * f_high > 0:
        raise AssertionError('phase bracket does not straddle the threshold')
    for _ in range(iterations):
        middle = 0.5 * (low + high)
        f_middle = reference_phase_at(solution, middle) - threshold
        if f_low * f_middle <= 0:
            high = middle
        else:
            low, f_low = middle, f_middle
    return 0.5 * (low + high)


def synthesis_window_crossings(solution, times):
    """Detect synthesis-window transitions with an unwrapped (wrap-aware) phase.

    Definition (also recorded in the fixture): a crossing is a transition of the
    strict branch predicate ``phi_off < wrap(phi) < phi_on``.  The phase is
    unwrapped so that rotation across +/-pi is never mistaken for a crossing,
    and the A-branch jump at ``phi_off`` is not a crossing.  Each crossing is
    refined by bisection on the continuous phase inside its bracket and
    carries a direction ('rising'/'falling') plus the synthesis state it enters
    ('SynOn' = synthesis active, 'SynOff' = synthesis inactive).
    """
    phases = [reference_phase_at(solution, t) for t in times]
    unwrapped = unwrap_phases(phases)
    crossings = []
    for index in range(1, len(times)):
        low, high = unwrapped[index - 1], unwrapped[index]
        if low == high:
            continue
        for base in (PHI_OFF, PHI_ON):
            level = base + 2.0 * math.pi * round((low - base) / (2.0 * math.pi))
            if not (min(low, high) < level < max(low, high)):
                continue
            bracket_low, bracket_high = times[index - 1], times[index]
            wrapped_low = phases[index - 1]
            wrapped_high = phases[index]
            if abs(wrapped_high - wrapped_low) > math.pi:
                # A wrap happened inside the bracket; fall back to linear
                # interpolation of the unwrapped series (never a silent skip).
                fraction = (level - low) / (high - low)
                crossing_time = bracket_low + fraction * (bracket_high - bracket_low)
            else:
                crossing_time = _bisect_phase(solution, bracket_low, bracket_high, base)
            rising = high > low
            if base == PHI_OFF:
                kind = 'SynOn' if rising else 'SynOff'
            else:
                kind = 'SynOff' if rising else 'SynOn'
            crossings.append((kind, crossing_time, 'rising' if rising else 'falling'))
    return crossings


# ---------------------------------------------------------------------------
# Fixture construction (contract-derived analytic constants).
# ---------------------------------------------------------------------------

def _parameters_fixture():
    return {
        'schema_version': 1,
        'provenance': 'contract-derived analytic constants (not frozen implementation '
                      'behavior); a disagreement with any implementation is adjudicated, '
                      'never edited into this file.',
        'description': 'Independent Breslow13 reference parameters transcribed from the '
                       'reviewed contract; per-second rates are converted to per-hour.',
        'source': {
            'pinned_source_path': PINNED_SOURCE_PATH,
            'pinned_source_sha256': PINNED_SOURCE_SHA256,
            'pinned_source_eol_guarantee': PINNED_EOL_GUARANTEE,
            'upstream_commit': PINNED_UPSTREAM_COMMIT,
            'primary_paper': 'PMC3733227',
        },
        'per_second_rates': dict(PER_SECOND),
        'per_hour_rates': dict(PER_HOUR),
        'parameters': dict(PARAMETERS),
        'phase': {
            'phi_on': PHI_ON,
            'phi_off': PHI_OFF,
            'span_radians': PHI_SPAN,
            'wrap': '[-pi, pi) half-open; the source evaluates the branch on the wrapped '
                    'atan2(x, xc) output',
        },
        'initial_state': list(INITIAL_STATE),
        'initial_state_phase_rad': reference_phase(INITIAL_STATE[0], INITIAL_STATE[1]),
        'paper_table2_a': PAPER_TABLE2_A,
        'pinned_a': PER_SECOND['a'],
        'notes': [
            'a_h = 0.0010422*3600 = 3.75192 pM/h (pinned source); the printed paper table '
            'gives 1.0442e-3 before conversion.',
            'betaIP/betaCP/betaAP are per-second source rates multiplied by 3600.',
            'beta is a per-minute Process L rate; dn/dt carries the source factor 60 '
            '(60*beta = 0.42 per hour).',
            'K = 0.55 is inherited from Jewett99/Hilaire07 and is not a Breslow13 parameter.',
            'kappa = 12/pi h, tauc = 24.1 h, f = 0.99729, delta = 600, r = 15.36.',
            'phi_on - phi_off is ~1.761 rad but not exactly the literal 1.761.',
            'exp(-600*1.761) underflows to 0.0 in float64, so the activation '
            'normalization is exactly 1.0.',
        ],
    }


def _process_l_fixture():
    dark_cases = []
    for n0 in (0.519, 1.0, 0.1):
        for t_hours in (0.1, 1.0, 6.0, 24.0, 48.0):
            dark_cases.append({
                'n0': n0,
                't_hours': t_hours,
                'light_lux': 0.0,
                'expected_n': process_l_dark(n0, t_hours),
                'rtol': RTOL_DEFAULT,
            })
    algebraic = []
    for x, xc, n in ((-0.219, -1.22, 0.519), (0.0, 0.0, 0.5), (0.4, 0.6, 0.2)):
        for light_lux in (0.0, 10.0, 100.0, 300.0, 1000.0, 9500.0):
            algebraic.append({
                'x': x, 'xc': xc, 'n': n, 'light_lux': light_lux,
                'expected_alpha': reference_alpha(light_lux),
                'expected_B': reference_light_drive(x, xc, n, light_lux),
                'rtol': RTOL_DEFAULT,
            })
    constant_cases = []
    for light_lux in (10.0, 100.0, 300.0, 1000.0, 9500.0):
        alpha = reference_alpha(light_lux)
        for n0 in (0.519, 0.0, 1.0):
            for t_hours in (0.5, 2.0, 12.0, 48.0):
                constant_cases.append({
                    'n0': n0,
                    'light_lux': light_lux,
                    't_hours': t_hours,
                    'expected_alpha': alpha,
                    'expected_n_eq': alpha / (alpha + PARAMETERS['beta']),
                    'expected_n': process_l_constant_light(n0, light_lux, t_hours),
                    'rtol': RTOL_DEFAULT,
                })
    return {
        'schema_version': 1,
        'provenance': 'contract-derived analytic constants (not frozen implementation behavior).',
        'description': 'Process L closed forms: dark decay, algebraic alpha/B at fixed '
                       'state, and the constant-light n_eq trajectory.',
        'dark_cases': dark_cases,
        'algebraic_cases': algebraic,
        'constant_light_cases': constant_cases,
        'notes': [
            'Dark: alpha = 0 and B = 0, so dn/dt = -60*beta*n.',
            'Constant light: n_eq = alpha/(alpha+beta); the n equation is independent of '
            'the oscillator, so it is exactly solvable.',
            'alpha/beta conventions are model conventions (per-minute rate), not physical '
            'dimensional quantities.',
        ],
    }


def _synthesis_phase_fixture():
    cases = []
    named_phases = [
        ('inside_just_after_phi_off', PHI_OFF + 1e-6),
        ('inside_phi_off_plus_0p1', PHI_OFF + 0.1),
        ('inside_quarter', PHI_OFF + 0.25 * PHI_SPAN),
        ('inside_midpoint', 0.5 * (PHI_OFF + PHI_ON)),
        ('inside_three_quarter', PHI_OFF + 0.75 * PHI_SPAN),
        ('inside_just_before_phi_on', PHI_ON - 1e-6),
        ('boundary_phi_off', PHI_OFF),
        ('boundary_phi_on', PHI_ON),
        ('outside_just_after_phi_on', PHI_ON + 1e-6),
        ('outside_zero', 0.0),
        ('outside_minus_pi', -math.pi),
        ('outside_minus_three', -3.0),
        ('outside_plus_three', 3.0),
    ]
    for label, phase in named_phases:
        cases.append({
            'label': label,
            'phase': phase,
            'expected_branch': reference_synthesis_branch(phase),
            'expected_A': reference_synthesis(phase),
            'rtol': RTOL_DEFAULT,
        })
    one_sided = []
    for epsilon in (1e-6, 1e-9, 1e-12):
        phase = PHI_ON - epsilon
        expected = reference_synthesis(phase)
        normalization_expm1 = -math.expm1(-PARAMETERS['delta'] * PHI_SPAN)
        normalization_naive = 1.0 - math.exp(-PARAMETERS['delta'] * PHI_SPAN)
        # (a) mathematically exact target, no phase-wrap rounding involved
        exact_target = (PER_HOUR['a'] * (-math.expm1(-PARAMETERS['delta'] * epsilon))
                        / normalization_expm1)
        # (b) cancellation-prone arithmetic on the literal epsilon
        naive_direct = (PER_HOUR['a'] * (1.0 - math.exp(-PARAMETERS['delta'] * epsilon))
                        / normalization_naive)
        # (c) cancellation-prone arithmetic on the phase-recovered epsilon
        recovered = (PHI_ON - wrap_phase(phase)) % (2.0 * math.pi)
        naive_wrapped = (PER_HOUR['a'] * (1.0 - math.exp(-PARAMETERS['delta'] * recovered))
                         / normalization_naive)
        band = near_boundary_band(epsilon)
        one_sided.append({
            'epsilon': epsilon,
            'phase': phase,
            'expected_A': expected,
            'exact_target_A': exact_target,
            'naive_direct_A': naive_direct,
            'naive_wrapped_A': naive_wrapped,
            'recovered_epsilon': recovered,
            'expected_inside_limit': 0.0,
            'rtol': band,
            'tolerance_reason': 'near-boundary regime: the source evaluates the branch on the '
                                'wrapped atan2 phase, so the branch difference is recoverable '
                                'only to ~1e-16 rad; with delta=600 that becomes a relative '
                                'activation error of ~1e-16/eps.  The band is max(1e-9, '
                                '1e-15/eps) and applies to this family only; the global rtol '
                                'stays 1e-9.',
        })
    periodicity = []
    for phase in (PHI_OFF + 0.3, 0.4 * (PHI_OFF + PHI_ON), PHI_ON + 0.2, -1.0):
        periodicity.append({
            'phase': phase,
            'phase_plus_2pi': phase + 2.0 * math.pi,
            'phase_minus_2pi': phase - 2.0 * math.pi,
            'expected_A': reference_synthesis(phase),
            'rtol': RTOL_DEFAULT,
        })
    return {
        'schema_version': 1,
        'provenance': 'contract-derived analytic constants (not frozen implementation behavior).',
        'description': 'Piecewise synthesis drive A(phi) on the wrapped source phase, '
                       'including the Lead-corrected one-sided boundary semantics.',
        'phi_on': PHI_ON,
        'phi_off': PHI_OFF,
        'span_radians': PHI_SPAN,
        'cases': cases,
        'one_sided_limits': one_sided,
        'periodicity': periodicity,
        'inside_limit_at_phi_off': PER_HOUR['a'],
        'inside_limit_at_phi_on': 0.0,
        'complementary_constant': PER_HOUR['a'] * math.exp(-PARAMETERS['r'] * PHI_SPAN),
        'crossing_definition': {
            'predicate': 'strict branch active while phi_off < wrap(atan2(x, xc)) < phi_on',
            'direction': 'rising/falling of the unwrapped phase',
            'interpolation': 'bisection on the continuous phase inside the bracket',
            'wrap_handling': 'the phase series is unwrapped, so rotation across +/-pi is '
                             'never a crossing and the A-branch jump at phi_off is not one',
            'utc_mapping': 'a crossing instant maps to elapsed hours; a candidate test maps '
                           'elapsed hours to UTC instants on its own sampling grid',
            'states': {'SynOn': 'synthesis branch becomes active',
                       'SynOff': 'synthesis branch becomes inactive'},
        },
        'evaluation_order': 'A = a*activation/normalization and '
                            'B = G*alpha*(1-n)*(1-b*x)*(1-b*xc) follow the pinned source order '
                            '(multiplication is not associative).',
        'notes': [
            'Strict branch selection: phi_off < phi < phi_on; the complementary branch is '
            'selected at both boundaries.',
            'Lead source-check correction: the inside limit at phi_on is zero (not a); at '
            'phi_off the inside limit is a while the boundary selects the small constant '
            'branch, i.e. a real jump.',
            'exp(-600*span) underflows to 0.0 in float64, so the normalization is exactly '
            '1.0 and A(phi_off + eps) == a exactly for eps < ~0.519.',
            'Periodicity holds because the source evaluates the branch on the wrapped '
            'atan2 output.',
        ],
    }


def _suppression_guard_fixture():
    m = PARAMETERS['m']
    threshold = PARAMETERS['H1_threshold']
    boundary = 1.0 / m
    cases = []
    for H1 in (0.0009, threshold, 0.0011):
        for B in (0.1, boundary - 1e-6, boundary, boundary + 1e-6, 0.5):
            cases.append({
                'H1': H1,
                'B': B,
                'one_minus_mB': 1.0 - m * B,
                'expected_S': reference_suppression_guard(H1, B),
                'rtol': RTOL_DEFAULT,
            })
    ulp_cases = []
    for label, B in (('nextafter_below_boundary', math.nextafter(boundary, 0.0)),
                     ('boundary', boundary),
                     ('nextafter_above_boundary', math.nextafter(boundary, math.inf))):
        for label_H1, H1 in (('below_threshold', math.nextafter(threshold, 0.0)),
                             ('at_threshold', threshold),
                             ('above_threshold', math.nextafter(threshold, math.inf))):
            ulp_cases.append({
                'label': f'{label}|{label_H1}',
                'H1': H1,
                'B': B,
                'one_minus_mB': 1.0 - m * B,
                'expected_S': reference_suppression_guard(H1, B),
                'rtol': RTOL_DEFAULT,
            })
    return {
        'schema_version': 1,
        'provenance': 'contract-derived analytic constants (not frozen implementation behavior).',
        'description': 'Light-suppression guard S(H1, B) boundary table, including one-ulp '
                       'probes that pin the strict comparison.',
        'm': m,
        'H1_threshold': threshold,
        'suppression_boundary_B': boundary,
        'cases': cases,
        'ulp_cases': ulp_cases,
        'equality_rule': 'S = 1 at either boundary: H1 == 0.001 pM keeps S = 1, and '
                         '1 - m*B == 0 keeps S = 1; S = 0 only for H1 < 0.001 and 1 - m*B < 0.',
        'reachability': 'the exact B = 1/m boundary is a reference-level property; a joint '
                        'test can only reach guard outcomes through the candidate own light '
                        'drive, so boundary equality is asserted here, not against a candidate.',
        'notes': [
            'The guard is a published threshold switch, not a global positivity guarantee.',
            'The exact 1/m case records the float value of 1 - m*B actually evaluated.',
            'math.nextafter probes pin the strict comparison one ulp either side of the '
            'boundary and of the H1 threshold.',
        ],
    }


def _clearance_fixture():
    cases = []
    for H10, H20 in ((50.0, 80.0), (100.0, 0.0), (10.0, 200.0), (0.0, 0.0)):
        for t_hours in (0.0, 0.5, 1.0, 2.0, 6.0, 12.0, 24.0):
            cases.append({
                'H10': H10,
                'H20': H20,
                't_hours': t_hours,
                'expected_H1': clearance_H1(H10, t_hours),
                'expected_H2': clearance_H2(H10, H20, t_hours),
                'expected_H3': 0.0,
                'rtol': RTOL_DEFAULT,
            })
    return {
        'schema_version': 1,
        'provenance': 'contract-derived analytic constants (not frozen implementation behavior).',
        'description': 'Two-compartment clearance closed form for the oral-zero, '
                       'synthesis-free case (a = 0, B = 0, H3 = 0).',
        'k1_betaIP_per_hour': PER_HOUR['betaIP'],
        'k2_betaCP_per_hour': PER_HOUR['betaCP'],
        'a': 0.0,
        'light_lux': 0.0,
        'cases': cases,
        'notes': [
            'a = 0 is permitted only as a declared analytic-clearance parameter value.',
            'H1(t) = H10*exp(-k1*t); H2(t) = H20*exp(-k2*t) + k1*H10*(exp(-k1*t)-exp(-k2*t))/(k2-k1).',
            'The endocrine subsystem does not depend on the oscillator when a = 0, so the '
            'closed form is exact regardless of the coupled oscillator trajectory.',
            'H3 (oral absorption) stays identically zero in endogenous-only operation.',
        ],
    }


def _deviations_fixture():
    return {
        'schema_version': 1,
        'provenance': 'contract-derived analytic constants (not frozen implementation behavior); '
                      'a disagreement with any implementation is adjudicated, never edited in.',
        'description': 'Recorded deviations between the pinned source transcription and the '
                       'printed primary source, plus the claims this work explicitly does '
                       'not make.',
        'deviations': [
            {
                'id': 'synthesis_constant_a',
                'topic': 'maximum synthesis drive a',
                'pinned_source': '0.0010422 per second (a = 3.75192 pM/h after the x3600 conversion)',
                'printed_paper': 'PMC3733227 Table 2 prints a = 1.0442e-3',
                'decision': 'the pinned numerical version is preserved; the mismatch is reported, not smoothed',
                'difference': PAPER_TABLE2_A - PER_SECOND['a'],
                'tested_by': 'parameters.json and deviations.json assertions',
            },
            {
                'id': 'process_l_factor_60',
                'topic': 'Process L time conversion',
                'pinned_source': 'dn/dt carries an explicit factor 60 (source comments it as a '
                                 'correction of a paper typo)',
                'printed_paper': 'the printed Eq. 2 has no factor 60',
                'decision': 'the source factor 60 is used and documented (60*beta = 0.42 per hour)',
                'tested_by': 'process_l.json dark and constant-light closed forms',
            },
            {
                'id': 'inherited_K',
                'topic': 'pacemaker parameter K',
                'pinned_source': 'K = 0.55, commented as taken from Jewett99/Hilaire07',
                'printed_paper': 'K is not specified by Breslow et al. 2013',
                'decision': 'K = 0.55 is used and labelled as inherited',
                'tested_by': 'parameters.json',
            },
            {
                'id': 'per_second_conversions',
                'topic': 'endocrine rate units',
                'pinned_source': 'betaIP/betaCP/betaAP/a are per-second values multiplied by 3600',
                'printed_paper': 'the paper table mixes second-based and hour-based labels',
                'decision': 'the source x3600 conversions are used; H1/H2 are pM and a is pM/hour',
                'tested_by': 'parameters.json, clearance_analytic.json',
            },
            {
                'id': 'phase_convention',
                'topic': 'phase expression',
                'pinned_source': 'atan2(x, xc) with xc on the horizontal axis (swapped-axis note)',
                'printed_paper': 'the paper text writes tan^-1(x/xc)',
                'decision': 'the source convention is used and stated explicitly',
                'tested_by': 'parameters.json initial_state_phase_rad',
            },
            {
                'id': 'phase_window_boundaries',
                'topic': 'synthesis phase window boundaries',
                'pinned_source': 'strict phi_off < phi < phi_on with the complementary branch '
                                 'selected at the boundaries (jump at phi_off)',
                'printed_paper': 'the printed complementary inequality reverses the ordering '
                                 'for these defaults',
                'decision': 'the strict source branch is preserved; the Lead source-check '
                            'correction is authoritative: the inside limit at phi_on is zero and '
                            'the boundary at phi_off selects the small constant branch',
                'tested_by': 'synthesis_phase.json one-sided and boundary cases',
            },
            {
                'id': 'printed_unit_labels',
                'topic': 'Process L / phase-shape unit labels',
                'pinned_source': 'mixed conventions (per-minute Process L rate, per-hour '
                                 'endocrine rates)',
                'printed_paper': 'some printed entries are labelled in second-based units',
                'decision': 'the printed labels are not treated as establishing the source '
                            'conventions; the pinned numerical RHS and its documented '
                            'conversions are the executable contract',
                'tested_by': 'deviations.json (documentation record only)',
            },
            {
                'id': 'h3_is_oral_not_saliva',
                'topic': 'H3 compartment meaning',
                'pinned_source': 'H3 is the oral absorption compartment, driven by oral doses',
                'printed_paper': 'the 2007 saliva compartment is a different model',
                'decision': 'no saliva concentration is produced; H3 stays identically zero in '
                            'endogenous-only operation',
                'tested_by': 'clearance_analytic.json expected_H3 = 0.0',
            },
            {
                'id': 'phase_reference_and_dlmo_helpers',
                'topic': 'source convenience helpers phi_ref = 0.97 and cbt_to_dlmo = 7.0',
                'pinned_source': 'the pinned class carries phi_ref = 0.97 and cbt_to_dlmo = 7.0 '
                                 'as convenience constants (the latter is the source own '
                                 'CBTmin - 7 h helper)',
                'printed_paper': 'a laboratory DLMO is a measured threshold analysis',
                'decision': 'neither helper is a model output: no public output may expose a '
                            'DLMO derived from CBTmin - 7 h, and phase markers must be '
                            'model synthesis crossings (SynOn/SynOff) instead',
                'tested_by': 'deviations.json plus the joint-test output-key assertions',
            },
            {
                'id': 'evaluation_order_and_cancellation',
                'topic': 'numerical evaluation order and the near-boundary cancellation regime',
                'pinned_source': 'A = a*activation/normalization and '
                                 'B = G*alpha*(1-n)*(1-b*x)*(1-b*xc)',
                'printed_paper': 'not specified',
                'decision': 'the reference keeps the source multiplication order; the '
                            'activation is evaluated with -expm1(-x) so the near-boundary '
                            'family is mathematically exact, and that family carries an '
                            'explicit per-case band max(1e-9, 1e-15/eps) that covers the '
                            'phase-realization resolution of the wrapped source phase instead '
                            'of widening the global rtol',
                'tested_by': 'synthesis_phase.json one_sided_limits (rtol and tolerance_reason)',
            },
            {
                'id': 'pinned_source_line_endings',
                'topic': 'pinned source byte identity',
                'pinned_source': 'app/sleep_model/sources/circadian/models.py.txt is LF '
                                 'normalized (70147 bytes, no CRLF) under the eol=lf guarantee',
                'printed_paper': 'not applicable',
                'decision': 'the reference asserts the SHA-256 pin together with the '
                            'normalization state, so a checkout without the eol=lf guarantee '
                            'fails with an explicit message instead of an opaque digest mismatch',
                'tested_by': 'the pinned-source hash test in the reference suite',
            },
        ],
        'claims_explicitly_not_made': [
            'not a literal reproduction of every printed paper equation',
            'not a measured DLMO, saliva prediction or laboratory assay',
            'not a personal or individualized melatonin estimate',
            'not a clinical threshold, diagnosis or dose recommendation',
            'not evidence of improved schedule ranking or planning quality',
            'not biological validation of the model parameters',
        ],
    }


def build_fixtures():
    """Return the analytic fixture payloads keyed by file name."""
    return {
        'parameters.json': _parameters_fixture(),
        'process_l.json': _process_l_fixture(),
        'synthesis_phase.json': _synthesis_phase_fixture(),
        'suppression_guard.json': _suppression_guard_fixture(),
        'clearance_analytic.json': _clearance_fixture(),
        'deviations.json': _deviations_fixture(),
    }


def write_fixtures(overwrite=False):
    """Regenerate the analytic fixtures, leaving a digest audit trail."""
    payloads = build_fixtures()
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    audit = []
    for name, payload in sorted(payloads.items()):
        target = FIXTURE_DIR / name
        text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + '\n'
        previous = hashlib.sha256(target.read_bytes()).hexdigest() if target.exists() else None
        if previous is not None and not overwrite:
            raise SystemExit(f'REFUSING: {target} exists; pass --overwrite to regenerate')
        # Write the canonical text as LF bytes so the audit digest is the digest
        # of the file that is actually on disk (no newline translation).
        target.write_text(text, encoding='utf-8', newline='\n')
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        audit.append(f'{name} {previous or "-"} -> {digest}')
        print('wrote', name, digest)
    with (FIXTURE_DIR / 'rebuild-audit.log').open('a', encoding='utf-8') as handle:
        handle.write('rebuild ' + '; '.join(audit) + '\n')
    return audit


def load_fixture(name):
    return json.loads((FIXTURE_DIR / name).read_text(encoding='utf-8'))


# ---------------------------------------------------------------------------
# Tests: fixtures against independently recomputed contract math.
# ---------------------------------------------------------------------------

def test_reference_module_imports_no_production_code():
    tree = ast.parse(Path(__file__).read_text(encoding='utf-8'))
    imported = set()
    for node in tree.body:                      # module level only
        if isinstance(node, ast.Import):
            imported.update(alias.name.split('.')[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split('.')[0])
    assert not (imported & {'app', 'rooster_bench'}), f'production imports found: {imported}'
    assert 'scipy' not in imported, 'scipy must be imported lazily inside the integrator'


def test_pinned_source_hash_and_line_endings():
    pinned = REPO_ROOT / PINNED_SOURCE_PATH
    assert pinned.is_file(), f'pinned source missing: {pinned}'
    raw = pinned.read_bytes()
    assert b'\r\n' not in raw, ('checkout is not LF-normalized; the eol=lf guarantee for '
                                'app/sleep_model/sources/** is missing')
    assert hashlib.sha256(raw).hexdigest() == PINNED_SOURCE_SHA256, (
        'pinned source digest mismatch (expected the reviewed pin under the eol=lf guarantee)')


def test_parameter_conversions_match_the_contract():
    fixture = load_fixture('parameters.json')
    for name, value in PER_SECOND.items():
        assert fixture['per_second_rates'][name] == value
        assert math.isclose(PER_HOUR[name], value * 3600.0, rel_tol=1e-15)
        assert math.isclose(fixture['per_hour_rates'][name], PER_HOUR[name], rel_tol=1e-15)
    # Literals quoted by the contract's parameter table.
    assert math.isclose(PER_HOUR['betaIP'], 2.8188, rel_tol=1e-12)
    assert math.isclose(PER_HOUR['betaCP'], 1.206, rel_tol=1e-12)
    assert math.isclose(PER_HOUR['betaAP'], 0.5832, rel_tol=1e-12)
    assert math.isclose(PER_HOUR['a'], 3.75192, rel_tol=1e-12)
    assert fixture['pinned_a'] == 0.0010422
    assert fixture['paper_table2_a'] == 0.0010442
    assert fixture['pinned_a'] != fixture['paper_table2_a']
    assert math.isclose(60.0 * PARAMETERS['beta'], 0.42, rel_tol=1e-15)
    assert math.isclose(PHI_ON, -0.1701853071795858, rel_tol=1e-15)
    assert math.isclose(PHI_OFF, -1.931185307179586, rel_tol=1e-15)
    assert -math.pi <= PHI_OFF < PHI_ON <= math.pi
    assert math.isclose(PHI_SPAN, 1.761, rel_tol=1e-15)
    assert math.isclose(PARAMETERS['kappa'], 12.0 / math.pi, rel_tol=0.0, abs_tol=0.0)
    phase = reference_phase(INITIAL_STATE[0], INITIAL_STATE[1])
    assert math.isclose(phase, -2.9639761250984815, rel_tol=1e-15)
    assert fixture['initial_state_phase_rad'] == phase
    # The pinned initial state is outside the synthesis window (complementary branch).
    assert reference_synthesis_branch(phase) == 'complementary'
    assert math.isclose(PARAMETERS['Hsat'], 861.0, rel_tol=0.0, abs_tol=0.0)


def _integration_state(n0):
    """Reference initial state with a chosen photoreceptor fraction n0."""
    return (INITIAL_STATE[0], INITIAL_STATE[1], n0, 0.0, 0.0, 0.0)


def test_process_l_dark_decay_matches_closed_form():
    fixture = load_fixture('process_l.json')
    assert fixture['dark_cases']
    for case in fixture['dark_cases']:
        expected = process_l_dark(case['n0'], case['t_hours'])
        assert math.isclose(case['expected_n'], expected, rel_tol=case['rtol'], abs_tol=1e-300)
    # Double entry: numeric integration of the reference transcription must
    # reproduce the closed form of the (independent) n equation.  Each case is
    # integrated from its own n0, because the n equation is independent of the
    # oscillator but not of its own initial value.
    solutions = {}
    for case in fixture['dark_cases']:
        n0 = case['n0']
        if n0 not in solutions:
            solutions[n0] = reference_integrate(_integration_state(n0), 0.0, duration_hours=48.0)
        numeric = float(solutions[n0].sol(case['t_hours'])[2])
        assert math.isclose(numeric, case['expected_n'], rel_tol=1e-6, abs_tol=1e-12)


def test_process_l_constant_light_matches_closed_form():
    fixture = load_fixture('process_l.json')
    assert fixture['constant_light_cases']
    for case in fixture['constant_light_cases']:
        expected = process_l_constant_light(case['n0'], case['light_lux'], case['t_hours'])
        assert math.isclose(case['expected_n'], expected, rel_tol=case['rtol'], abs_tol=1e-300)
        assert math.isclose(case['expected_alpha'], reference_alpha(case['light_lux']),
                            rel_tol=1e-15)
        assert math.isclose(case['expected_n_eq'],
                            case['expected_alpha'] / (case['expected_alpha'] + PARAMETERS['beta']),
                            rel_tol=1e-15)
    # Double entry: numeric integration under constant light, one integration per
    # (light, n0) pair because the n equation depends on its own initial value.
    solutions = {}
    for case in fixture['constant_light_cases']:
        key = (case['light_lux'], case['n0'])
        if key not in solutions:
            solutions[key] = reference_integrate(_integration_state(case['n0']), case['light_lux'],
                                                 duration_hours=48.0)
        numeric = float(solutions[key].sol(case['t_hours'])[2])
        assert math.isclose(numeric, case['expected_n'], rel_tol=1e-6, abs_tol=1e-12)


def test_algebraic_alpha_and_light_drive_match():
    fixture = load_fixture('process_l.json')
    assert fixture['algebraic_cases']
    for case in fixture['algebraic_cases']:
        alpha = reference_alpha(case['light_lux'])
        drive = reference_light_drive(case['x'], case['xc'], case['n'], case['light_lux'])
        assert math.isclose(case['expected_alpha'], alpha, rel_tol=case['rtol'])
        assert math.isclose(case['expected_B'], drive, rel_tol=case['rtol'])
    assert reference_alpha(0.0) == 0.0
    assert reference_light_drive(0.0, 0.0, 0.0, 0.0) == 0.0
    # Source evaluation order is preserved for B.
    manual = (PARAMETERS['G'] * reference_alpha(1000.0) * (1.0 - 0.519)
              * (1.0 - PARAMETERS['b'] * -0.219) * (1.0 - PARAMETERS['b'] * -1.22))
    assert reference_light_drive(-0.219, -1.22, 0.519, 1000.0) == manual


def test_synthesis_piecewise_and_boundary_semantics():
    fixture = load_fixture('synthesis_phase.json')
    assert fixture['cases']
    for case in fixture['cases']:
        assert case['expected_branch'] == reference_synthesis_branch(case['phase'])
        assert math.isclose(case['expected_A'], reference_synthesis(case['phase']),
                            rel_tol=case['rtol'], abs_tol=1e-300)
    # Lead source-check correction: the inside limit at phi_on is zero and the
    # boundary selects the complementary constant branch (a real jump at phi_off).
    assert fixture['inside_limit_at_phi_on'] == 0.0
    assert math.isclose(fixture['inside_limit_at_phi_off'], PER_HOUR['a'], rel_tol=1e-15)
    constant = fixture['complementary_constant']
    assert math.isclose(constant, PER_HOUR['a'] * math.exp(-PARAMETERS['r'] * PHI_SPAN),
                        rel_tol=1e-15)
    assert math.isclose(constant, 6.714899241565771e-12, rel_tol=1e-12)
    boundary_off = next(case for case in fixture['cases'] if case['label'] == 'boundary_phi_off')
    boundary_on = next(case for case in fixture['cases'] if case['label'] == 'boundary_phi_on')
    assert boundary_off['expected_branch'] == 'complementary'
    assert boundary_on['expected_branch'] == 'complementary'
    inside_off = next(case for case in fixture['cases']
                      if case['label'] == 'inside_just_after_phi_off')
    # Jump at phi_off: the inside limit is a, the boundary value is the small constant.
    assert math.isclose(inside_off['expected_A'], PER_HOUR['a'], rel_tol=1e-15)
    assert inside_off['expected_A'] / boundary_off['expected_A'] > 1e9
    # Inside limit at phi_on -> 0, with the documented near-boundary band.
    for case in fixture['one_sided_limits']:
        assert case['rtol'] == near_boundary_band(case['epsilon'])
        assert case['rtol'] >= RTOL_DEFAULT
        assert case['tolerance_reason']
        # The frozen target is the reference evaluation of the source branch.
        assert math.isclose(case['expected_A'], reference_synthesis(case['phase']),
                            rel_tol=0.0, abs_tol=0.0)
        # Every legitimate evaluation variant stays inside the documented band.
        for key in ('exact_target_A', 'naive_direct_A', 'naive_wrapped_A'):
            deviation = abs(case[key] - case['expected_A']) / abs(case['expected_A'])
            assert deviation <= case['rtol'], f'{key} deviates {deviation} > {case["rtol"]}'
        # The band is an explicit exception, never a widened global policy.
        assert RTOL_DEFAULT == 1e-9
    trend = [case['expected_A'] for case in fixture['one_sided_limits']]
    assert trend[0] > trend[1] > trend[2]
    assert trend[2] < 1e-8 * PER_HOUR['a']
    # Periodicity under phi +/- 2*pi.
    for case in fixture['periodicity']:
        assert math.isclose(case['expected_A'], reference_synthesis(case['phase']),
                            rel_tol=0.0, abs_tol=0.0)
        assert math.isclose(reference_synthesis(case['phase_plus_2pi']), case['expected_A'],
                            rel_tol=0.0, abs_tol=0.0)
        assert math.isclose(reference_synthesis(case['phase_minus_2pi']), case['expected_A'],
                            rel_tol=0.0, abs_tol=0.0)
    # The crossing definition is recorded with the fixture.
    definition = fixture['crossing_definition']
    for key in ('predicate', 'direction', 'interpolation', 'wrap_handling', 'utc_mapping'):
        assert definition[key]


def test_suppression_guard_boundary_table():
    fixture = load_fixture('suppression_guard.json')
    assert fixture['cases']
    for case in fixture['cases']:
        assert case['expected_S'] == reference_suppression_guard(case['H1'], case['B'])
        assert math.isclose(case['one_minus_mB'], 1.0 - PARAMETERS['m'] * case['B'],
                            rel_tol=1e-15)
    # Documented equality semantics: either boundary keeps S = 1.
    equal_threshold = next(case for case in fixture['cases']
                           if case['H1'] == fixture['H1_threshold'] and case['one_minus_mB'] < 0)
    assert equal_threshold['expected_S'] == 1.0
    below_threshold = next(case for case in fixture['cases']
                           if case['H1'] < fixture['H1_threshold'] and case['one_minus_mB'] < 0)
    assert below_threshold['expected_S'] == 0.0
    equal_suppression = next(case for case in fixture['cases']
                             if case['H1'] < fixture['H1_threshold'] and case['one_minus_mB'] == 0.0)
    assert equal_suppression['expected_S'] == 1.0
    # One-ulp probes pin the strict comparison on both sides of both boundaries.
    assert fixture['ulp_cases']
    for case in fixture['ulp_cases']:
        assert case['expected_S'] == reference_suppression_guard(case['H1'], case['B'])
    above = next(case for case in fixture['ulp_cases']
                 if case['label'] == 'nextafter_above_boundary|below_threshold')
    assert above['one_minus_mB'] < 0.0
    assert above['expected_S'] == 0.0
    below = next(case for case in fixture['ulp_cases']
                 if case['label'] == 'nextafter_below_boundary|below_threshold')
    assert below['one_minus_mB'] > 0.0
    assert below['expected_S'] == 1.0
    at_threshold = next(case for case in fixture['ulp_cases']
                        if case['label'] == 'nextafter_above_boundary|at_threshold')
    assert at_threshold['expected_S'] == 1.0


def test_clearance_closed_form_and_oral_zero():
    fixture = load_fixture('clearance_analytic.json')
    assert fixture['cases']
    for case in fixture['cases']:
        assert math.isclose(case['expected_H1'], clearance_H1(case['H10'], case['t_hours']),
                            rel_tol=case['rtol'], abs_tol=1e-300)
        assert math.isclose(case['expected_H2'],
                            clearance_H2(case['H10'], case['H20'], case['t_hours']),
                            rel_tol=case['rtol'], abs_tol=1e-300)
        assert case['expected_H3'] == 0.0
    # Double entry: numeric integration of the reference transcription with a = 0,
    # dark light and H3 = 0 must reproduce the two-compartment closed form.
    initial = (INITIAL_STATE[0], INITIAL_STATE[1], INITIAL_STATE[2], 50.0, 80.0, 0.0)
    solution = reference_integrate(initial, 0.0, duration_hours=24.0, a=0.0)
    for case in fixture['cases']:
        if case['H10'] != 50.0 or case['H20'] != 80.0:
            continue
        state = solution.sol(case['t_hours'])
        assert math.isclose(float(state[3]), case['expected_H1'], rel_tol=1e-6, abs_tol=1e-9)
        assert math.isclose(float(state[4]), case['expected_H2'], rel_tol=1e-6, abs_tol=1e-9)
        assert float(state[5]) == 0.0
    # The coupled oscillator does not contaminate the endocrine clearance result.
    final = solution.sol(24.0)
    assert float(final[0]) != float(final[1])
    assert math.isclose(float(final[3]), clearance_H1(50.0, 24.0), rel_tol=1e-6, abs_tol=1e-9)


def test_melatonin_drive_sigmoid():
    for H2 in (0.0, 50.0, 100.0, 500.0, 861.0, 1000.0, 2000.0):
        assert math.isclose(reference_melatonin_drive(H2),
                            PARAMETERS['Mmax'] / (1.0 + math.exp((PARAMETERS['Hsat'] - H2)
                                                                   / PARAMETERS['sigma'])),
                            rel_tol=1e-15)
    assert math.isclose(reference_melatonin_drive(PARAMETERS['Hsat']), PARAMETERS['Mmax'] / 2.0,
                        rel_tol=1e-15)
    values = [reference_melatonin_drive(H2) for H2 in (0.0, 100.0, 400.0, 861.0, 1500.0)]
    assert values == sorted(values)
    assert 0.0 <= values[0] <= values[-1] <= PARAMETERS['Mmax']


def test_dark_autonomous_time_shift_preserves_crossings():
    """The dark system is autonomous, so a state advanced along its own trajectory
    reproduces the same trajectory shifted in time and the synthesis-window
    crossings shift by exactly the same amount.

    Bounds (explicit): the shift is 3.0 h, the observation window is 24.0 h, the
    full six-state vector is shifted (the pinned initial state is *not* on a
    limit cycle: H1 = H2 = 0), states are compared with rel_tol 1e-6 /
    abs_tol 1e-9 and crossing instants with abs_tol 1e-6 h.
    """
    shift = 3.0
    window = 24.0
    settle = reference_integrate(INITIAL_STATE, 0.0, duration_hours=72.0)
    state_on_orbit = settle.sol(72.0)          # full state vector, including H1/H2
    original = reference_integrate(state_on_orbit, 0.0, duration_hours=window + shift)
    shifted_state = original.sol(shift)
    shifted = reference_integrate(shifted_state, 0.0, duration_hours=window)
    for t_hours in (0.0, 1.0, 6.0, 12.0, 24.0):
        left = shifted.sol(t_hours)
        right = original.sol(t_hours + shift)
        for index in range(6):
            assert math.isclose(float(left[index]), float(right[index]), rel_tol=1e-6, abs_tol=1e-9)
    grid = [index * 0.05 for index in range(int((window + shift) / 0.05) + 1)]
    original_crossings = synthesis_window_crossings(original, grid)
    shifted_grid = [index * 0.05 for index in range(int(window / 0.05) + 1)]
    shifted_crossings = synthesis_window_crossings(shifted, shifted_grid)
    assert original_crossings, 'the dark orbit must traverse the synthesis window'
    assert shifted_crossings, 'the shifted orbit must traverse the synthesis window'
    assert {kind for kind, _, _ in original_crossings} == {'SynOn', 'SynOff'}
    expected = [(kind, t_hours - shift, direction)
                for kind, t_hours, direction in original_crossings if t_hours >= shift - 1e-9]
    assert len(expected) == len(shifted_crossings)
    for (kind, expected_time, direction), (shifted_kind, shifted_time, shifted_direction) in zip(
            expected, shifted_crossings):
        assert kind == shifted_kind
        assert direction == shifted_direction
        assert math.isclose(shifted_time, expected_time, rel_tol=0.0, abs_tol=1e-6)


def test_fixture_files_are_complete_and_deterministic():
    payloads = build_fixtures()
    assert set(payloads) == {'parameters.json', 'process_l.json', 'synthesis_phase.json',
                             'suppression_guard.json', 'clearance_analytic.json',
                             'deviations.json'}
    for name, payload in payloads.items():
        on_disk = load_fixture(name)
        assert on_disk == payload, f'{name} differs from the regenerated analytic payload'
        assert on_disk['schema_version'] == 1
        text = json.dumps(on_disk, allow_nan=False)
        for forbidden in ('generated_at', 'executed_at', 'timestamp', 'duration_s'):
            assert forbidden not in text
        assert 'contract-derived analytic constants' in on_disk['provenance']


def test_deviations_are_recorded():
    fixture = load_fixture('deviations.json')
    identifiers = {item['id'] for item in fixture['deviations']}
    required = {'synthesis_constant_a', 'process_l_factor_60', 'inherited_K',
                'per_second_conversions', 'phase_convention', 'phase_window_boundaries',
                'printed_unit_labels', 'h3_is_oral_not_saliva',
                'phase_reference_and_dlmo_helpers', 'evaluation_order_and_cancellation',
                'pinned_source_line_endings'}
    assert required <= identifiers
    assert fixture['claims_explicitly_not_made']
    entry = next(item for item in fixture['deviations'] if item['id'] == 'synthesis_constant_a')
    assert math.isclose(entry['difference'], PAPER_TABLE2_A - PER_SECOND['a'], rel_tol=1e-15)
    assert entry['difference'] > 0.0


def main(argv=None):
    parser = argparse.ArgumentParser(description='Melatonin reference fixtures.')
    parser.add_argument('--rebuild-fixture', action='store_true',
                        help='regenerate the analytic fixtures under tests/fixtures/melatonin/')
    parser.add_argument('--overwrite', action='store_true',
                        help='allow overwriting existing fixture files (audited)')
    args = parser.parse_args(argv)
    if not args.rebuild_fixture:
        parser.error('nothing to do; pass --rebuild-fixture')
    write_fixtures(overwrite=args.overwrite)
    return 0


if __name__ == '__main__':
    sys.exit(main())
