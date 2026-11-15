"""Opt-in endogenous-only Breslow13 melatonin component.

A small Rooster-owned transcription of the pinned MIT-licensed ``Breslow13``
class of ``app/sleep_model/sources/circadian/models.py.txt`` (upstream
Arcascope/circadian, commit ``db6eb08b6c0b0586cdac7795bf10bbbbfd323c1a``,
manifest sha256 ``d2635b34eebe9a8e4c59383f10413a452544a4405df6fcc8e3190aa4e7977435``).

MIT License

Copyright (c) 2023 Kevin Hannay

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Primary paper (separate copyright; cited, not relicensed here): Breslow ER,
Phillips AJK, Huang JM, St. Hilaire MA, Klerman EB, "A Mathematical Model of
the Circadian Phase-Shifting Effects of Exogenous Melatonin", Journal of
Biological Rhythms 28(1):79-89 (2013), doi:10.1177/0748730412468081.

Endogenous-only mode: the published equations with the exogenous dose input
held identically at zero. The oral compartment ``H3`` is identically zero and
oral-dose events are rejected, never ignored. Five active states
``(x, xc, n, H1, H2)`` are exposed; the oscillator is independent of the
Forger99 state used by the legacy sleep model.

Documented deviations from the printed paper (the pinned numerical RHS and its
conversions are the executable contract):

* Process L factor ``60`` in ``dn/dt`` — a correction the pinned source
  applies to a paper typo, not a literal paper equation;
* ``K = 0.55`` is inherited from earlier models (Jewett99/Hilaire07) and is
  not specified by Breslow et al. 2013;
* endocrine rates ``betaIP``/``betaCP``/``betaAP`` and the synthesis constant
  ``a`` are multiplied by 3600 from per-second source/table values, so
  concentrations are pM and ``a`` is pM/hour;
* the phase uses ``atan2(x, xc)`` with ``xc`` as the horizontal coordinate
  (the source's swapped-axis convention);
* the pinned source uses ``a = 1.0422e-3`` per second (3.75192 per hour) while
  the indexed paper Table 2 prints ``1.0442e-3``; the pinned numerical version
  is preserved and the mismatch is reported in the result provenance;
* the paper prints second-based labels for some Process L/phase-shape entries
  that do not establish the source's mixed time conventions.

No literal paper reproduction and no biological validation is claimed: this is
a coupled model built from research protocols and population defaults, not a
personal melatonin assay, clinical assessment, or dosing recommendation. The
model produces no saliva concentration, and its phase diagnostics are modeled
synthesis crossings — never a measured DLMO.
"""
import math
from copy import deepcopy
from datetime import datetime, timedelta, timezone

import numpy as np
from scipy.integrate import solve_ivp

from .contracts import digest
from .normalize import dt, stamp

MELATONIN_MODEL_VERSION = 'rooster-melatonin-breslow13-v1'
MELATONIN_API_VERSION = 'rooster-melatonin-api-v1'
SOURCE_UPSTREAM_COMMIT = 'db6eb08b6c0b0586cdac7795bf10bbbbfd323c1a'
SOURCE_MANIFEST_SHA256 = 'd2635b34eebe9a8e4c59383f10413a452544a4405df6fcc8e3190aa4e7977435'
SOURCE_LICENSE = 'MIT (Kevin Hannay 2023, Arcascope/circadian)'
PAPER_CITATION = ('Breslow ER, Phillips AJK, Huang JM, St. Hilaire MA, Klerman EB (2013) '
                  'A Mathematical Model of the Circadian Phase-Shifting Effects of Exogenous '
                  'Melatonin. Journal of Biological Rhythms 28(1):79-89, '
                  'doi:10.1177/0748730412468081')
INITIALIZATION_LABEL = ('model default (pinned numerical initial state), not a personal estimate')

# Pinned source defaults (per-hour conversions applied as in the pinned source)
# plus this component's own solver settings (never shared with the Forger99
# physiological configuration).
MELATONIN_DEFAULTS = {
    'H1_threshold': 0.001,
    'K': 0.55,
    'I0': 9500.0, 'I1': 100.0, 'alpha_0': 0.1, 'beta': 0.007,
    'p': 0.5, 'r': 15.36, 'G': 37.0, 'b': 0.4, 'gamma': 0.13,
    'kappa': 12.0 / math.pi, 'tauc': 24.1, 'f': 0.99729, 'm': 7.0,
    'eta': 0.04, 'xi': 0.54,
    'betaIP': 0.000783 * 3600.0, 'betaCP': 0.000335 * 3600.0,
    'betaAP': 0.000162 * 3600.0, 'a': 0.0010422 * 3600.0,
    'phi_on': 6.113 - 2 * math.pi, 'phi_off': 4.352 - 2 * math.pi,
    'delta': 600.0, 'Mmax': 0.019513, 'Hsat': 861.0, 'sigma': 50.0,
    'max_step_minutes': 15.0, 'rtol': 1e-8, 'atol': 1e-10,
}

# Pinned numerical initialization (not a measured person-specific state).
DEFAULT_INITIAL_STATE = (-0.219, -1.22, 0.519, 0.0, 0.0)

MELATONIN_UNITS = {
    'time_utc': 'UTC ISO-8601 instant',
    'elapsed_hours': 'hours elapsed since the window start',
    'x': 'dimensionless circadian pacemaker state',
    'xc': 'dimensionless circadian pacemaker state',
    'n': 'dimensionless fraction of activated retinal photoreceptors in [0, 1]',
    'H1': 'pmol/L (pM) endogenous pineal melatonin concentration',
    'H2': 'pmol/L (pM) plasma melatonin concentration',
    'A': 'pM/hour endogenous synthesis drive',
    'B': 'dimensionless photopic light drive (model convention)',
    'M': 'dimensionless melatonin receptor drive (model convention)',
    'suppression_guard': 'dimensionless guard state (0 or 1)',
    'phase_rad': 'radians, atan2(x, xc) in [-pi, pi]',
    'light_lux': 'photopic lux',
    'light_provenance': 'provenance label of the consumed light sample',
    'plasma_pM': 'pmol/L (pM)',
    'synthesis_active_hours': 'hours',
}

DEVIATIONS = (
    'Process L factor 60 in dn/dt is a source-corrected paper typo, not a literal paper equation',
    'K=0.55 is inherited from Jewett99/Hilaire07 and is not specified by Breslow et al. 2013',
    'endocrine rates (betaIP, betaCP, betaAP) and the synthesis constant a are multiplied by 3600 '
    'from per-second source/table values; concentrations are pM and a is pM/hour',
    'phase uses atan2(x, xc) with xc as the horizontal coordinate (source swapped-axis convention)',
    'pinned source synthesis constant a=1.0422e-3 per second (3.75192 per hour) differs from the '
    'indexed paper Table 2 value 1.0442e-3; the pinned numerical version is preserved and reported',
    'the paper prints second-based labels for some Process L/phase-shape entries that do not '
    'establish the source mixed time conventions',
    'no literal paper reproduction and no biological validation is claimed',
)

_POSITIVE_KEYS = ('I0', 'I1', 'alpha_0', 'G', 'kappa', 'tauc', 'delta', 'Mmax', 'Hsat', 'sigma',
                  'max_step_minutes', 'rtol', 'atol')
_NONNEGATIVE_KEYS = ('K', 'beta', 'p', 'r', 'b', 'gamma', 'eta', 'xi', 'm', 'f', 'betaIP',
                     'betaCP', 'betaAP', 'a', 'H1_threshold')


class _DeclaredExperimentConfiguration(dict):
    """Internal marker for an explicitly declared experiment configuration.

    Constructed only by ``app.sleep_model.experiments.run`` so that
    non-published parameter values (the ``a = 0`` analytic-clearance fixture)
    can flow through the formal computation path without becoming a public
    bypass. The marker never changes the configuration identity: the model
    config hash covers the dictionary content only.
    """

    __slots__ = ()


def _declared_experiment_configuration(resolved):
    return _DeclaredExperimentConfiguration(resolved)


def _is_declared_experiment_configuration(config):
    return isinstance(config, _DeclaredExperimentConfiguration)


def melatonin_configuration(overrides=None):
    """Resolve and validate the versioned melatonin parameter object.

    Unknown keys are rejected. ``a`` must be nonnegative; ``a == 0`` is only
    reachable through the declared-experiment channel (see
    ``_declared_experiment_configuration``) and never through the production
    defaults.
    """
    result = deepcopy(MELATONIN_DEFAULTS)
    if overrides:
        unknown = set(overrides) - set(result)
        if unknown:
            raise ValueError('unknown melatonin parameters: ' + ','.join(sorted(unknown)))
        result.update(overrides)
    for key in _POSITIVE_KEYS:
        value = result[key]
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError('invalid positive melatonin parameter: ' + key)
    for key in _NONNEGATIVE_KEYS:
        value = result[key]
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError('invalid nonnegative melatonin parameter: ' + key)
    if result['b'] >= 1:
        raise ValueError('invalid melatonin parameter: b must be below 1')
    if result['p'] > 1:
        raise ValueError('invalid melatonin parameter: p must not exceed 1')
    if result['f'] > 1:
        raise ValueError('invalid melatonin parameter: f must not exceed 1')
    return result


def melatonin_model_config_hash(config):
    """Content hash of the resolved melatonin configuration (marker agnostic)."""
    return digest({key: config[key] for key in sorted(config)})


# ---------------------------------------------------------------------------
# Pinned endogenous-only right-hand side.
# ---------------------------------------------------------------------------

def alpha(lux, config):
    """Process L activation ``alpha0 * (I/I0)**p * I/(I+I1)`` (photopic lux)."""
    if not isinstance(lux, (int, float)) or not math.isfinite(lux) or lux < 0:
        raise ValueError('invalid photopic light')
    return config['alpha_0'] * ((lux / config['I0']) ** config['p']) * (lux / (lux + config['I1']))


def light_drive(x, xc, n, lux, config):
    """Photopic drive ``B`` with the phase-sensitivity factors."""
    return (config['G'] * alpha(lux, config) * (1.0 - n)
            * (1.0 - config['b'] * x) * (1.0 - config['b'] * xc))


def melatonin_drive(H2, config):
    """Receptor drive ``M`` from the plasma concentration."""
    return config['Mmax'] / (1.0 + math.exp((config['Hsat'] - H2) / config['sigma']))


def suppression_guard(H1, B, config):
    """The paper's near-zero guard.

    ``S = 0`` only when ``H1 < H1_threshold`` **and** ``(1 - m*B) < 0`` (both
    strict); equality at either boundary leaves ``S = 1``. This is a published
    part of the model, not a post-hoc clamp and not a global positivity
    invariant.
    """
    if H1 < config['H1_threshold'] and (1.0 - config['m'] * B) < 0:
        return 0.0
    return 1.0


def phase(x, xc):
    """``atan2(x, xc)`` — the pinned source's swapped-axis phase convention."""
    return math.atan2(x, xc)


def synthesis_drive(phi, config):
    """The exact pinned piecewise synthesis drive ``A(phi)``.

    The inside branch is active strictly for ``phi_off < phi < phi_on``; the
    complementary branch is selected at **both** boundaries (it is the small
    constant ``a*exp(-r*mod_2pi(phi_on-phi_off))``). The inside limit is ``a``
    at ``phi_off`` and ``0`` at ``phi_on``; no continuity is imposed.
    """
    phi_on, phi_off = config['phi_on'], config['phi_off']
    if phi < phi_on and phi > phi_off:
        activation = 1.0 - math.exp(-config['delta'] * ((phi_on - phi) % (2 * math.pi)))
        normalization = 1.0 - math.exp(-config['delta'] * ((phi_on - phi_off) % (2 * math.pi)))
        return config['a'] * activation / normalization
    return config['a'] * math.exp(-config['r'] * ((phi_on - phi_off) % (2 * math.pi)))


def rhs(t, state, lux, config):
    """Derivative of the five active states ``(x, xc, n, H1, H2)``.

    The oral compartment ``H3`` is identically zero, so the plasma influx term
    ``betaAP*H3`` is absent and ``dH2/dt = betaIP*H1 - betaCP*H2``.
    """
    x, xc, n, H1, H2 = (float(value) for value in state)
    activation = alpha(lux, config)
    B = (config['G'] * activation * (1.0 - n)
         * (1.0 - config['b'] * x) * (1.0 - config['b'] * xc))
    gamma_term = config['gamma'] * (x / 3.0 + 4.0 / 3.0 * x ** 3 - 256.0 / 105.0 * x ** 7)
    tauc_term = (24.0 / (config['f'] * config['tauc'])) ** 2 + config['K'] * B
    M = melatonin_drive(H2, config)
    guard = suppression_guard(H1, B, config)
    A = synthesis_drive(phase(x, xc), config)
    return np.array([
        (1.0 / config['kappa']) * (xc + gamma_term + B - config['eta'] * M),
        (1.0 / config['kappa']) * (B * xc / 3.0 - x * tauc_term - config['xi'] * M),
        60.0 * (activation * (1.0 - n) - config['beta'] * n),
        -config['betaIP'] * H1 + A * (1.0 - config['m'] * B) * guard,
        config['betaIP'] * H1 - config['betaCP'] * H2,
    ])


# ---------------------------------------------------------------------------
# Grid, light and checkpoint helpers.
# ---------------------------------------------------------------------------

def default_grid(start, end):
    """Minute grid covering ``[start, end]`` with both endpoints included.

    A non-minute-aligned window yields ``[start] + whole-minute marks strictly
    inside + [end]``; a window shorter than one minute yields exactly the two
    endpoints.
    """
    marks = []
    cursor = start.replace(second=0, microsecond=0)
    if cursor <= start:
        cursor += timedelta(minutes=1)
    while cursor < end:
        marks.append(cursor)
        cursor += timedelta(minutes=1)
    return [start] + marks + [end]


def _validate_grid(grid, start, end):
    if len(grid) < 2:
        raise ValueError('melatonin sampling grid needs at least two instants')
    instants = []
    for value in grid:
        instant = dt(value)
        if not math.isfinite(instant.timestamp()):
            raise ValueError('non-finite melatonin sampling grid instant')
        instants.append(instant)
    for previous, current in zip(instants, instants[1:]):
        if current <= previous:
            raise ValueError('melatonin sampling grid must be strictly increasing')
    if stamp(instants[0]) != stamp(start) or stamp(instants[-1]) != stamp(end):
        raise ValueError('melatonin sampling grid must cover [start_utc, end_utc] exactly')
    return instants


def _canonical_light_series(samples, start, end):
    """Minute-resolution ``(time_utc, lux, label)`` series of a light history.

    ``samples`` is an ordered list of constant-light intervals, each a mapping
    with ``start_utc``, ``light_lux`` and ``light_provenance``. The canonical
    series is grid-independent, so two runs of the same declared scenario hash
    identically.
    """
    ordered = sorted(samples, key=lambda item: dt(item['start_utc']))
    rows = []
    marks = []
    cursor = start.replace(second=0, microsecond=0)
    if cursor < start:
        cursor += timedelta(minutes=1)
    while cursor < end:
        marks.append(cursor)
        cursor += timedelta(minutes=1)
    for mark in marks:
        value, label = _light_in_effect(ordered, mark)
        rows.append([stamp(mark), value, label])
    return rows


def _light_in_effect(samples, instant):
    """Light value and provenance label in effect at ``instant`` (left-edge rule)."""
    if not samples:
        raise ValueError('melatonin light history is empty')
    chosen = samples[0]
    for sample in samples:
        if dt(sample['start_utc']) <= instant:
            chosen = sample
        else:
            break
    return chosen['light_lux'], chosen['light_provenance']


def _validated_light_rows(rows, allow_empty=False):
    """Validate explicit light-history rows (start instant, lux, provenance)."""
    if not isinstance(rows, (list, tuple)):
        raise TypeError('melatonin light history rows must be a sequence')
    if not rows and not allow_empty:
        raise ValueError('melatonin light history rows are required')
    prepared = []
    for row in rows:
        if not isinstance(row, dict):
            raise TypeError('melatonin light history entries must be mappings')
        for key in ('start_utc', 'light_lux', 'light_provenance'):
            if key not in row:
                raise ValueError('melatonin light history entry missing field: ' + key)
        value = row['light_lux']
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError('melatonin light history lux must be a number')
        if not math.isfinite(value) or value < 0:
            raise ValueError('melatonin light history lux must be finite and nonnegative')
        label = row['light_provenance']
        if not isinstance(label, str) or not label.strip():
            raise ValueError('melatonin light history provenance labels must be non-empty strings')
        prepared.append({'start_utc': stamp(dt(row['start_utc'])), 'light_lux': float(value),
                         'light_provenance': label})
    return sorted(prepared, key=lambda item: item['start_utc'])


def _light_prefix_hash(rows):
    return digest({'series': rows})


def _validate_light(lux_samples, provenance_samples):
    if not isinstance(lux_samples, (list, tuple)) or not lux_samples:
        raise ValueError('melatonin light_lux samples are required')
    luxes = []
    for value in lux_samples:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError('melatonin light_lux samples must be numbers')
        if not math.isfinite(value) or value < 0:
            raise ValueError('melatonin light_lux samples must be finite and nonnegative')
        luxes.append(float(value))
    if provenance_samples is None:
        labels = ['assumed'] * len(luxes)
    else:
        if not isinstance(provenance_samples, (list, tuple)):
            raise TypeError('melatonin light_provenance samples must be a sequence')
        if len(provenance_samples) != len(luxes):
            raise ValueError('melatonin light_provenance length must match light_lux length')
        labels = []
        for label in provenance_samples:
            if not isinstance(label, str) or not label.strip():
                raise ValueError('melatonin light_provenance labels must be non-empty strings')
            labels.append(label)
    return luxes, labels


def _initial_state_values(initial_state):
    if initial_state is None:
        return list(DEFAULT_INITIAL_STATE), INITIALIZATION_LABEL
    if isinstance(initial_state, dict):
        missing = [key for key in ('x', 'xc', 'n', 'H1', 'H2') if key not in initial_state]
        if missing:
            raise ValueError('incomplete melatonin initial state: ' + ','.join(missing))
        values = [initial_state[key] for key in ('x', 'xc', 'n', 'H1', 'H2')]
    elif isinstance(initial_state, (list, tuple)):
        if len(initial_state) != 5:
            raise ValueError('melatonin initial state needs five values (x, xc, n, H1, H2)')
        values = list(initial_state)
    else:
        raise TypeError('melatonin initial state must be a mapping or a five-value sequence')
    numbers = []
    for value in values:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError('melatonin initial state values must be numbers')
        if not math.isfinite(value):
            raise ValueError('non-finite melatonin initial state')
        numbers.append(float(value))
    if numbers[3] < 0 or numbers[4] < 0:
        raise ValueError('melatonin initial state requires nonnegative H1 and H2')
    if not 0 <= numbers[2] <= 1:
        raise ValueError('melatonin initial state requires n in [0, 1]')
    return numbers, 'caller supplied initial state'


def _checkpoint_state(checkpoint):
    missing = [key for key in ('x', 'xc', 'n', 'H1', 'H2') if key not in checkpoint]
    if missing:
        raise ValueError('incomplete melatonin checkpoint: ' + ','.join(missing))
    values = []
    for key in ('x', 'xc', 'n', 'H1', 'H2'):
        value = checkpoint[key]
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError('melatonin checkpoint state values must be numbers')
        if not math.isfinite(value):
            raise ValueError('non-finite melatonin checkpoint state')
        values.append(float(value))
    return values


def _make_checkpoint(values, at, window_start, config_hash, light_hash):
    state = {'x': values[0], 'xc': values[1], 'n': values[2], 'H1': values[3], 'H2': values[4]}
    checkpoint = dict(state)
    checkpoint.update({
        'at_utc': stamp(at),
        'window_start_utc': stamp(window_start),
        'melatonin_model_version': MELATONIN_MODEL_VERSION,
        'melatonin_api_version': MELATONIN_API_VERSION,
        'melatonin_config_hash': config_hash,
        'source_sha256': SOURCE_MANIFEST_SHA256,
        'light_prefix_hash': light_hash,
    })
    checkpoint['checkpoint_hash'] = digest({
        'state': state, 'at_utc': checkpoint['at_utc'],
        'window_start_utc': checkpoint['window_start_utc'],
        'melatonin_config_hash': config_hash,
        'melatonin_model_version': MELATONIN_MODEL_VERSION,
        'source_sha256': SOURCE_MANIFEST_SHA256,
        'light_prefix_hash': light_hash,
    })
    return checkpoint


# ---------------------------------------------------------------------------
# Solver.
# ---------------------------------------------------------------------------

def integrate(grid_hours, light_per_interval, initial, config):
    """Integrate interval by interval (the pinned source steps between points).

    Returns the state at every grid point. A solver that cannot integrate
    raises ``ArithmeticError`` with the solver message (the convention of the
    existing circadian module); domain problems of the exposed states are
    reported by the caller as validity markers, never clamped.
    """
    states = [np.array(initial, dtype=float)]
    current = np.array(initial, dtype=float)
    for index, lux in enumerate(light_per_interval):
        span = grid_hours[index + 1] - grid_hours[index]
        if span <= 0:
            raise ValueError('melatonin sampling grid must be strictly increasing')
        solution = solve_ivp(lambda t, s: rhs(t, s, lux, config), (0.0, span), current,
                             method='DOP853', rtol=config['rtol'], atol=config['atol'],
                             max_step=config['max_step_minutes'] / 60.0)
        if not solution.success or not np.all(np.isfinite(solution.y)):
            raise ArithmeticError('melatonin integration failed: ' + solution.message)
        current = solution.y[:, -1]
        states.append(current)
    return states


# ---------------------------------------------------------------------------
# Diagnostics: synthesis crossings and interval summaries.
# ---------------------------------------------------------------------------

def synthesis_active(phi, config):
    """Strict synthesis-window membership (the model's own branch condition)."""
    return config['phi_off'] < phi < config['phi_on']


def synthesis_crossings(trace, config):
    """Modeled synthesis phase crossings (``SynOn``/``SynOff``).

    The synthesis window ``(phi_off, phi_on)`` is contiguous inside
    ``[-pi, pi]``; the wrap concern is the phase *trajectory* crossing
    ``+-pi``. The raw ``atan2(x, xc)`` series is cumulatively unwrapped, the
    window membership uses the raw strict comparison, the crossed boundary
    follows from the transition direction, and the instant is interpolated
    linearly on the unwrapped phase. A crossing is invalid when the raw phase
    difference between the two samples exceeds ``pi`` (the unwrap direction
    would be ambiguous) or the sample spacing is not strictly increasing.
    """
    if len(trace) < 2:
        return []
    raw = np.array([point['phase_rad'] for point in trace], dtype=float)
    unwrapped = np.unwrap(raw)
    crossings = []
    for index in range(len(trace) - 1):
        left, right = trace[index], trace[index + 1]
        left_inside = synthesis_active(left['phase_rad'], config)
        right_inside = synthesis_active(right['phase_rad'], config)
        if left_inside == right_inside:
            continue
        delta = unwrapped[index + 1] - unwrapped[index]
        if delta == 0:
            continue
        direction = 'increasing' if delta > 0 else 'decreasing'
        boundary = config['phi_off'] if (not left_inside and direction == 'increasing') else \
            config['phi_on'] if (not left_inside and direction == 'decreasing') else \
            config['phi_on'] if (left_inside and direction == 'increasing') else config['phi_off']
        revolutions = round((unwrapped[index] - boundary) / (2 * math.pi))
        target = boundary + revolutions * 2 * math.pi
        span = unwrapped[index + 1] - unwrapped[index]
        fraction = (target - unwrapped[index]) / span
        valid = True
        if abs(right['phase_rad'] - left['phase_rad']) > math.pi:
            valid = False
        if fraction < 0 or fraction > 1:
            valid = False
        elapsed_left = left['elapsed_hours']
        elapsed_right = right['elapsed_hours']
        crossing_elapsed = elapsed_left + fraction * (elapsed_right - elapsed_left)
        crossings.append({
            'kind': 'SynOn' if right_inside else 'SynOff',
            'at_utc': stamp(dt(left['time_utc']) + timedelta(hours=crossing_elapsed - elapsed_left)),
            'direction': direction,
            'phase_rad': boundary,
            'validity': {'valid': valid,
                         'reason': None if valid else 'ambiguous phase wrap between the bracketing samples'},
            'definition': ('phase atan2(x, xc) crossing the strict synthesis window '
                           '(phi_off, phi_on); a modeled phase diagnostic, not a measured DLMO'),
        })
    return crossings


def interval_summaries(trace, requests, start, end, provenance, crossings=None):
    """Per-request plasma/synthesis summaries over the requested intervals.

    Two distinct interval rules apply and are deliberately not conflated:

    * **Statistics** use the grid rows whose ``time_utc`` lies inside
      ``[start_utc, end_utc]`` of the request — **both endpoints inclusive**.
      ``plasma_pM`` is min/mean/max of those rows' ``H2`` and
      ``synthesis_active_hours`` sums the sample intervals whose left-edge row
      is synthesis-active (phase inside the strict window and ``A > 0``).
    * **Crossing attribution** is **half-open**: a crossing record belongs to
      the request when its interpolated ``at_utc`` satisfies
      ``interval.start_utc <= at_utc < interval.end_utc``. A record exactly at
      an interval's start instant belongs to that interval, a record exactly
      at an interval's end instant belongs to the next interval, and a record
      exactly at the window's final end instant belongs to no summary (it
      stays in the trajectory-wide list). Each crossing is therefore attributed
      to at most one summary.

    ``id`` is echoed verbatim; an interval that does not intersect the
    evaluation window raises.
    """
    rows = []
    for request in requests or []:
        if not isinstance(request, dict):
            raise TypeError('melatonin summary requests must be mappings')
        for key in ('id', 'start_utc', 'end_utc'):
            if key not in request:
                raise ValueError('melatonin summary request missing field: ' + key)
        identifier = request['id']
        left = dt(request['start_utc'])
        right = dt(request['end_utc'])
        if right <= left:
            raise ValueError('melatonin summary interval must have end_utc after start_utc')
        if right <= start or left >= end:
            raise ValueError('melatonin summary interval does not intersect the evaluation window')
        selected = [point for point in trace if left <= dt(point['time_utc']) <= right]
        if not selected:
            raise ValueError('melatonin summary interval contains no sample')
        plasma = [point['H2'] for point in selected]
        active_hours = 0.0
        for index in range(len(selected) - 1):
            point = selected[index]
            following = selected[index + 1]
            hours = (dt(following['time_utc']) - dt(point['time_utc'])).total_seconds() / 3600
            if hours <= 0:
                continue
            inside = synthesis_active(point['phase_rad'], provenance['config'])
            if inside and point['A'] > 0:
                active_hours += hours
        attributed = [record for record in (crossings or [])
                      if left <= dt(record['at_utc']) < right]
        rows.append({
            'id': identifier,
            'start_utc': stamp(left),
            'end_utc': stamp(right),
            'plasma_pM': {'min': min(plasma), 'mean': sum(plasma) / len(plasma), 'max': max(plasma)},
            'synthesis_active_hours': active_hours,
            'crossings': attributed,
            'sample_count': len(selected),
            'units': {'plasma_pM': MELATONIN_UNITS['plasma_pM'],
                      'synthesis_active_hours': MELATONIN_UNITS['synthesis_active_hours'],
                      'crossings': 'modeled synthesis phase crossings attributed to this interval by the '
                                   'half-open rule start_utc <= at_utc < end_utc (empty when crossings were '
                                   'not requested or none fell inside the interval)'},
            'validity': {
                'finite_concentrations': all(math.isfinite(value) for value in plasma),
                'nonnegative_concentrations': all(value >= 0 for value in plasma),
                'crossing_count': len(attributed),
            },
            'provenance': dict(provenance['summary']),
        })
    return rows


# ---------------------------------------------------------------------------
# Public computation.
# ---------------------------------------------------------------------------

def compute_melatonin(*, start_utc, end_utc, grid_utc=None, light_lux, light_provenance=None,
                       initial_state=None, checkpoint=None, config=None, dose_events=None,
                       summaries=None, crossings=False, light_prefix=None, identity_light=None):
    """Full endogenous-only Breslow13 computation over an explicit window.

    ``grid_utc`` must be strictly increasing and cover ``[start_utc, end_utc]``
    exactly (default: the UTC minute grid with both endpoints included).
    ``light_lux`` holds one nonnegative finite photopic value per grid
    interval (``len(light_lux) == len(grid_utc) - 1``), piecewise constant per
    interval, with per-sample provenance labels.

    ``dose_events``: any non-empty value raises — the endogenous-only mode has
    no oral input and the oral compartment ``H3`` stays identically zero.

    ``checkpoint``: an optional five-state checkpoint; its identity (model
    version, config hash, source pin, ``at_utc`` and the light-prefix hash) is
    verified, and a resume must start exactly at the checkpoint instant. The
    caller may re-supply the consumed light prefix (``light_prefix``) for
    explicit re-verification — it is canonicalized over the checkpoint's
    **consumed span** ``[window_start_utc, at_utc)`` and compared with the
    stored ``light_prefix_hash``; when it is omitted the stored hash is
    authoritative and only state, config, version and ``at_utc`` are
    re-verified. ``identity_light`` supplies the canonical light series used
    for the new checkpoint's identity (the caller's own samples by default).
    The **initial** checkpoint consumed nothing, so its light-prefix digest
    covers its empty consumed span; the **final** checkpoint's digest covers
    the window it actually consumed.

    ``summaries``: the requested interval summaries. Two distinct interval
    rules apply and are not conflated: the **statistics** use the grid rows
    inside ``[start_utc, end_utc]`` (both endpoints inclusive), while the
    **crossing attribution** is half-open — a crossing belongs to an interval
    when ``interval.start_utc <= at_utc < interval.end_utc`` (a record at an
    interval's end instant belongs to the next interval, a record at the
    window's final end instant belongs to no summary). The per-summary
    ``crossings`` key is always present and empty when crossings were not
    requested or none fell inside the interval.

    Returns the five state traces, units, validity, assumptions, provenance,
    initial/final checkpoints, the requested interval summaries and — strictly
    opt-in — the modeled synthesis phase crossings (trajectory-wide; the
    per-summary lists are filtered copies of it). Never a saliva concentration
    and never a DLMO.
    """
    declared = _is_declared_experiment_configuration(config)
    resolved = melatonin_configuration(config)
    if resolved['a'] == 0 and not declared:
        raise ValueError('a=0 is only permitted for the declared analytic-clearance experiment channel')
    if dose_events:
        raise ValueError('oral dose events are not supported in endogenous-only mode')
    start = dt(start_utc)
    end = dt(end_utc)
    if end <= start:
        raise ValueError('melatonin window must have end_utc after start_utc')
    grid = default_grid(start, end) if grid_utc is None else _validate_grid(grid_utc, start, end)
    luxes, labels = _validate_light(light_lux, light_provenance)
    if len(luxes) != len(grid) - 1:
        raise ValueError('melatonin light_lux length must be len(grid_utc) - 1')
    assumptions = [
        'endogenous-only mode: the oral compartment H3 is identically zero and oral events are rejected',
        'no saliva concentration is produced by this model',
        'SynOn/SynOff are modeled synthesis phase diagnostics, not a measured DLMO',
        'photopic lux approximates circadian-effective retinal light; it is not a melatonin assay',
        'population/protocol defaults; not an individualized hormone estimate',
        INITIALIZATION_LABEL,
    ]

    # --- identity and resume -------------------------------------------------
    config_hash = melatonin_model_config_hash(resolved)
    values = None
    initialization_label = INITIALIZATION_LABEL
    window_start = start
    stored_light_hash = None
    if checkpoint is not None:
        if not isinstance(checkpoint, dict):
            raise TypeError('melatonin checkpoint must be a mapping')
        if checkpoint.get('melatonin_model_version') != MELATONIN_MODEL_VERSION:
            raise ValueError('stale or incompatible melatonin checkpoint: model version')
        if checkpoint.get('melatonin_config_hash') != config_hash:
            raise ValueError('stale or incompatible melatonin checkpoint: configuration')
        if checkpoint.get('source_sha256') != SOURCE_MANIFEST_SHA256:
            raise ValueError('stale or incompatible melatonin checkpoint: source pin')
        if stamp(start) != checkpoint.get('at_utc'):
            raise ValueError('melatonin resume must start exactly at the checkpoint instant')
        if 'H3' in checkpoint:
            if checkpoint['H3'] != 0:
                raise ValueError('endogenous-only mode requires H3 to be identically zero')
            assumptions.append('external checkpoint carried H3=0; the oral compartment was dropped')
        values = _checkpoint_state(checkpoint)
        initialization_label = 'resumed checkpoint (not a personal estimate)'
        window_start = dt(checkpoint.get('window_start_utc', checkpoint['at_utc']))
        stored_light_hash = checkpoint.get('light_prefix_hash')
    if values is None:
        values, initialization_label = _initial_state_values(initial_state)

    # --- light identity -------------------------------------------------------
    own_samples = [{'start_utc': stamp(grid[index]), 'light_lux': luxes[index],
                    'light_provenance': labels[index]} for index in range(len(luxes))]
    if light_prefix is not None and checkpoint is None:
        raise ValueError('melatonin light_prefix is only meaningful when resuming a checkpoint')
    if checkpoint is not None and light_prefix is not None:
        consumed_span_empty = dt(checkpoint['at_utc']) <= window_start
        check_rows = _canonical_light_series(
            _validated_light_rows(light_prefix, allow_empty=consumed_span_empty),
            window_start, dt(checkpoint['at_utc']))
        if _light_prefix_hash(check_rows) != stored_light_hash:
            raise ValueError('stale or revised light history: the consumed light prefix no longer '
                             'matches the checkpoint identity; re-initialize')
    if identity_light is not None:
        identity_rows = _canonical_light_series(_validated_light_rows(identity_light), start, end)
    else:
        identity_rows = _canonical_light_series(own_samples, start, end)

    # --- integration ----------------------------------------------------------
    grid_hours = [(instant - start).total_seconds() / 3600 for instant in grid]
    states = integrate(grid_hours, luxes, values, resolved)

    rows = []
    failures = []
    guard_activations = 0
    for index, state in enumerate(states):
        x, xc, n, H1, H2 = (float(value) for value in state)
        lux = luxes[index] if index < len(luxes) else luxes[-1]
        label = labels[index] if index < len(labels) else labels[-1]
        B = light_drive(x, xc, n, lux, resolved)
        M = melatonin_drive(H2, resolved)
        guard = suppression_guard(H1, B, resolved)
        A = synthesis_drive(phase(x, xc), resolved)
        if guard == 0:
            guard_activations += 1
        for name, value in (('H1', H1), ('H2', H2), ('n', n)):
            if not math.isfinite(value):
                failures.append({'time_utc': stamp(grid[index]), 'state': name, 'value': value,
                                 'reason': 'non-finite exposed state'})
            elif name != 'n' and value < 0:
                failures.append({'time_utc': stamp(grid[index]), 'state': name, 'value': value,
                                 'reason': 'negative exposed concentration'})
            elif name == 'n' and not 0 <= value <= 1:
                failures.append({'time_utc': stamp(grid[index]), 'state': name, 'value': value,
                                 'reason': 'exposed state outside [0, 1]'})
        rows.append({
            'time_utc': stamp(grid[index]),
            'elapsed_hours': grid_hours[index],
            'x': x, 'xc': xc, 'n': n, 'H1': H1, 'H2': H2,
            'phase_rad': phase(x, xc),
            'A': A, 'B': B, 'M': M, 'suppression_guard': guard,
            'light_lux': lux, 'light_provenance': label,
        })

    # --- checkpoints ----------------------------------------------------------
    final_values = [rows[-1]['x'], rows[-1]['xc'], rows[-1]['n'], rows[-1]['H1'], rows[-1]['H2']]
    light_hash = _light_prefix_hash(identity_rows)
    # The initial checkpoint consumed nothing: its light-prefix digest covers
    # its (empty) consumed span, while the final checkpoint's covers the
    # window it actually consumed.
    initial_checkpoint = _make_checkpoint(values, start, start, config_hash,
                                          _light_prefix_hash([]))
    final_checkpoint = _make_checkpoint(final_values, end, start, config_hash, light_hash)

    summary_provenance = {
        'model_version': MELATONIN_MODEL_VERSION,
        'api_version': MELATONIN_API_VERSION,
        'melatonin_config_hash': config_hash,
        'source_sha256': SOURCE_MANIFEST_SHA256,
        'source_upstream_commit': SOURCE_UPSTREAM_COMMIT,
        'initialization': initialization_label,
    }
    all_crossings = synthesis_crossings(rows, resolved) if crossings else []
    result = {
        'api_version': MELATONIN_API_VERSION,
        'model_version': MELATONIN_MODEL_VERSION,
        'status': 'partial' if failures else 'complete',
        'validity': {
            'status': 'partial' if failures else 'complete',
            'numerical_failures': failures,
            'guard_activations': guard_activations,
            'nonnegative_concentrations': not any(item['state'] in ('H1', 'H2') for item in failures),
            'state_domain_ok': not any(item['state'] == 'n' for item in failures),
            'finite_states': not any(item['reason'] == 'non-finite exposed state' for item in failures),
            'initialization': initialization_label,
        },
        'units': dict(MELATONIN_UNITS),
        'assumptions': assumptions,
        'provenance': {
            'model_version': MELATONIN_MODEL_VERSION,
            'api_version': MELATONIN_API_VERSION,
            'configuration': resolved,
            'melatonin_config_hash': config_hash,
            'source': {'upstream_commit': SOURCE_UPSTREAM_COMMIT,
                       'manifest_sha256': SOURCE_MANIFEST_SHA256, 'license': SOURCE_LICENSE,
                       'path': 'app/sleep_model/sources/circadian/models.py.txt (class Breslow13)'},
            'paper': {'citation': PAPER_CITATION, 'doi': '10.1177/0748730412468081',
                      'primary_source': 'https://pmc.ncbi.nlm.nih.gov/articles/PMC3733227/'},
            'deviations': list(DEVIATIONS),
            'initialization': initialization_label,
            'equation_source': 'sources/circadian/manifest.json',
        },
        'trace': rows,
        'initial_checkpoint': initial_checkpoint,
        'final_checkpoint': final_checkpoint,
        'summaries': interval_summaries(rows, summaries, start, end, {'config': resolved,
                                                                        'summary': summary_provenance},
                                       crossings=all_crossings),
        'crossings': all_crossings,
        'initialization': {'label': initialization_label,
                            'source': 'pinned numerical initial state' if values == list(DEFAULT_INITIAL_STATE)
                            else 'caller supplied or resumed state'},
    }
    return result
