"""Versioned, JSON serializable sleep-model contracts and resolved defaults."""
from copy import deepcopy
import hashlib
import json
import math

VERSION = 'rooster-sleep-core-v2.1.0'
DEFAULTS = {
    'tau_w': 18.2, 'tau_s': 4.2, 'max_step_minutes': 15.0,
    'rtol': 1e-8, 'atol': 1e-10, 'inertia_hours': .5,
    'burden_weights': [.60, .25, .15],
    'rank_weights': [.65, .20, .10, .05],
    'threshold': .60, 'softness': .05,
    'light_factor': 1., 'latency_minutes': 15., 'early_minutes': 0.,
    'sleep_count_threshold': 4, 'sleep_count_coefficient': 2.,
}
SCENARIOS = {
    'nominal': {}, 'latency_low': {'latency_minutes': 5.},
    'sleep_disrupted': {'latency_minutes': 30., 'early_minutes': 30.},
    'light_dim': {'light_factor': .5}, 'light_bright': {'light_factor': 2.},
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def configuration(overrides=None):
    result = deepcopy(DEFAULTS)
    if overrides:
        unknown = set(overrides) - set(result)
        if unknown:
            raise ValueError('unknown model parameters: ' + ','.join(sorted(unknown)))
        result.update(overrides)
    for key in ('tau_w', 'tau_s', 'max_step_minutes', 'rtol', 'atol', 'inertia_hours', 'softness'):
        if not 0 < result[key] < float('inf'):
            raise ValueError('invalid positive parameter: ' + key)
    if any(not math.isfinite(result[k]) or result[k]<0 for k in ('latency_minutes','early_minutes','light_factor')):
        raise ValueError('negative input assumption')
    if (not math.isfinite(result['sleep_count_threshold']) or result['sleep_count_threshold']<0
            or int(result['sleep_count_threshold'])!=result['sleep_count_threshold']
            or not math.isfinite(result['sleep_count_coefficient']) or result['sleep_count_coefficient']<0):
        raise ValueError('invalid sleep count parameters')
    for key, size in [('burden_weights', 3), ('rank_weights', 4)]:
        weights = result[key]
        if len(weights) != size or any(not math.isfinite(w) or w < 0 for w in weights) or abs(sum(weights)-1) > 1e-10:
            raise ValueError('weights must be nonnegative and sum to one')
    return result


# Module decoupling contract step 2: physiological parameters are separated
# from product policy. Rank weights and sleep-count compensation belong to
# planning; the public model API resolves and validates physiology only.
POLICY_DEFAULT_KEYS = ('rank_weights', 'sleep_count_threshold', 'sleep_count_coefficient')
PHYSIOLOGICAL_DEFAULTS = {key: deepcopy(value) for key, value in DEFAULTS.items()
                          if key not in POLICY_DEFAULT_KEYS}


def physiological_configuration(overrides=None):
    """Resolve physiological parameters only; policy keys are explicit errors.

    Validation rules for the physiological keys are identical to
    ``configuration()``. Error strings are new-API strings (they are not the
    legacy ``unknown model parameters`` messages; the legacy path is
    unchanged).
    """
    result = deepcopy(PHYSIOLOGICAL_DEFAULTS)
    if overrides:
        unknown = set(overrides) - set(result)
        if unknown:
            raise ValueError('unknown physiological parameters: ' + ','.join(sorted(unknown)))
        result.update(overrides)
    for key in ('tau_w', 'tau_s', 'max_step_minutes', 'rtol', 'atol', 'inertia_hours', 'softness'):
        if not 0 < result[key] < float('inf'):
            raise ValueError('invalid positive parameter: ' + key)
    if any(not math.isfinite(result[k]) or result[k] < 0 for k in ('latency_minutes', 'early_minutes', 'light_factor')):
        raise ValueError('negative input assumption')
    weights = result['burden_weights']
    if len(weights) != 3 or any(not math.isfinite(w) or w < 0 for w in weights) or abs(sum(weights)-1) > 1e-10:
        raise ValueError('weights must be nonnegative and sum to one')
    return result


def compatibility_configuration(physiological):
    """Materialize the full legacy-shaped configuration for identity interchange.

    Checkpoint identity and hashes (``input_prefix_hash``, ``initialization_id``,
    ``configuration_hash``) embed the full legacy configuration. Pairing a
    resolved physiological configuration with the current policy defaults
    makes public-API states byte-identical to legacy ``evaluate`` states for
    the same physiological overrides, so checkpoints interoperate in both
    directions and planning need not duplicate biology. This never mutates
    ``DEFAULTS``.
    """
    if not isinstance(physiological, dict) or set(physiological) != set(PHYSIOLOGICAL_DEFAULTS):
        raise ValueError('compatibility configuration requires a resolved physiological configuration')
    return configuration(physiological)
