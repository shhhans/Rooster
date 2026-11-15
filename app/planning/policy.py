"""Planning-owned product policy: rank weights and sleep-count compensation.

The physiological parameters belong to the model (``app.sleep_model.contracts``
resolves and validates them); the product policy belongs to planning. The legacy
full configuration is materialized as ``{**physiology, **policy}`` so that the
legacy configuration hash, the checkpoint identity and the frozen output
dictionaries stay byte-identical for the default legacy path.

Two surfaces exist, and they are deliberately different:

* ``planning_configuration`` resolves the **legacy** full configuration. It
  delegates to the model's existing legacy validator
  (``app.sleep_model.contracts.configuration``) so that the legacy error
  strings and the legacy validation order are reproduced exactly for every
  combination of invalid overrides -- unknown keys first, then the positive
  physiological parameters, then the non-negative assumptions, then the
  sleep-count parameters, then the weight parameters (burden weights before
  rank weights). A planning-side split of that resolver cannot reproduce this
  precedence (measured evidence:
  ``groups/planning/tasks/DECOUPLE-PLANNING-01/review-00/``), so the legacy
  path delegates instead. No physiological rule is copied here, no model file
  is modified, and the resolved policy part is split out and used by planning.
* ``policy_configuration`` is the **new planning API**: it resolves and
  validates the planning policy parameters alone, with planning-owned messages.

The policy values are planning's (``POLICY_DEFAULTS``); the model keeps a copy
of them only for the compatibility materialization of the legacy full
configuration. ``tests/test_planning_model_boundary.py`` guards the two copies
against drift, value and numeric type alike (the type is part of the legacy
configuration hash: ``json.dumps`` renders ``2`` and ``2.0`` differently).
"""
from copy import deepcopy
import math

from app.sleep_model.contracts import (PHYSIOLOGICAL_DEFAULTS, POLICY_DEFAULT_KEYS,
                                        configuration as legacy_configuration,
                                        physiological_configuration)

# The key partition is owned by the model (it excludes the policy keys from the
# physiological configuration); the policy *values* below are planning's.
POLICY_KEYS = tuple(POLICY_DEFAULT_KEYS)

# Frozen legacy policy values. The numeric types are part of the legacy
# configuration hash, so they must match the legacy defaults exactly: int
# threshold, float coefficient, float rank weights.
POLICY_DEFAULTS = {
    'rank_weights': [.65, .20, .10, .05],
    'sleep_count_threshold': 4,
    'sleep_count_coefficient': 2.,
}


def _validate_count_parameters(policy):
    """Sleep-count parameter rules of the NEW planning API surface.

    These rules are planning-owned (the sleep-count compensation is planning
    policy), but the message is deliberately a NEW-API string: the legacy
    message ``invalid sleep count parameters`` belongs to the legacy
    compatibility path, which delegates to the model's legacy resolver and
    therefore keeps its own strings untouched.
    """
    if (not math.isfinite(policy['sleep_count_threshold']) or policy['sleep_count_threshold'] < 0
            or int(policy['sleep_count_threshold']) != policy['sleep_count_threshold']
            or not math.isfinite(policy['sleep_count_coefficient'])
            or policy['sleep_count_coefficient'] < 0):
        raise ValueError('invalid sleep count policy parameters')


def _validate_rank_weights(policy):
    """Rank-weight rules of the NEW planning API surface (NEW-API string)."""
    weights = policy['rank_weights']
    if (len(weights) != 4 or any(not math.isfinite(w) or w < 0 for w in weights)
            or abs(sum(weights) - 1) > 1e-10):
        raise ValueError('policy weights must be nonnegative and sum to one')


def policy_configuration(overrides=None):
    """Resolve the planning policy parameters only (new planning API).

    Unknown keys raise ``unknown policy parameters: ...``; the value rules and
    their messages are the legacy ones for the sleep-count parameters and a
    planning-owned message for the rank weights. The public model resolver
    keeps rejecting policy keys with ``unknown physiological parameters: ...``.
    """
    result = deepcopy(POLICY_DEFAULTS)
    if overrides:
        unknown = set(overrides) - set(result)
        if unknown:
            raise ValueError('unknown policy parameters: ' + ','.join(sorted(unknown)))
        result.update(overrides)
    _validate_count_parameters(result)
    _validate_rank_weights(result)
    return result


def public_planning_configuration(overrides=None):
    """Resolve the full configuration through the **public** model resolver.

    This is the new planning API path: the physiological part is resolved by
    the public ``physiological_configuration`` (with its documented new
    strings, e.g. ``unknown physiological parameters: ...``) and the policy part
    by planning's own validation. It never reproduces the legacy full
    configuration, so it is not used by the legacy compatibility path.
    """
    overrides = dict(overrides or {})
    unknown = set(overrides) - set(PHYSIOLOGICAL_DEFAULTS) - set(POLICY_DEFAULTS)
    if unknown:
        raise ValueError('unknown planning parameters: ' + ','.join(sorted(unknown)))
    physio = physiological_configuration({key: value for key, value in overrides.items()
                                          if key in PHYSIOLOGICAL_DEFAULTS})
    policy = policy_configuration({key: value for key, value in overrides.items()
                                   if key in POLICY_DEFAULTS})
    return physio, policy, {**physio, **policy}


def planning_configuration(overrides=None):
    """Resolve the legacy-shaped full configuration as ``(physio, policy, full)``.

    ``full`` is byte-equal to the legacy ``configuration(overrides)`` for every
    valid input, so the legacy configuration hash, the checkpoint identity and
    the frozen provenance stay unchanged; ``physio`` is what the public model
    API is called with and ``policy`` is what planning composes with.
    """
    full = legacy_configuration(overrides)
    physio = {key: value for key, value in full.items() if key in PHYSIOLOGICAL_DEFAULTS}
    policy = {key: value for key, value in full.items() if key in POLICY_KEYS}
    return physio, policy, full
