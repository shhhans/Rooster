"""Lazy compatibility bridge: the preference policy moved to planning.

The legacy ``evaluate_preferences``/``sleep_targets`` implementation moved to
``app.planning.preferences`` (module decoupling contract step 3); this module
keeps the historical import path working and contains no implementation.
Compatibility only: the losses, the fixed need/excess buckets and the error
strings are unchanged.
"""
__all__ = ['evaluate_preferences', 'sleep_targets']


def __getattr__(name):
    if name in __all__:
        from app.planning.preferences import evaluate_preferences, sleep_targets
        return {'evaluate_preferences': evaluate_preferences, 'sleep_targets': sleep_targets}[name]
    raise AttributeError('module %r has no attribute %r' % (__name__, name))


def __dir__():
    return sorted(set(globals()) | set(__all__))
