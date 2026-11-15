"""Lazy compatibility bridge: the sleep-count compensation moved to planning.

The legacy ``sleep_count_cost`` implementation moved to
``app.planning.sleep_count`` (module decoupling contract step 3); this module
keeps the historical import path working and contains no implementation.
Compatibility only: the rolling-24h episode cost and the reported fields are
unchanged.
"""
__all__ = ['sleep_count_cost']


def __getattr__(name):
    if name in __all__:
        from app.planning.sleep_count import sleep_count_cost
        return sleep_count_cost
    raise AttributeError('module %r has no attribute %r' % (__name__, name))


def __dir__():
    return sorted(set(globals()) | set(__all__))
