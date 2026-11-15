"""Lazy compatibility bridge: the scoring adapter moved to planning.

The legacy ``score`` implementation moved to ``app.planning.adapter`` (module
decoupling contract step 3); this module keeps the historical import path
working and contains no implementation. Compatibility only: the result shape
and the behaviour are unchanged.
"""
__all__ = ['score']


def __getattr__(name):
    if name in __all__:
        from app.planning.adapter import score
        return score
    raise AttributeError('module %r has no attribute %r' % (__name__, name))


def __dir__():
    return sorted(set(globals()) | set(__all__))
