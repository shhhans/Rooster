"""Versioned deterministic sleep-state simulation, not a clinical predictor.

``evaluate`` and ``evaluate_many`` are lazy compatibility bridges to
``app.planning.evaluation`` (module decoupling contract step 3): the scoring
orchestration is planning-owned, the physiology stays model-owned. Importing
this package never imports planning -- only attribute access does.
"""
from .contracts import VERSION

__all__ = ['evaluate', 'evaluate_many', 'VERSION']


def __getattr__(name):
    if name in ('evaluate', 'evaluate_many'):
        from .evaluation import evaluate, evaluate_many
        return {'evaluate': evaluate, 'evaluate_many': evaluate_many}[name]
    raise AttributeError('module %r has no attribute %r' % (__name__, name))


def __dir__():
    return sorted(set(globals()) | set(__all__))
