"""Lazy compatibility bridge: the scoring orchestration moved to planning.

The legacy ``evaluate``/``evaluate_many``/``check_constraints`` implementation
moved to ``app.planning.evaluation`` (module decoupling contract step 3); this
module keeps the historical import path working and contains no implementation.
Compatibility only: the formulas, defaults, output dictionaries, hashes and
error strings are unchanged; the biology is computed through the public model
API and the policy composition is planning-owned.
"""
__all__ = ['evaluate', 'evaluate_many', 'check_constraints']


def __getattr__(name):
    if name in __all__:
        from app.planning.evaluation import check_constraints, evaluate, evaluate_many
        return {'evaluate': evaluate, 'evaluate_many': evaluate_many,
                'check_constraints': check_constraints}[name]
    raise AttributeError('module %r has no attribute %r' % (__name__, name))


def __dir__():
    return sorted(set(globals()) | set(__all__))
