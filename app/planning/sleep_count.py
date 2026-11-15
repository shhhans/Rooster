"""Exact rolling-24h episode cost adopted from the count-only experiment.

The sleep-count penalty is a model-defect compensation that stays in the
planning layer: it is reported as a separate loss line, enabled by default and
ablatable through the explicit keyword switch of the evaluator and of the
planner/search. Moved from ``app.sleep_model.sleep_count`` (module decoupling
contract step 3); the historical import path is a lazy compatibility bridge.
"""
from datetime import timedelta

from app.contracts.time import interval, merge


def sleep_count_cost(records, start, end, *, threshold=4, coefficient=2.):
    if end <= start:
        raise ValueError('positive evaluation duration required')
    sleeps = merge(interval(e) for e in records)
    day = timedelta(hours=24)
    cuts = sorted({start, end} | {t for a, b in sleeps for t in (a, b, a + day, b + day) if start < t < end})
    exposure = above = 0.
    maximum = 0
    for a, b in zip(cuts, cuts[1:]):
        t = a + (b - a) / 2
        count = sum(x < t and y > t - day for x, y in sleeps)
        length = (b - a).total_seconds()
        maximum = max(maximum, count)
        exposure += length * max(0, count - threshold) ** 2
        above += length * (count > threshold)
    duration = (end - start).total_seconds()
    return dict(points=coefficient * exposure / duration, max_rolling_episodes=maximum,
                fraction_over_threshold=above / duration, coefficient=coefficient, threshold=threshold)
