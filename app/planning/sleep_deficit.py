"""Report sleep deficits on a resolved schedule timeline."""

from datetime import timedelta

from app.contracts.time import dt, interval, merge, overlap, stamp
from app.sleep_model.api import forecast_sleep_opportunities


LOOKBACK = timedelta(hours=24)
TARGET_HOURS = 8
TARGET = timedelta(hours=TARGET_HOURS)


def check_sleep_deficit(actual_sleep, opportunities, *, cutoff_utc,
                        start_utc, end_utc, duties=(), flights=(), config=None):
    """Return issues for a UTC evaluation range using resolved records.

    Actual sleep records are visible as of cutoff. For an ongoing observation,
    pass the recorded portion ending at cutoff and any predicted continuation
    as part of its original sleep opportunity. Include sleep context from
    at least 24 hours before the evaluation start.
    """
    start, end, cutoff = dt(start_utc), dt(end_utc), dt(cutoff_utc)
    if end <= start:
        raise ValueError('evaluation end must follow start')

    actual = list(actual_sleep)
    if any(interval(record)[1] > cutoff for record in actual):
        raise ValueError('actual sleep extends past cutoff')

    predicted = []
    for record in forecast_sleep_opportunities(opportunities, config=config):
        left, right = interval(record)
        if right > cutoff:
            predicted.append(dict(record, start_utc=stamp(max(left, cutoff))))

    inflight = []
    for flight in flights:
        left, right = interval(flight)
        hours = (right - left) / timedelta(hours=1)
        if hours < 8 or any(overlap(interval(record), (left, right)) for record in actual):
            continue
        duration = timedelta(hours=1 + (hours - 8) * (0.5 if hours <= 12 else 0.25)) if hours <= 12 else timedelta(hours=3 + (hours - 12) * 0.25)
        middle = left + (right - left) / 2
        a, b = middle - duration / 2, middle + duration / 2
        if b > cutoff:
            inflight.append({'flight_id': flight['id'], 'start_utc': stamp(max(a, cutoff)), 'end_utc': stamp(b)})

    sleep = merge(interval(record) for record in actual + predicted + inflight)
    bounds = {start, end}
    for left, right in sleep:
        bounds.update(point for point in (left, right, left + LOOKBACK, right + LOOKBACK)
                      if start < point < end)

    # Between these boundaries the rolling total is linear. Its slope is
    # -1, 0 or +1 hours per hour, even when sleep records overlap.
    points = [
        (at, sum((min(right, at) - max(left, at - LOOKBACK)
                  for left, right in sleep if left < at and right > at - LOOKBACK),
                 timedelta()))
        for at in sorted(bounds)
    ]
    spans = []
    for (left, before), (right, after) in zip(points, points[1:]):
        if before >= TARGET and after >= TARGET:
            continue
        low_at, minimum = (left, before) if before <= after else (right, after)
        if before >= TARGET:
            left += before - TARGET
        if after >= TARGET:
            right -= after - TARGET

        # A shared boundary at exactly eight hours separates two deficits.
        if spans and spans[-1][1] == left and before < TARGET:
            left, _, previous_at, previous_minimum = spans.pop()
            if previous_minimum <= minimum:
                low_at, minimum = previous_at, previous_minimum
        spans.append((left, right, low_at, minimum))

    duty_intervals = [(duty['id'], interval(duty)) for duty in duties]
    issues = []
    for left, right, low_at, minimum in spans:
        evidence = (low_at - LOOKBACK, low_at)
        samples = []
        at = left
        while at < right:
            total = sum((min(b, at) - max(a, at - LOOKBACK)
                         for a, b in sleep if a < at and b > at - LOOKBACK), timedelta())
            samples.append({'at_utc': stamp(at), 'window_start_utc': stamp(at - LOOKBACK),
                            'deficit_hours': max(0, (TARGET - total) / timedelta(hours=1))})
            at += timedelta(hours=2)
        issues.append({
            'type': 'sleep_deficit',
            'start_utc': stamp(left),
            'end_utc': stamp(right),
            'lookback_hours': 24,
            'threshold_hours': TARGET_HOURS,
            'max_deficit_hours': (TARGET - minimum) / timedelta(hours=1),
            'samples': samples,
            'evidence': {
                'start_utc': stamp(evidence[0]),
                'end_utc': stamp(evidence[1]),
                'sleep_hours': minimum / timedelta(hours=1),
                'actual_sleep_ids': [r['id'] for r in actual if overlap(interval(r), evidence)],
                'sleep_opportunity_ids': [r['opportunity_id'] for r in predicted
                                          if overlap(interval(r), evidence)],
                'inflight_sleep_estimates': [r for r in inflight if overlap(interval(r), evidence)],
                'forecast': low_at > cutoff,
                'missing_sleep_policy': 'no_imputation',
            },
            'duty_ids': [identifier for identifier, duty in duty_intervals
                         if overlap(duty, (left, right))],
        })
    return {'issues': issues}
