"""Compatibility adapter for candidate scoring; no trace is sent to the LLM in
candidate summaries.

Moved from ``app.sleep_model.adapter`` (module decoupling contract step 3); the
historical import path is a lazy compatibility bridge. The scoring behaviour is
unchanged; the orchestration behind it is planning-owned.
"""
from app.contracts.time import dt
from app.sleep_model.contracts import VERSION

from .evaluation import evaluate


def score(snapshot, sleep_blocks=None, start_utc=None, end_utc=None, *, sleep_count=None):
    now = dt(snapshot['now_utc'])
    start = dt(start_utc) if start_utc else now
    end = dt(end_utc or snapshot['horizon_end_utc'])
    if end <= start:
        return {'utility': None, 'status': 'invalid_input', 'scorer_version': VERSION,
                'model_version': VERSION, 'error': 'empty evaluation window'}
    mode = 'retrospective' if start < now and end <= now and not sleep_blocks else 'forecast'
    result = evaluate(snapshot, sleep_blocks, start_utc=start.isoformat(), end_utc=end.isoformat(),
                      mode=mode, scenarios=False, sleep_count=sleep_count)
    metrics = result.get('physiological_metrics', {})
    return {'utility': result['utility'], 'status': result['status'],
            'model_version': result['scorer_version'], 'scorer_version': result['scorer_version'],
            'longest_wake_hours': metrics.get('longest_wake_hours'),
            'rolling_sleep_hours_24': metrics.get('rolling_actual_sleep_hours', {}).get('24'),
            'rolling_sleep_hours_48': metrics.get('rolling_actual_sleep_hours', {}).get('48'),
            'rolling_sleep_hours_72': metrics.get('rolling_actual_sleep_hours', {}).get('72'),
            'sleep_hours': metrics.get('sleep_hours'), 'duty_burden': metrics.get('duty_burden'),
            'fragmentation': metrics.get('fragmentation_interruptions'),
            'losses': result.get('losses'), 'publishability': result.get('publishability'),
            'provenance': result.get('provenance'), 'error': result.get('error')}
