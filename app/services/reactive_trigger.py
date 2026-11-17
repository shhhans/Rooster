"""Queue a review for changed real inputs, when a sleep profile exists."""
from app.domain.timeline.sleep_track import _resolve_sleep_local
from app.services.reactive_timeline import _parse, queue_review, run_queued_review


def queue_input_review(store, background_tasks, events, trigger):
    events = [event for event in events if event is not None]
    if not events:
        return None
    user_id = events[0].user_id
    if not store.connection.execute('SELECT 1 FROM sleep_profiles WHERE user_id=?', (user_id,)).fetchone():
        return None
    spans = []
    for event in events:
        if event.user_id != user_id:
            continue
        if event.track == 'sleep':
            end = _resolve_sleep_local('sleep_end', event.payload['sleep_end'], event.timezone)
        else:
            key = {'flight': 'arrival', 'duty': 'duty_end'}.get(event.title, 'end_utc')
            value = event.payload.get(key)
            if not value:
                continue
            try:
                end = _parse(value)
            except (ValueError, TypeError, AttributeError):
                continue
        if end > event.occurred_at_utc:
            spans.append((event.occurred_at_utc, end))
    if not spans:
        return None
    job = queue_review(store, user_id, min(a for a, _ in spans), max(b for _, b in spans), trigger=trigger)
    background_tasks.add_task(run_queued_review, store, job)
    return job
