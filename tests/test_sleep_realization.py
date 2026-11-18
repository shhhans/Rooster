from datetime import datetime, timedelta, timezone

from app.domain.timeline.models import ScheduleEvent
from app.domain.timeline.store import EventStore
from app.services.sleep_realization import realize_completed_sleep


def test_completed_accepted_agent_sleep_is_realized_once():
    store = EventStore(':memory:')
    now = datetime.now(timezone.utc).replace(microsecond=0)
    def add(identifier, start, end, source='agent', confirmation='accepted'):
        event = ScheduleEvent(event_id=identifier, track='sleep', user_id='Eva', dedupe_key=identifier,
            occurred_at_utc=start, local_time=start.replace(tzinfo=None).isoformat(), timezone='UTC',
            source=source, status='planned', title='sleep', confirmation=confirmation, realization='planned',
            payload={'sleep_start': start.replace(tzinfo=None).isoformat(), 'sleep_end': end.replace(tzinfo=None).isoformat()},
            created_at=now, updated_at=now)
        store.insert(event)
    add('finished', now-timedelta(hours=8), now)
    add('ongoing', now-timedelta(hours=1), now+timedelta(hours=1))
    add('baseline', now-timedelta(hours=8), now, source='baseline', confirmation='proposed')
    add('proposal', now-timedelta(hours=8), now, confirmation='proposed')
    assert realize_completed_sleep(store, now) == {'Eva'}
    assert store.get_by_id('finished').realization == 'actual'
    for identifier in ('ongoing', 'baseline', 'proposal'):
        assert store.get_by_id(identifier).realization == 'planned'
    revision = store.get_input_revision('Eva')
    assert not realize_completed_sleep(store, now)
    assert store.get_input_revision('Eva') == revision


def test_exact_actual_reconciles_plan_without_creating_duplicate():
    store = EventStore(':memory:')
    now = datetime.now(timezone.utc).replace(microsecond=0)
    start = now - timedelta(hours=8)
    def make(identifier, source, realization):
        return ScheduleEvent(event_id=identifier, track='sleep', user_id='Eva', dedupe_key=identifier,
            occurred_at_utc=start, local_time=start.replace(tzinfo=None).isoformat(), timezone='UTC',
            source=source, status='planned', title='sleep', confirmation='accepted', realization=realization,
            payload={'sleep_start': start.replace(tzinfo=None).isoformat(), 'sleep_end': now.replace(tzinfo=None).isoformat()},
            created_at=now, updated_at=now)
    store.insert(make('plan', 'agent', 'planned'))
    store.insert(make('actual', 'user', 'actual'))
    assert realize_completed_sleep(store, now) == {'Eva'}
    assert store.get_by_id('plan').retracted
    assert store.get_by_id('actual').realization == 'actual'
    assert store.connection.execute('SELECT parent_id FROM event_lineage WHERE child_id=?', ('actual',)).fetchone()[0] == 'plan'
