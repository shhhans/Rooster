"""Persistent, isolated candidate batches for the event timeline."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from app.domain.timeline.models import ScheduleEvent
from app.domain.timeline.sleep_track import _resolve_pair
from app.domain.timeline.store import EventStore, _iso_z


class ProposalConflict(ValueError):
    """The proposal no longer applies to the effective timeline."""


def create_proposal(store: EventStore, user_id: str, retired_ids: list[str], additions: list[dict]) -> str:
    """Stage a complete sleep edit. Staging never mutates effective events."""
    txn_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    with store.transaction() as conn:
        revision = store.get_input_revision(user_id)
        targets = {}
        for event_id in set(retired_ids):
            event = store.get_by_id(event_id)
            if (not event or event.user_id != user_id or event.retracted or event.superseded_by
                    or event.retired_in_txn or event.realization == 'actual'
                    or event.track != 'sleep'):
                raise ProposalConflict('retirement target is not an editable effective sleep event')
            if event.created_in_txn:
                row = conn.execute('SELECT status FROM proposal_batches WHERE txn_id=?', (event.created_in_txn,)).fetchone()
                if not row or row['status'] != 'accepted':
                    raise ProposalConflict('retirement target is not effective')
            targets[event_id] = event
        conn.execute('INSERT INTO proposal_batches VALUES (?, ?, ?, ?, ?, NULL)',
                     (txn_id, user_id, 'pending', revision, _iso_z(now)))
        for event_id in targets:
            conn.execute('INSERT INTO proposal_retirements VALUES (?, ?)', (txn_id, event_id))
        for item in additions:
            parents = item.get('parent_event_ids', [])
            if any(parent not in targets for parent in parents):
                raise ValueError('candidate parents must be retired by the same batch')
            start, end = _resolve_pair(item['sleep_start'], item['sleep_end'], item['timezone'])
            event = ScheduleEvent(
                event_id=str(uuid.uuid4()), track='sleep', user_id=user_id,
                dedupe_key='proposal:' + str(uuid.uuid4()), occurred_at_utc=start,
                local_time=item['sleep_start'], timezone=item['timezone'], source='agent',
                status='planned', title='sleep',
                payload={'sleep_start': item['sleep_start'], 'sleep_end': item['sleep_end'],
                         'role': item.get('role', 'main'),
                         **({'crew_rest_id': item['crew_rest_id']} if item.get('crew_rest_id') else {})},
                confirmation='proposed', realization='planned', created_in_txn=txn_id,
                created_at=now, updated_at=now)
            store.insert(event, bump_revision=False)
            store.add_lineage(user_id, event.event_id, parents)
    return txn_id


def proposal_diff(store: EventStore, user_id: str, txn_id: str) -> dict:
    with store.lock:
        row = store.connection.execute('SELECT * FROM proposal_batches WHERE txn_id=? AND user_id=?', (txn_id, user_id)).fetchone()
        if row is None:
            raise KeyError(txn_id)
        removed = store.connection.execute('SELECT event_id FROM proposal_retirements WHERE txn_id=? ORDER BY event_id', (txn_id,)).fetchall()
        added = store.connection.execute('SELECT event_id FROM events WHERE created_in_txn=? ORDER BY occurred_at_utc, event_id', (txn_id,)).fetchall()
        return {'txn_id': txn_id, 'status': row['status'], 'input_revision': row['input_revision'],
                'before': [store.get_by_id(r['event_id']).to_dict() for r in removed],
                'after': [store.get_by_id(r['event_id']).to_dict() for r in added]}


def decide_proposal(store: EventStore, user_id: str, txn_id: str, accept: bool) -> dict:
    with store.transaction() as conn:
        diff = proposal_diff(store, user_id, txn_id)
        if diff['status'] != 'pending':
            raise ProposalConflict('proposal is no longer pending')
        if accept and store.get_input_revision(user_id) != diff['input_revision']:
            raise ProposalConflict('timeline changed after proposal was created')
        now = _iso_z(datetime.now(timezone.utc))
        if accept:
            for old in diff['before']:
                event = store.get_by_id(old['event_id'])
                if not event or event.retracted or event.superseded_by or event.retired_in_txn or event.realization == 'actual':
                    raise ProposalConflict('retirement target changed')
                conn.execute('UPDATE events SET retired_in_txn=?, updated_at=? WHERE event_id=?', (txn_id, now, event.event_id))
            conn.execute("UPDATE events SET confirmation='accepted', updated_at=? WHERE created_in_txn=?", (now, txn_id))
            affected = diff['before'] + diff['after']
            if affected:
                first = min(affected, key=lambda e: e['occurred_at_utc'])
                store._bump_input_revision(user_id, first['local_time'], origin='apply', affected_at_utc=first['occurred_at_utc'])
        else:
            conn.execute("UPDATE events SET confirmation='rejected', updated_at=? WHERE created_in_txn=?", (now, txn_id))
        conn.execute('UPDATE proposal_batches SET status=?, decided_at=? WHERE txn_id=?',
                     ('accepted' if accept else 'rejected', now, txn_id))
        return proposal_diff(store, user_id, txn_id)
