"""Build an observer-only snapshot from a private SQLite backup."""
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from app.planning.observation import observe_event, observe_track
from rooster_eval.scoring import instant


def stamp(value):
    return value.astimezone(timezone.utc).isoformat().replace('+00:00', 'Z')


def create_sandbox(source, directory, now):
    directory.mkdir(parents=True, exist_ok=False)
    uri = Path(source).resolve().as_uri() + '?mode=ro'
    with sqlite3.connect(uri, uri=True) as original, sqlite3.connect(directory / 'timeline.db') as copy:
        original.backup(copy)
    # The raw backup is private to the harness. Only this filtered view reaches tools.
    with sqlite3.connect(f'{(directory / "timeline.db").as_uri()}?mode=ro', uri=True) as conn:
        conn.row_factory = sqlite3.Row
        profile = json.loads(conn.execute('SELECT profile_json FROM sleep_profiles WHERE user_id=?', ('Eva',)).fetchone()[0])
        rows = conn.execute("""SELECT * FROM events WHERE user_id='Eva' AND retracted=0 AND superseded_by IS NULL
            AND (created_in_txn IS NULL OR created_in_txn IN (SELECT txn_id FROM proposal_batches WHERE status='accepted'))
            AND (retired_in_txn IS NULL OR retired_in_txn NOT IN (SELECT txn_id FROM proposal_batches WHERE status='accepted'))""").fetchall()
    facts, blocks = [], []
    for row in rows:
        payload = json.loads(row['payload'])
        base = {'id': row['event_id'], 'start_utc': row['occurred_at_utc'], 'timezone': row['timezone'],
                'source': row['source'], 'status': row['realization'], 'confirmed': row['confirmation'] == 'accepted',
                'accepted': row['confirmation'] == 'accepted'}
        if row['track'] == 'sleep':
            end = datetime.fromisoformat(payload['sleep_end'])
            if end.tzinfo is None:
                end = end.replace(tzinfo=ZoneInfo(row['timezone']))
            base.update(end_utc=stamp(end), role=payload.get('role', 'main'))
            if row['realization'] == 'actual':
                if end <= instant(now):
                    facts.append(dict(base, kind='sleep_actual'))
            else:
                blocks.append(dict(base, kind='sleep_plan'))
        elif row['track'] == 'schedule':
            kind = row['title']
            end = payload.get({'flight': 'arrival', 'duty': 'duty_end'}.get(kind, 'end_utc'))
            if end:
                facts.append(dict(base, kind=kind, end_utc=stamp(instant(end))))
    return {'now_utc': stamp(instant(now)), 'profile': profile, 'events': facts}, blocks


def execute(name, arguments, snapshot, blocks):
    if name == 'observe_profile':
        if arguments:
            raise ValueError('observe_profile takes no arguments')
        return snapshot['profile']
    if name == 'observe_track':
        return observe_track(snapshot, blocks, **arguments)
    if name == 'observe_event':
        return observe_event(snapshot, blocks, **arguments)
    raise ValueError('unsupported observer tool')
