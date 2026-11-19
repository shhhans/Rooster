"""Bridge the persisted event timeline to the native pure-LLM planner."""
from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from app.domain.timeline.proposals import create_proposal
from app.domain.timeline.sleep_track import _resolve_sleep_local
from app.domain.timeline.store import _iso_z, _row_to_event
from app.planning import timeline_agent
from app.planning.reactive_review import review


def _parse(value):
    result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if result.tzinfo is None:
        raise ValueError('an explicit UTC offset is required')
    return result.astimezone(timezone.utc)


def snapshot(store, user_id, start, end, *, historical_review=True):
    with store.lock:
        profile_row = store.connection.execute('SELECT profile_json FROM sleep_profiles WHERE user_id=?', (user_id,)).fetchone()
        if not profile_row:
            raise ValueError('sleep profile is required')
        profile = json.loads(profile_row['profile_json'])
        bedtime = round(float(profile['preferred_sleep_start_hour']) * 60)
        profile.setdefault('habit_bed_local', f'{bedtime // 60:02d}:{bedtime % 60:02d}')
        wake = (bedtime + round(float(profile['sleep_need_hours']) * 60)) % 1440
        profile.setdefault('habit_wake_local', f'{wake // 60:02d}:{wake % 60:02d}')
        rows = store.connection.execute(
            "SELECT * FROM events WHERE user_id=? AND retracted=0 AND superseded_by IS NULL "
            "AND (created_in_txn IS NULL OR created_in_txn IN (SELECT txn_id FROM proposal_batches WHERE status='accepted')) "
            "AND (retired_in_txn IS NULL OR retired_in_txn NOT IN (SELECT txn_id FROM proposal_batches WHERE status='accepted'))", (user_id,)).fetchall()
        records = [_row_to_event(row) for row in rows]
        revision = store.get_input_revision(user_id)
    facts, blocks = [], []
    current = datetime.now(timezone.utc)
    now = min(current, start) if historical_review else current
    for event in records:
        base = {'id': event.event_id, 'timezone': event.timezone, 'source': event.source,
                'start_utc': _iso_z(event.occurred_at_utc)}
        if event.track == 'sleep':
            end_utc = _resolve_sleep_local('sleep_end', event.payload['sleep_end'], event.timezone)
            base['end_utc'] = _iso_z(end_utc)
            if event.realization == 'actual':
                if end_utc <= current:
                    facts.append({**base, 'kind': 'sleep_actual', 'protection': 'actual'})
            else:
                blocks.append({**base, 'kind': 'sleep_plan', 'role': event.payload.get('role', 'main'),
                               'origin_event_id': event.event_id, 'status': 'suggested',
                               'accepted': event.confirmation == 'accepted',
                               **({'crew_rest_id': event.payload['crew_rest_id']} if event.payload.get('crew_rest_id') else {})})
        elif event.track == 'schedule':
            kind = event.title if event.title in ('flight', 'duty', 'commute', 'preparation', 'crew_rest') else 'commitment'
            end_key = {'flight': 'arrival', 'duty': 'duty_end'}.get(kind, 'end_utc')
            end_value = event.payload.get(end_key)
            if end_value:
                facts.append({**base, 'kind': kind, 'end_utc': _iso_z(_parse(end_value)),
                              'protection': None if kind == 'crew_rest' else 'fixed_external',
                              **({'locked': True, 'title': event.title} if kind == 'commitment' else {}),
                              **({'authorized': event.payload.get('authorized') is True,
                                  'authorized_event_ids': event.payload.get('authorized_event_ids') or []} if kind == 'crew_rest' else {}),
                              **({'flight_id': event.payload.get('flight_id'),
                                  'origin_tz': event.payload.get('origin_tz'),
                                  'destination_tz': event.payload.get('destination_tz')} if event.title == 'flight' else {})})
    return {'now_utc': _iso_z(now), 'horizon_end_utc': _iso_z(max(end, now + timedelta(days=1))),
            'profile': profile, 'events': facts, 'revision': revision,
            'active_plan': {'sleep_blocks': blocks}}


def refresh_issues(store, user_id, start, end):
    try:
        result = _refresh_issues(store, user_id, start, end)
    except Exception as exc:
        with store.transaction() as conn:
            conn.execute('INSERT INTO reactive_detection_status VALUES (?, ?, ?) ON CONFLICT(user_id) DO UPDATE SET error=excluded.error, updated_at=excluded.updated_at', (user_id, f'{type(exc).__name__}: {exc}', _iso_z(datetime.now(timezone.utc))))
        raise
    error = result[1].get('model_unavailable')
    with store.transaction() as conn:
        conn.execute('INSERT INTO reactive_detection_status VALUES (?, ?, ?) ON CONFLICT(user_id) DO UPDATE SET error=excluded.error, updated_at=excluded.updated_at', (user_id, error, _iso_z(datetime.now(timezone.utc))))
    return result

def _refresh_issues(store, user_id, start, end):
    now_utc = datetime.now(timezone.utc)
    snap = snapshot(store, user_id, start, end, historical_review=False)
    if end <= now_utc:
        return snap, {'issues': [], 'feasible': True}
    start = max(start, now_utc)
    report = review(snap, snap['active_plan']['sleep_blocks'], [_iso_z(start), _iso_z(end)])
    now = _iso_z(datetime.now(timezone.utc))
    with store.transaction() as conn:
        if store.get_input_revision(user_id) != snap['revision']:
            raise ValueError('timeline changed during issue review')
        existing = conn.execute("SELECT issue_key, issue_json FROM reactive_issues WHERE user_id=? AND status='active' AND start_utc<? AND end_utc>?", (user_id, _iso_z(end), _iso_z(start))).fetchall()
        for row in existing:
            kind = json.loads(row['issue_json'])['type']
            # An unavailable physiological model is not evidence that its
            # previously reported issues have been resolved.
            evaluable = (kind not in ('sleep_onset_pressure_low', 'sleep_maintenance_pressure_low', 'sleep_deficit')
                         or (kind == 'sleep_deficit' and 'assumptions' in report)
                         or (kind != 'sleep_deficit' and 'sleep_pressure' in report))
            if evaluable:
                conn.execute("UPDATE reactive_issues SET status='resolved', updated_at=? WHERE user_id=? AND issue_key=?", (now, user_id, row['issue_key']))
        for issue in report['issues']:
            identity = json.dumps([issue['type'], issue.get('rule'), sorted(issue.get('event_ids', [])), issue['start_utc'], issue['end_utc']], separators=(',', ':'))
            key = hashlib.sha256(identity.encode()).hexdigest()
            conn.execute("INSERT INTO reactive_issues VALUES (?, ?, 'active', ?, ?, ?, ?) ON CONFLICT(user_id,issue_key) DO UPDATE SET status='active', issue_json=excluded.issue_json, updated_at=excluded.updated_at",
                         (key, user_id, issue['start_utc'], issue['end_utc'], json.dumps(issue, ensure_ascii=False), now))
    return snap, report


def create_job(store, user_id, start, end):
    snap, report = refresh_issues(store, user_id, start, end)
    job_id = str(uuid.uuid4())
    now = _iso_z(datetime.now(timezone.utc))
    with store.transaction() as conn:
        conn.execute('INSERT INTO reactive_jobs VALUES (?, ?, ?, ?, ?, ?, NULL, NULL, NULL, ?, ?)',
                     (job_id, user_id, 'queued', _iso_z(start), _iso_z(end), snap['revision'], now, now))
    return job_id, report


def queue_review(store, user_id, start, end, *, trigger='roster_updated', reasoning_effort='high'):
    """Persist a job before returning control to the client."""
    job_id = str(uuid.uuid4())
    now = _iso_z(datetime.now(timezone.utc))
    with store.transaction() as conn:
        conn.execute('INSERT INTO reactive_jobs VALUES (?, ?, ?, ?, ?, ?, NULL, ?, NULL, ?, ?)',
                     (job_id, user_id, 'reviewing', _iso_z(start), _iso_z(end), store.get_input_revision(user_id), json.dumps({'trigger': trigger, 'reasoning_effort': reasoning_effort}), now, now))
    return job_id


def run_queued_review(store, job_id):
    with store.lock:
        job = store.connection.execute('SELECT * FROM reactive_jobs WHERE job_id=?', (job_id,)).fetchone()
    if not job or job['status'] != 'reviewing':
        return
    user_id, start, end = job['user_id'], _parse(job['start_utc']), _parse(job['end_utc'])
    queued_state = json.loads(job['state_json']) if job['state_json'] else {}
    try:
        for attempt in range(3):
            try:
                snap, report = refresh_issues(store, user_id, start, end)
                if queued_state.get('trigger') == 'actual_sleep_updated':
                    current = datetime.now(timezone.utc)
                    _, _ = refresh_issues(store, user_id, current, max(current + timedelta(hours=48), end + timedelta(hours=48)))
                with store.transaction() as conn:
                    if store.get_input_revision(user_id) != snap['revision']:
                        raise ValueError('timeline changed after issue review')
                    if queued_state.get('trigger') == 'manual' and not report['issues']:
                        conn.execute("UPDATE reactive_jobs SET status='complete', state_json=?, updated_at=? WHERE job_id=?",
                                     (json.dumps({'trigger': 'manual', 'message': 'no issues in selected range'}), _iso_z(datetime.now(timezone.utc)), job_id))
                        return
                    conn.execute("UPDATE reactive_jobs SET status='queued', input_revision=?, updated_at=? WHERE job_id=?",
                                 (snap['revision'], _iso_z(datetime.now(timezone.utc)), job_id))
                break
            except ValueError as exc:
                if not str(exc).startswith('timeline changed') or attempt == 2:
                    raise
        run_job(store, job_id)
    except Exception as exc:
        with store.transaction() as conn:
            conn.execute("UPDATE reactive_jobs SET status='failed', error=?, updated_at=? WHERE job_id=?",
                         (str(exc), _iso_z(datetime.now(timezone.utc)), job_id))


def review_and_run(store, user_id, start, end):
    run_queued_review(store, queue_review(store, user_id, start, end))


def run_job(store, job_id, client=None):
    with store.lock:
        job = store.connection.execute('SELECT * FROM reactive_jobs WHERE job_id=?', (job_id,)).fetchone()
    if not job or job['status'] != 'queued':
        return
    def update(status, state=None, txn=None, error=None):
        with store.transaction() as conn:
            conn.execute('UPDATE reactive_jobs SET status=?, state_json=?, txn_id=?, error=?, updated_at=? WHERE job_id=?',
                         (status, json.dumps(state, ensure_ascii=False) if state else None, txn, error, _iso_z(datetime.now(timezone.utc)), job_id))
    def record_tool(state, index, call, result):
        from app.services.reactive_activity import tool_details, tool_summary

        now = datetime.now(timezone.utc).isoformat()
        key = (job_id, state['step_count'], index)
        with store.transaction() as conn:
            if result is None:
                conn.execute(
                    'INSERT INTO reactive_tool_activity VALUES (?, ?, ?, ?, NULL, ?, NULL)',
                    (*key, now, json.dumps(tool_details(call, state))),
                )
            else:
                conn.execute(
                    'UPDATE reactive_tool_activity SET finished_at=?, summary=? '
                    'WHERE job_id=? AND step_index=? AND tool_index=?',
                    (now, json.dumps(tool_summary(result), ensure_ascii=False), *key),
                )

    sequence = {}
    def record_stream(event, payload):
        now = _iso_z(datetime.now(timezone.utc))
        with store.transaction() as conn:
            if event == 'start':
                request_id = str(uuid.uuid4())
                conn.execute('INSERT INTO reactive_model_requests(request_id,job_id,step_index,max_tokens,started_at) VALUES (?,?,?,?,?)',
                             (request_id, job_id, state['step_count'] + 1, payload['max_tokens'], now))
                sequence[request_id] = 0
                return request_id
            request_id = payload['request_id']
            if event == 'data':
                seq = sequence[request_id]
                conn.execute('INSERT INTO reactive_model_chunks VALUES (?,?,?,?)', (request_id, seq, now, payload['data']))
                sequence[request_id] = seq + 1
            elif event == 'end':
                conn.execute('UPDATE reactive_model_requests SET finished_at=?, finish_reason=?, error=? WHERE request_id=?',
                             (now, payload['finish_reason'], payload['error'], request_id))
    try:
        start, end = _parse(job['start_utc']), _parse(job['end_utc'])
        if end + timedelta(hours=48) <= datetime.now(timezone.utc):
            update('complete')
            return
        snap = snapshot(store, job['user_id'], start - timedelta(hours=48), end + timedelta(hours=48), historical_review=False)
        if snap['revision'] != job['input_revision']:
            raise ValueError('timeline changed before agent started')
        queued_state = json.loads(job['state_json']) if job['state_json'] else {}
        request = {'kind': 'reactive', 'trigger': queued_state.get('trigger', 'roster_updated'), 'start_utc': _iso_z(start), 'end_utc': _iso_z(end), 'reasoning_effort': queued_state.get('reasoning_effort', 'high')}
        state = timeline_agent.begin_run(snap, request=request, run_id=job_id, budget={'max_llm_calls': 32})
        update('running', state)
        for _ in range(64):
            model_client = client or (lambda messages, tools: timeline_agent.request(messages, tools, on_stream=record_stream, reasoning_effort=request['reasoning_effort']))
            result = timeline_agent.step(state, client=model_client, on_tool=record_tool)
            state = result['run_state']
            update('running', state)
            if result['step_status'] in ('complete', 'budget_exhausted', 'failed'):
                break
        else:
            raise ValueError('agent step limit exceeded')
        if state['status'] != 'complete':
            raise ValueError('agent did not complete: ' + state['status'])
        original = {b['id']: b for b in snap['active_plan']['sleep_blocks']}
        final = {b['id']: b for b in state['plan']['sleep_blocks']}
        changed = {key for key in original if key not in final or final[key] != original[key]}
        new = [b for key, b in final.items() if key not in original or key in changed]
        additions = []
        for block in new:
            zone = ZoneInfo(block['timezone'])
            parents = set(block.get('parent_event_ids') or []) & changed
            if block['id'] in changed:
                parents.add(block['id'])
            additions.append({'sleep_start': _parse(block['start_utc']).astimezone(zone).replace(tzinfo=None).isoformat(),
                              'sleep_end': _parse(block['end_utc']).astimezone(zone).replace(tzinfo=None).isoformat(),
                              'timezone': zone.key, 'role': block.get('role', 'main'),
                              'crew_rest_id': block.get('crew_rest_id'), 'parent_event_ids': sorted(parents)})
        with store.transaction():
            if store.get_input_revision(job['user_id']) != job['input_revision']:
                raise ValueError('timeline changed before proposal staging')
            txn = create_proposal(store, job['user_id'], sorted(changed), additions) if changed or additions else None
            update('awaiting_review' if txn else 'complete', state, txn)
    except Exception as exc:
        update('failed', locals().get('state'), error=str(exc))
