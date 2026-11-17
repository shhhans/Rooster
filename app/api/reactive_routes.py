"""Issue, agent progress and proposal-review endpoints."""
import json
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel
from typing import Literal

from app.domain.timeline.proposals import ProposalConflict, decide_proposal, proposal_diff
from app.services.reactive_timeline import _parse, refresh_issues, queue_review, run_queued_review
from app.services.reactive_activity import activity, tool_details

router = APIRouter(prefix='/api/v1/reactive')


@router.get('/issues')
def issues(request: Request, user_id: str):
    store = request.app.state.timeline_service.store
    with store.lock:
        rows = store.connection.execute("SELECT issue_key, issue_json FROM reactive_issues WHERE user_id=? AND status='active' ORDER BY start_utc", (user_id,)).fetchall()
        return {'issues': [{'issue_id': row['issue_key'], **json.loads(row['issue_json'])} for row in rows]}


@router.get('/jobs')
def jobs(request: Request, user_id: str):
    store = request.app.state.timeline_service.store
    with store.lock:
        rows = store.connection.execute('SELECT * FROM reactive_jobs WHERE user_id=? ORDER BY created_at DESC, rowid DESC LIMIT 20', (user_id,)).fetchall()
        result = []
        for row in rows:
            state = json.loads(row['state_json']) if row['state_json'] else {}
            calls = []
            for message in state.get('messages', []):
                for call in message.get('tool_calls') or []:
                    calls.append(tool_details(call, state))
            stale = False
            if row['status'] == 'awaiting_review' and row['txn_id']:
                diff = proposal_diff(store, user_id, row['txn_id'])
                stale = diff['input_revision'] != store.get_input_revision(user_id) or any(
                    _parse(event['occurred_at_utc']) <= datetime.now(timezone.utc) for event in diff['before'] + diff['after'])
            result.append({'job_id': row['job_id'], 'status': row['status'], 'start_utc': row['start_utc'],
                           'end_utc': row['end_utc'], 'txn_id': row['txn_id'], 'error': row['error'],
                           'stale': stale, 'tool_calls': calls[-3:]})
        return {'jobs': result}


@router.get('/proposals/{txn_id}')
def proposal(txn_id: str, request: Request, user_id: str):
    try:
        return proposal_diff(request.app.state.timeline_service.store, user_id, txn_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail='proposal not found') from exc


@router.get('/jobs/{job_id}/model-requests')
def model_requests(job_id: str, request: Request, user_id: str):
    store = request.app.state.timeline_service.store
    with store.lock:
        if not store.connection.execute('SELECT 1 FROM reactive_jobs WHERE job_id=? AND user_id=?', (job_id, user_id)).fetchone():
            raise HTTPException(status_code=404, detail='job not found')
        rows = store.connection.execute('SELECT request_id,step_index,max_tokens,started_at,finished_at,finish_reason,error FROM reactive_model_requests WHERE job_id=? ORDER BY rowid', (job_id,)).fetchall()
        return {'requests': [dict(row) for row in rows]}


@router.get('/jobs/{job_id}/model-requests/{request_id}/stream')
def model_stream(job_id: str, request_id: str, request: Request, user_id: str):
    store = request.app.state.timeline_service.store
    with store.lock:
        found = store.connection.execute('SELECT 1 FROM reactive_model_requests r JOIN reactive_jobs j ON j.job_id=r.job_id WHERE r.request_id=? AND r.job_id=? AND j.user_id=?', (request_id, job_id, user_id)).fetchone()
        if not found:
            raise HTTPException(status_code=404, detail='model request not found')
        chunks = store.connection.execute('SELECT data FROM reactive_model_chunks WHERE request_id=? ORDER BY seq', (request_id,)).fetchall()
        return Response(content=b''.join(row['data'] for row in chunks), media_type='application/octet-stream')


class Decision(BaseModel):
    user_id: str
    accept: bool


class RetryRequest(BaseModel):
    user_id: str


class StartRequest(BaseModel):
    user_id: str
    start_utc: datetime
    end_utc: datetime
    reasoning_effort: Literal['off', 'low', 'high', 'max'] = 'high'


@router.post('/jobs')
def start_job(body: StartRequest, request: Request, background_tasks: BackgroundTasks):
    store = request.app.state.timeline_service.store
    if not body.user_id.strip() or body.start_utc.tzinfo is None or body.end_utc.tzinfo is None or body.end_utc <= body.start_utc:
        raise HTTPException(status_code=400, detail='valid user and timezone-aware time range required')
    with store.transaction() as conn:
        if not conn.execute('SELECT 1 FROM sleep_profiles WHERE user_id=?', (body.user_id,)).fetchone():
            raise HTTPException(status_code=400, detail='请先填写睡眠作息')
        running = conn.execute("SELECT job_id FROM reactive_jobs WHERE user_id=? AND status IN ('reviewing','queued','running') LIMIT 1", (body.user_id,)).fetchone()
        if running:
            raise HTTPException(status_code=409, detail='已有排程任务正在运行，请等待完成')
        job_id = queue_review(store, body.user_id, body.start_utc, body.end_utc, trigger='manual', reasoning_effort=body.reasoning_effort)
    background_tasks.add_task(run_queued_review, store, job_id)
    return {'status': 'queued', 'job_id': job_id}


@router.post('/jobs/{job_id}/retry')
def retry_job(job_id: str, body: RetryRequest, request: Request, background_tasks: BackgroundTasks):
    store = request.app.state.timeline_service.store
    with store.lock:
        job = store.connection.execute('SELECT * FROM reactive_jobs WHERE job_id=? AND user_id=?', (job_id, body.user_id)).fetchone()
    if not job:
        raise HTTPException(status_code=404, detail='job not found')
    if job['status'] not in ('failed', 'rejected'):
        raise HTTPException(status_code=409, detail='only failed or rejected jobs can be retried')
    state = json.loads(job['state_json']) if job['state_json'] else {}
    trigger = state.get('trigger') or (state.get('request') or {}).get('trigger') or 'roster_updated'
    effort = state.get('reasoning_effort') or (state.get('request') or {}).get('reasoning_effort', 'high')
    new_job = queue_review(store, body.user_id, _parse(job['start_utc']), _parse(job['end_utc']), trigger=trigger, reasoning_effort=effort)
    background_tasks.add_task(run_queued_review, store, new_job)
    return {'status': 'queued', 'job_id': new_job}


@router.post('/proposals/{txn_id}/decision')
def decide(txn_id: str, body: Decision, request: Request):
    store = request.app.state.timeline_service.store
    try:
        with store.transaction() as conn:
            job = conn.execute('SELECT * FROM reactive_jobs WHERE txn_id=? AND user_id=?', (txn_id, body.user_id)).fetchone()
            if not job:
                raise KeyError(txn_id)
            if body.accept:
                diff = proposal_diff(store, body.user_id, txn_id)
                now = datetime.now(timezone.utc)
                if any(_parse(event['occurred_at_utc']) <= now for event in diff['before'] + diff['after']):
                    raise ProposalConflict('proposal contains a sleep event that has already started')
            result = decide_proposal(store, body.user_id, txn_id, body.accept)
            if body.accept:
                affected = diff['before'] + diff['after']
                starts = [_parse(job['start_utc'])] + [_parse(event['occurred_at_utc']) for event in affected]
                from app.domain.timeline.sleep_track import _resolve_sleep_local
                ends = [_parse(job['end_utc'])] + [_resolve_sleep_local('sleep_end', event['payload']['sleep_end'], event['timezone']) for event in affected]
                refresh_issues(store, body.user_id, min(starts) - timedelta(hours=48), max(ends) + timedelta(hours=48))
            conn.execute('UPDATE reactive_jobs SET status=?, updated_at=? WHERE job_id=?',
                         ('accepted' if body.accept else 'rejected', datetime.now(timezone.utc).isoformat(), job['job_id']))
            return result
    except KeyError as exc:
        raise HTTPException(status_code=404, detail='proposal not found') from exc
    except ProposalConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


def _job_state(store, job_id, user_id):
    job = store.connection.execute(
        'SELECT * FROM reactive_jobs WHERE job_id=? AND user_id=?', (job_id, user_id)
    ).fetchone()
    if not job:
        raise HTTPException(status_code=404, detail='job not found')
    return job, json.loads(job['state_json'] or '{}')


@router.get('/jobs/{job_id}/activity')
def job_activity(job_id: str, request: Request, user_id: str):
    store = request.app.state.timeline_service.store
    with store.lock:
        job, state = _job_state(store, job_id, user_id)
        return activity(store.connection, job, state)


@router.get('/jobs/{job_id}/messages')
def job_messages(job_id: str, request: Request, user_id: str):
    store = request.app.state.timeline_service.store
    with store.lock:
        _, state = _job_state(store, job_id, user_id)
        return {'messages': state.get('messages', [])}
