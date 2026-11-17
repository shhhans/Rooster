"""Structured activity for the scheduling agent."""
import json
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from app.domain.timeline.store import _iso_z
from app.services.reactive_timeline import _parse


def tool_details(call, state):
    name = call.get('function', {}).get('name', '')
    try:
        args = json.loads(call['function']['arguments'])
        events = {e.get('id'): e for e in state.get('snapshot', {}).get('events', [])}
        events.update({e.get('id'): e for e in (state.get('snapshot', {}).get('active_plan') or {}).get('sleep_blocks', [])})
        events.update({e.get('id'): e for e in state.get('plan', {}).get('sleep_blocks', [])})
        spans = []
        if args.get('start') and args.get('end'):
            spans.append((_parse(args['start']), _parse(args['end'])))
        if args.get('event_id') in events:
            item = events[args['event_id']]
            if item.get('start_utc') and item.get('end_utc'):
                spans.append((_parse(item['start_utc']) - timedelta(hours=float(args.get('before_hours', 0))),
                              _parse(item['end_utc']) + timedelta(hours=float(args.get('after_hours', 0)))))
        for operation in args.get('operations', []):
            item = events.get(operation.get('event_id'))
            if item and item.get('start_utc') and item.get('end_utc'):
                spans.append((_parse(item['start_utc']), _parse(item['end_utc'])))
            fields = operation.get('fields') or {}
            if fields.get('start') and fields.get('end'):
                zone = ZoneInfo(fields.get('timezone') or args.get('display_timezone') or 'UTC')
                def instant(value):
                    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
                    return parsed.replace(tzinfo=zone) if parsed.tzinfo is None else parsed
                spans.append((instant(fields['start']), instant(fields['end'])))
        return {'name': name, 'range': [_iso_z(min(s[0] for s in spans)), _iso_z(max(s[1] for s in spans))] if spans else [None, None],
                'event_id': args.get('event_id')}
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
        return {'name': name, 'range': [None, None], 'event_id': None}


def tool_summary(result):
    if not isinstance(result, dict):
        return {}
    summary = {key: result[key] for key in ('error', 'retained') if key in result}
    for key in ('events', 'changes', 'overlaps'):
        if isinstance(result.get(key), list):
            summary[key] = len(result[key])
    if not summary:
        summary['fields'] = list(result)
    return summary


def activity(conn, job, state):
    steps = {}
    def step(index):
        return steps.setdefault(index, {'index': index, 'requests': [], 'tools': []})

    requests = conn.execute(
        'SELECT r.*, (SELECT coalesce(sum(length(data)),0) FROM reactive_model_chunks '
        'WHERE request_id=r.request_id) AS bytes_received, '
        '(SELECT max(received_at) FROM reactive_model_chunks WHERE request_id=r.request_id) AS last_received_at '
        'FROM reactive_model_requests r WHERE job_id=? ORDER BY r.rowid', (job['job_id'],)
    )
    for row in requests:
        step(row['step_index'])['requests'].append(dict(row))
    recorded = set()
    for row in conn.execute(
        'SELECT * FROM reactive_tool_activity WHERE job_id=? ORDER BY step_index, tool_index', (job['job_id'],)
    ):
        recorded.add((row['step_index'], row['tool_index']))
        step(row['step_index'])['tools'].append({
            **json.loads(row['details']), 'started_at': row['started_at'], 'finished_at': row['finished_at'],
            'summary': json.loads(row['summary']) if row['summary'] else None,
        })
    messages = state.get('messages', [])
    results = {m.get('tool_call_id'): m for m in messages if m.get('role') == 'tool'}
    index = 0
    for message in messages:
        if message.get('role') != 'assistant':
            continue
        index += 1
        current = step(index)
        for i, call in enumerate(message.get('tool_calls') or []):
            if (index, i) not in recorded:
                result = results.get(call.get('id'))
                current['tools'].append({
                    **tool_details(call, state),
                    'summary': tool_summary(json.loads(result['content'])) if result else None,
                })
    return {'status': job['status'], 'error': job['error'], 'steps': [steps[i] for i in sorted(steps)]}
