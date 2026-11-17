"""Native-tool sleep planning against the shared incremental timeline."""
from copy import deepcopy
from datetime import datetime, timedelta
import json
import os
import urllib.error
import urllib.request
from zoneinfo import ZoneInfo
from app.contracts.time import dt, local_instant, stamp
from app.planning.incremental_agent import PROMPT
from app.planning.observation import TRACKS, observe_event, observe_track
from app.planning.reactive_review import compare, review
from app.planning.schedule_edit import edit_schedule
from app.planning.state import digest
MODEL = 'deepseek-v4.1-flash'
URL = 'https://ark.cn-beijing.volces.com/api/plan/v3/chat/completions'
REPAIR_PADDING = timedelta(hours=48)

def _schema(name, description, properties, required):
    return {'type': 'function', 'function': {'name': name, 'description': description, 'parameters': {'type': 'object', 'properties': properties, 'required': required, 'additionalProperties': False}}}
_tracks = {'type': 'array', 'items': {'type': 'string', 'enum': list(TRACKS)}, 'minItems': 1}
_zone = {'type': 'string', 'description': 'IANA 显示时区。'}
_fields = {'type': 'object', 'properties': {'kind': {'type': 'string', 'enum': ['sleep_plan']}, 'start': {'type': 'string', 'description': 'ISO 日期时间，无偏移时按事项时区解释。'}, 'end': {'type': 'string', 'description': 'ISO 日期时间，无偏移时按事项时区解释。'}, 'timezone': _zone, 'title': {'type': 'string'}, 'role': {'type': 'string', 'enum': ['main', 'nap']}, 'crew_rest_id': {'type': ['string', 'null'], 'description': '机组休息许可事项 ID，清除时传 null。'}, 'parent_event_ids': {'type': 'array', 'items': {'type': 'string'}, 'description': '所替代或拆分的历史睡眠事项 ID。'}}, 'additionalProperties': False}
TOOLS = [_schema('observe_profile', '读取用户睡眠档案。', {}, []), _schema('observe_track', '读取选定时间窗口中所选轨道的事项；返回完整边界、连续睡眠及重叠。时间同时按UTC和指定时区显示；未来没有已知占用的时间默认可自由支配。', {'tracks': _tracks, 'start': {'type': 'string'}, 'end': {'type': 'string'}, 'timezone': _zone}, ['tracks', 'start', 'end', 'timezone']), _schema('observe_event', '读取事项及其前后指定小时数的轨道事项、相邻间隔和重叠。', {'event_id': {'type': 'string'}, 'before_hours': {'type': 'number', 'minimum': 0}, 'after_hours': {'type': 'number', 'minimum': 0}, 'tracks': _tracks, 'timezone': _zone}, ['event_id', 'before_hours', 'after_hours', 'tracks', 'timezone']), _schema('edit_schedule', '批量创建、修改或删除睡眠建议，失败时整批不生效；返回修改和局部关系。不能修改用户已确认的安排或事实。', {'operations': {'type': 'array', 'minItems': 1, 'items': {'type': 'object', 'properties': {'op': {'type': 'string', 'enum': ['create', 'update', 'delete']}, 'event_id': {'type': 'string'}, 'fields': _fields}, 'required': ['op'], 'additionalProperties': False}}, 'display_timezone': _zone, 'context_hours': {'type': 'number', 'minimum': 0.1}}, ['operations', 'display_timezone'])]

def request(messages, tools=TOOLS, *, on_stream=None, reasoning_effort='high'):
    for max_tokens in (65536, 131072):
        message, finish = _request_once(messages, tools, max_tokens, on_stream=on_stream, reasoning_effort=reasoning_effort)
        if finish in ('stop', 'tool_calls'):
            return message
        if finish != 'length':
            raise ValueError('model response incomplete: ' + str(finish))
    raise ValueError('model response incomplete: length (65536 and 131072 token limits reached)')

def _request_once(messages, tools, max_tokens, *, on_stream=None, reasoning_effort='high'):
    key = os.environ.get('ARK_AGENT_API_KEY')
    if not key:
        raise ValueError('ARK_AGENT_API_KEY is required for pure_llm planning')
    payload = {'model': MODEL, 'messages': messages, 'tools': tools, 'temperature': 1.0, 'top_p': 0.95, 'max_tokens': max_tokens, 'stream': True, 'thinking': {'type': 'enabled'}, 'reasoning_effort': 'high'}
    if reasoning_effort == 'off':
        payload['thinking'] = {'type': 'disabled'}
        payload.pop('reasoning_effort')
    elif reasoning_effort in ('low', 'high', 'max'):
        payload['reasoning_effort'] = reasoning_effort
    else:
        raise ValueError('unsupported reasoning effort')
    req = urllib.request.Request(URL, data=json.dumps(payload, ensure_ascii=False).encode('utf-8'), headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'}, method='POST')
    stream_id = on_stream('start', {'max_tokens': max_tokens}) if on_stream else None
    finish = None
    try:
      with urllib.request.urlopen(req, timeout=240) as response:
        message = {'role': 'assistant', 'content': '', 'reasoning_content': ''}
        calls, finish, usage = ({}, None, None)
        buffer = b''
        def lines():
            nonlocal buffer
            while True:
                data = response.read1(65536)
                if not data:
                    if buffer:
                        yield buffer
                    return
                if on_stream:
                    on_stream('data', {'request_id': stream_id, 'data': data})
                buffer += data
                while b'\n' in buffer:
                    line, buffer = buffer.split(b'\n', 1)
                    yield line
        for line in lines():
            if not line.startswith(b'data:'):
                continue
            data = line[5:].strip()
            if data == b'[DONE]':
                break
            chunk = json.loads(data)
            usage = chunk.get('usage') or usage
            if chunk.get('error'):
                raise ValueError(str(chunk['error']))
            for choice in chunk.get('choices', []):
                finish = choice.get('finish_reason') or finish
                delta = choice.get('delta') or {}
                for key in ('content', 'reasoning_content'):
                    message[key] += delta.get(key) or ''
                for call in delta.get('tool_calls') or []:
                    entry = calls.setdefault(call['index'], {'id': '', 'type': 'function', 'function': {'name': '', 'arguments': ''}})
                    entry['id'] += call.get('id') or ''
                    for key in ('name', 'arguments'):
                        entry['function'][key] += (call.get('function') or {}).get(key) or ''
    except BaseException as exc:
        if on_stream:
            on_stream('end', {'request_id': stream_id, 'finish_reason': finish, 'error': f'{type(exc).__name__}: {exc}'})
        raise
    if on_stream:
        on_stream('end', {'request_id': stream_id, 'finish_reason': finish, 'error': None if finish in ('stop', 'tool_calls', 'length') else 'model response incomplete'})
    if calls:
        message['tool_calls'] = [calls[i] for i in sorted(calls)]
    return (message, finish)

def begin_run(snapshot, *, request, budget=None, run_id=None, client=None, decision_mode=None):
    blocks = deepcopy((snapshot.get('active_plan') or {}).get('sleep_blocks') or [])
    prompt = PROMPT + '\n面对较长的规划范围，先了解全局班表与既有安排，再结合行程按较小的时间段逐步规划和编辑，关注前后行程和睡眠的关系，必要时回头调整，直至完成任务。'
    prompt += '\n当前时刻：' + snapshot['now_utc'] + '。'
    user = '请根据我的班表和已有安排制定可执行的睡眠计划。请通过工具观察和编辑，完成后检查并说明结果。'
    if request.get('kind') == 'feedback':
        user = '请根据用户反馈调整睡眠计划：' + json.dumps(request, ensure_ascii=False)
    elif request.get('kind') == 'text':
        user = request['text']
    state = {'run_id': run_id, 'snapshot': deepcopy(snapshot), 'request': deepcopy(request), 'plan': {'sleep_blocks': blocks}, 'segments': [], 'step_count': 0, 'budget': deepcopy(budget or {}), 'usage': {'llm_calls': 0}, 'status': 'running', 'messages': [{'role': 'system', 'content': prompt}, {'role': 'user', 'content': user}]}
    if request.get('kind') == 'reactive':
        start = max(dt(request['start_utc']) - REPAIR_PADDING, dt(snapshot['now_utc']))
        end = min(dt(request['end_utc']) + REPAIR_PADDING, dt(snapshot['horizon_end_utc']))
        if start >= end:
            raise ValueError('reactive window does not intersect the future planning horizon')
        state['repair_window'] = [start.isoformat(), end.isoformat()]
        baseline = review(snapshot, blocks, state['repair_window'])
        state['issue_review'] = compare(baseline, baseline)
        state['messages'][1]['content'] = '按下列变更和 Issue 检查局部排程。仅在有可行改善时编辑，每批编辑会返回复检。允许保留生理赤字；未找到修复不能断言无解。尊重用户忽略的问题，证据变化时可以重新检查。' + json.dumps({'trigger': request, 'effective': baseline, 'feedback': snapshot.get('user_feedback', []), 'rejection_history': snapshot.get('rejection_history', []), 'previous_issue_states': snapshot.get('previous_issue_states', [])}, ensure_ascii=False)
        if request.get('trigger') != 'proposal_rejected' and (not baseline['issues']) and (not baseline.get('model_unavailable')):
            state['answer'] = '受影响窗口未检出需要修复的 Issue。'
    return state

def step(draft, snapshot=None, *, mode='step', limit=None, client=None, on_tool=None):
    result = _step(draft, snapshot, mode=mode, limit=limit, client=client, on_tool=on_tool)
    return result

def _step(draft, snapshot=None, *, mode='step', limit=None, client=None, on_tool=None):
    state = deepcopy(draft)
    if 'repair_window' in state and snapshot and any((snapshot.get(key) != state['snapshot'].get(key) for key in ('revision', 'active_plan'))):
        raise ValueError('reactive repair snapshot changed; restart on the current timeline')
    state['snapshot'] = deepcopy(snapshot or state['snapshot'])
    if 'repair_window' in state and state.get('answer'):
        state['status'] = 'complete'
        return _result(state, 'complete', [])
    max_steps = state['budget'].get('max_steps')
    if max_steps is not None and state['step_count'] >= max_steps:
        state['status'] = 'budget_exhausted'
        return _result(state, 'budget_exhausted', [])
    max_calls = state['budget'].get('max_llm_calls', 64)
    if max_calls is not None and state['usage']['llm_calls'] >= max_calls:
        state['status'] = 'budget_exhausted'
        return _result(state, 'budget_exhausted', [])
    message = (client or request)(deepcopy(state['messages']), TOOLS)
    state['usage']['llm_calls'] += 1
    state['step_count'] += 1
    calls = message.get('tool_calls') or []
    state['messages'].append(message)
    diffs = []
    for index, call in enumerate(calls):
        if on_tool:
            on_tool(state, index, call, None)
        name = call['function']['name']
        try:
            args = json.loads(call['function']['arguments'])
            result, changes = _execute(state, name, args)
            diffs.extend(changes)
        except (KeyError, TypeError, ValueError) as exc:
            result = {'error': type(exc).__name__ + ': ' + str(exc)}
        if on_tool:
            on_tool(state, index, call, result)
        state['messages'].append({'role': 'tool', 'tool_call_id': call['id'], 'name': name, 'content': json.dumps(result, ensure_ascii=False)})
    if not calls:
        state['answer'] = message.get('content')
        state['status'] = 'complete'
    return _result(state, 'step' if calls else 'complete', diffs)

def _execute(state, name, args):
    snapshot, blocks = (state['snapshot'], state['plan']['sleep_blocks'])
    aliases = {b['origin_event_id']: b['id'] for b in blocks if b.get('origin_event_id')}
    args = deepcopy(args)
    if name == 'observe_event' and args.get('event_id') in aliases:
        args['event_id'] = aliases[args['event_id']]
    if name == 'edit_schedule':
        for op in args.get('operations', []):
            if op.get('event_id') in aliases:
                op['event_id'] = aliases[op['event_id']]
            fields = op.get('fields') or {}
            if 'parent_event_ids' in fields:
                by_id = {block['id']: block for block in blocks}
                fields['parent_event_ids'] = [by_id[p].get('origin_event_id') or p if p in by_id else p for p in fields['parent_event_ids']]
    observed = dict(snapshot, events=[event for event in snapshot['events'] if event.get('kind') != 'sleep_plan'])
    if name == 'observe_profile':
        return ({'profile': snapshot['profile']}, [])
    if name == 'observe_track':
        result = observe_track(observed, blocks, **args)
        if snapshot.get('snapshot_notes'):
            result['snapshot_notes'] = snapshot['snapshot_notes']
        return (result, [])
    if name == 'observe_event':
        return (observe_event(observed, blocks, **args), [])
    if name != 'edit_schedule':
        raise ValueError('unknown tool')
    existing = {b['id']: b for b in blocks}
    for op in args.get('operations', []):
        before = existing.get(op.get('event_id'))
        if before and (before.get('accepted') and 'repair_window' not in state or dt(before['start_utc']) < dt(snapshot['now_utc'])):
            raise ValueError('已确认或已开始的安排不可由工具修改')
        if (op.get('fields') or {}).get('kind', 'sleep_plan') != 'sleep_plan':
            raise ValueError('当前只支持编辑睡眠建议')
        fields = op.get('fields') or {}
        timezone = fields.get('timezone') or (before or {}).get('timezone')
        for key in ('start', 'end'):
            if fields.get(key) and timezone:
                parsed = datetime.fromisoformat(fields[key].replace('Z', '+00:00'))
                if parsed.tzinfo is not None and (not fields[key].endswith('Z')) and (parsed.utcoffset() != parsed.astimezone(ZoneInfo(timezone)).utcoffset()):
                    raise ValueError('时间偏移与事项时区不一致，请使用当地时间或正确偏移')
    result = edit_schedule(observed, blocks, **args)
    updated = result.pop('blocks')
    intervals = [item for change in result['changes'] for item in (change.get('before'), change.get('after')) if item]
    for b in updated:
        if b['id'] not in existing or b != existing[b['id']]:
            if dt(b['start_utc']) < dt(snapshot['now_utc']):
                raise ValueError('睡眠建议必须位于未来')
            if b.get('role') not in ('main', 'nap'):
                raise ValueError('睡眠建议 role 必须为 main 或 nap')
    if 'repair_window' in state:
        start, end = map(dt, state['repair_window'])
        if intervals:
            start = max(dt(snapshot['now_utc']), min(start, *(dt(b['start_utc']) for b in intervals)))
            end = max(end, *(dt(b['end_utc']) for b in intervals))
        window = [stamp(start), stamp(end)]
        review_snapshot = dict(snapshot, horizon_end_utc=stamp(max(end, dt(snapshot['horizon_end_utc']))))
        baseline = review(review_snapshot, state['snapshot']['active_plan']['sleep_blocks'], window)
        candidate = review(review_snapshot, updated, window)
        report = compare(baseline, candidate)
        if any((issue['type'] != 'sleep_deficit' for issue in report['added'])):
            return ({**result, 'retained': False, 'issue_review': report}, [])
        state['issue_review'] = report
        state['repair_window'] = window
        result.update(retained=True, issue_review=report)
    state['plan']['sleep_blocks'] = updated
    diffs = []
    for change in result['changes']:
        event_id = change['id']
        after = next((b for b in updated if b['id'] == event_id), None)
        diffs.append({'event_id': event_id, 'op': {'create': 'add', 'update': 'move', 'delete': 'remove'}[change['op']], 'from_revision': None, 'to_revision': None, 'before': existing.get(event_id), 'after': after, 'target': 'sleep_block', 'affects_committed': False, 'reason': 'agent_schedule_edit'})
    return (result, diffs)

def _result(state, status, diffs):
    summary = {}
    if 'issue_review' in state:
        report = state['issue_review']
        summary['issue_review'] = report
        if status == 'complete':
            candidate = report['candidate']
            if candidate['feasible'] is True and (not candidate['issues']):
                summary['repair_outcome'] = 'repaired'
            elif candidate['feasible'] is True and (report['resolved_in_candidate'] or (report['physiological_improvement'] or 0) > 0):
                summary['repair_outcome'] = 'partial_improvement'
            else:
                summary['repair_outcome'] = 'no_feasible_repair_found'
    return {'step_index': state['step_count'], 'step_status': status, 'event_diffs': diffs, 'constraint_summary': summary, 'score_summary': {'physiological_improvement': state['issue_review']['physiological_improvement']} if 'issue_review' in state else {}, 'needs': {'clarification': None, 'confirmations': None}, 'draft_digest': digest(state), 'run_state': state, 'decision_mode': 'pure_llm'}

def resume_run(draft, *, input=None, budget_extension=None, snapshot=None, client=None):
    state = deepcopy(draft)
    if budget_extension:
        for key, value in budget_extension.items():
            state['budget'][key] = value
    if input:
        state['messages'].append({'role': 'user', 'content': json.dumps(input, ensure_ascii=False)})
    state['status'] = 'running'
    return state
