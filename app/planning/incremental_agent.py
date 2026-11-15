"""Agent-owned tool decisions; constraints and physiological evidence stay in code.

The public incremental API is unchanged. This module stores tool observations and
an unevaluated/evaluated trial in the existing JSON draft, never in fact events.
Each step is one decision plus one tool call. There is no score-ascent rule.
"""
from copy import deepcopy
from datetime import timedelta
from app.contracts.time import dt, overlap, stamp
from .state import digest
from .provider import SchemaError
from .safety import inspect_output
from .incremental_proposals import fact_view, block_protected, commitment_protected_block_ids, event_protected, ongoing_session_spans, hard_validate, constraint_report, summarize_constraints, run_evaluation, habitual_proposals, move_proposals, search_seed_proposals, candidate_blocks
VERSION = 'incremental-react-4'
ROLE_VALUES = ('main', 'nap')
TOOL_CONTRACTS = {'read': {'description': '读取指定范围的事实、睡眠安排和约束摘要。', 'required': ['start_utc', 'end_utc']}, 'search': {'description': '在不超过48小时的局部范围查找候选，结果不改变草稿。', 'required': ['start_utc', 'end_utc'], 'optional': ['strategy'], 'enums': {'strategy': ['habitual', 'moves', 'beam']}}, 'edit': {'description': '暂存一组新增、移动或删除睡眠块的编辑；新增ID由系统分配。时间使用UTC，role为main或nap，新增时默认为main，移动时沿用原值。', 'required': ['edits']}, 'evaluate': {'description': '在所选范围比较当前方案与试排；须同时提供起止时间或同时省略。评价从真实now连续模拟至结束，只计分所选范围。提交或保留编辑前须覆盖相关范围。', 'required': [], 'optional': ['start_utc', 'end_utc']}, 'retain': {'description': '保留已评价的试排并更新正式时间线，可基于用户意愿接受较低的总分。', 'required': []}, 'revert': {'description': '撤回当前试排。', 'required': []}, 'submit': {'description': '提交已读取且当前版本已评价的范围；planned、preserved表示已有安排，no_window用于确无空闲窗口。', 'required': ['start_utc', 'end_utc', 'disposition'], 'enums': {'disposition': ['planned', 'preserved', 'no_window']}}, 'clarify': {'description': '在继续所需信息缺失时提出具体问题。', 'required': ['question']}, 'record_event': {'description': '记录用户明确说明的一次未来占用；source_reference引用用户消息。', 'required': ['start_utc', 'end_utc', 'timezone', 'title', 'source_reference']}}
TOOLS = tuple(TOOL_CONTRACTS)

def _tool_metadata():
    result = {}
    for name, contract in TOOL_CONTRACTS.items():
        fields = contract['required'] + contract.get('optional', [])
        properties = {field: {'type': 'string'} for field in fields}
        for field, choices in contract.get('enums', {}).items():
            properties[field]['enum'] = choices
        if name == 'edit':
            after = {'type': 'object', 'properties': {'start_utc': {'type': 'string'}, 'end_utc': {'type': 'string'}, 'timezone': {'type': 'string'}, 'role': {'type': 'string', 'enum': list(ROLE_VALUES)}}, 'required': ['start_utc', 'end_utc'], 'additionalProperties': False}
            properties['edits'] = {'type': 'array', 'minItems': 1, 'maxItems': 8, 'items': {'oneOf': [{'type': 'object', 'properties': {'op': {'const': 'add'}, 'after': after}, 'required': ['op', 'after'], 'additionalProperties': False}, {'type': 'object', 'properties': {'op': {'const': 'move'}, 'event_id': {'type': 'string'}, 'after': after}, 'required': ['op', 'event_id', 'after'], 'additionalProperties': False}, {'type': 'object', 'properties': {'op': {'const': 'remove'}, 'event_id': {'type': 'string'}}, 'required': ['op', 'event_id'], 'additionalProperties': False}]}}
        result[name] = {'description': contract['description'], 'parameters': {'type': 'object', 'properties': properties, 'required': contract['required'], 'additionalProperties': False}}
    return result
MAX_LOCAL_HOURS = 48
MAX_EDITS = 8
PROMPT = '你是 Rooster 的专业行程与睡眠规划助手，熟悉睡眠医学、生理学、健康科学与航空疲劳管理的相关知识及实践。你的服务对象包括面临早班、夜班、长航段、跨时区旅行或不规律执勤的空勤人员。\n\n你的任务是根据用户的实际执勤安排、个人事务、睡眠习惯和偏好，制定高质量、可执行的睡眠计划。面对睡眠压力较高或作息紊乱的情况，你需要综合考虑睡眠机会、连续清醒时间、昼夜节律、近期睡眠情况和后续任务，安排主睡眠、必要的补觉及勤前小睡。\n\n一个好的计划应当符合用户的真实行程和个人意愿，能够解释关键安排的原因，并在恢复睡眠、保持合理作息和满足后续执勤要求之间作出权衡。当信息不足时，明确说明哪些判断依赖假设，并随新的班表、实际睡眠记录或用户反馈调整计划。\n\n制定计划时，请设身处地地从用户的角度审视每一项安排。结合执勤、通勤、所在地、个人事务、睡眠环境和必要的准备时间，判断用户是否有机会按计划行动。不要因为某个时间段在日历上空闲，就假定用户一定能够入睡或完成休息。\n\n当你认为某项睡眠安排难以切实执行时，应调整时间、时长或安排方式，并说明取舍。如果现有条件不足以支持可行方案，应指出具体限制，并在必要时向用户询问。\n\n当某种睡眠安排具有明确的恢复目标，但用户因习惯、临时环境或其他实际原因可能难以实施时，可以结合情况向用户提出咖啡因、褪黑素等辅助调节手段作为讨论选项。应说明使用目的、与计划睡眠及后续执勤的关系，并考虑用户意愿、个人情况和适用的航空医学及运营要求。褪黑素的具体使用方案应结合航空医师意见和运营方规定。\n\n尊重用户已表达的偏好和反馈，保护已发生的事实和固定安排。区分已知事实、计划与推断，不编造实际睡眠或提供适航保证。'

def _state(draft):
    state = draft.get('agent_state')
    if state is None:
        state = {'version': VERSION, 'trial': None, 'observation': None, 'reads': [], 'submissions': [], 'last_evaluation': None, 'planning_issues': [], 'tool_history': []}
        draft['agent_state'] = state
        for segment in draft['segments']:
            segment.update(status='pending', decision=None, reason=None)
        if draft['pending_confirmations']:
            draft['pending_confirmations'] = []
            draft['notes'].append('legacy selector confirmations invalidated; Agent must derive a newly evaluated diff')
            draft['status'] = 'pending'
        if draft['status'] in ('complete', 'failed'):
            draft['status'] = 'pending'
        draft['provenance']['decision'].update(prompt_version=VERSION, workflow_version=VERSION)
    if not state.get('last_comparison'):
        from .incremental import _current_score_entry
        cached = _current_score_entry(draft)
        if cached and cached.get('status') == 'complete':
            window = {'start_utc': draft['snapshot']['now_utc'], 'end_utc': draft['snapshot']['horizon_end_utc']}
            state['last_comparison'] = {'range': window, 'facts_version': draft['facts_digest'], 'plan_version': draft['plan']['edit_sequence'], 'baseline': deepcopy(cached), 'trial': deepcopy(cached)}
            state['last_evaluated_plan_version'] = draft['plan']['edit_sequence']
            state['last_evaluation'] = deepcopy(draft.get('model_observation') or {})
    if state.get('version') in ('incremental-react-1', 'incremental-react-2', 'incremental-react-3'):
        state.update(version=VERSION, submissions=[], planning_issues=[], reads=[], trial=None, last_evaluation=None, last_comparison=None, last_evaluated_plan_version=None)
        state['retained_edit_ranges'] = []
        state.pop('confirmation_trial', None)
        if draft.get('pending_confirmations'):
            draft['pending_confirmations'] = []
            draft['notes'].append('legacy pending confirmations invalidated by Agent evaluation-range upgrade')
        if draft.get('status') == 'needs_confirmation':
            draft['status'] = 'pending'
        for segment in draft['segments']:
            segment.update(status='pending', decision=None, reason=None)
        if draft['status'] == 'complete':
            draft['status'] = 'pending'
        draft['notes'].append('prior Agent completion requires evidence review under ' + VERSION)
        draft['provenance']['decision'].update(prompt_version=VERSION, workflow_version=VERSION)
    if state.get('version') != VERSION:
        raise ValueError('unsupported Agent tool draft version')
    return state

def _range(draft, arguments, *, bounded=False):
    if set(arguments) - {'start_utc', 'end_utc', 'strategy', 'disposition'}:
        raise ValueError('unknown range argument')
    a, b = (dt(arguments['start_utc']), dt(arguments['end_utc']))
    if not dt(draft['snapshot']['now_utc']) <= a < b <= dt(draft['snapshot']['horizon_end_utc']):
        raise ValueError('range must lie within current future horizon')
    if bounded and b - a > timedelta(hours=MAX_LOCAL_HOURS):
        raise ValueError('local search range exceeds 48-hour tool cost limit')
    return {'start_utc': stamp(a), 'end_utc': stamp(b)}

def _in_range(item, window):
    return item.get('start_utc') is not None and item.get('end_utc') is not None and overlap((dt(item['start_utc']), dt(item['end_utc'])), (dt(window['start_utc']), dt(window['end_utc'])))

def _default_window(draft):
    pending = [s for s in draft['segments'] if s['status'] != 'processed']
    if pending:
        return {k: pending[0][k] for k in ('start_utc', 'end_utc')}
    return {k: draft['snapshot'][k] for k in ('now_utc', 'horizon_end_utc')}

def _context(draft, state):
    from .incremental import _score_summary
    return {'now_utc': draft['snapshot']['now_utc'], 'horizon_end_utc': draft['snapshot']['horizon_end_utc'], 'profile': deepcopy(draft['snapshot']['profile']), 'tool_contracts': _tool_metadata(), 'tool_response_schema': {'type': 'object', 'properties': {'tool': {'type': 'string'}, 'arguments': {'type': 'object'}, 'explanation': {'type': 'string'}}, 'required': ['tool', 'arguments', 'explanation'], 'additionalProperties': False}, 'request': deepcopy(draft['request']), 'feedback_context': deepcopy(draft['snapshot'].get('feedback_context')), 'user_feedback': deepcopy(draft['snapshot'].get('user_feedback', [])), 'clarification_answers': deepcopy(draft['clarification_answers']), 'user_messages': _user_messages(draft), 'facts': deepcopy(draft['snapshot']['events']), 'plan_blocks': deepcopy(draft['plan']['sleep_blocks']), 'current_evaluation': deepcopy(state['last_evaluation'] or draft.get('model_observation') or _score_summary(draft)), 'observation': deepcopy(state['observation']), 'work_progress': {'read_ranges': deepcopy(state.get('reads', [])), 'submitted_ranges': deepcopy(state.get('submissions', [])), 'retained_edit_ranges': deepcopy(state.get('retained_edit_ranges', [])), 'candidate_attempts': deepcopy(state.get('candidate_attempts', [])[-12:]), 'recent_tool_history': deepcopy(state.get('tool_history', [])[-8:])}, 'unresolved_tool_errors': deepcopy(state.get('tool_errors', [])), 'trial': None if state['trial'] is None else {'edits': deepcopy(state['trial']['edits']), 'evaluation': deepcopy(state['trial'].get('evaluation'))}, 'suggested_read': _default_window(draft), 'scoring_knowledge': {'utility': '100 * (1 - weighted product-policy losses.total); not a health probability', 'losses': {'physiological': 'public sleep model loss; with duties measured on duty segments', 'need': 'sleep-need policy cost', 'preference': 'expressed preference cost', 'excess': 'excess sleep policy cost', 'sleep_count': 'explicit planning compensation'}, 'evaluation_window': 'agent-selected [start_utc,end_utc] for incumbent and trial; physiology continuously simulated from actual now to end', 'limits': 'finite recorded horizon; <=14d visible history, 72h fallback; future missing blocks mean awake; scores do not show effect extinction', 'retention': 'Agent decides after evaluation; no strict utility increase required'}}

def _user_messages(draft):
    messages = [{'reference': 'answer:' + a['question_id'], 'text': a['text']} for a in draft.get('clarification_answers', [])]
    if draft['request'].get('reason'):
        context = draft['snapshot'].get('feedback_context') or {}
        messages.append({'reference': 'feedback:' + (context.get('current_feedback_id') or draft['run_id']), 'text': draft['request']['reason']})
    messages.extend(({'reference': 'feedback:' + f['feedback_id'], 'text': f['reason']} for f in draft['snapshot'].get('user_feedback', []) if f.get('reason') and (not any((m['reference'] == 'feedback:' + f['feedback_id'] for m in messages)))))
    return messages

def _validate(value, snapshot):
    if not isinstance(value, dict) or set(value) != {'tool', 'arguments', 'explanation'}:
        raise SchemaError('expected tool, arguments and explanation only')
    if value['tool'] not in TOOLS or not isinstance(value['arguments'], dict):
        raise SchemaError('unknown tool or invalid arguments')
    if not isinstance(value['explanation'], str) or not value['explanation'].strip():
        raise SchemaError('brief explanation required')
    guard = inspect_output({'explanation': value['explanation'], 'recommendation': value['arguments'].get('question', '')}, snapshot)
    if guard['blocked']:
        raise SchemaError('unsafe tool output: ' + ','.join(guard['reason_codes']))
    return value

def agent_step(draft, diffs, client):
    """Run one Agent-selected tool, preserving rejected trials and exact evidence."""
    from .incremental import _finalize
    state = _state(draft)
    feedback = draft.get('feedback')
    if feedback and feedback['status'] == 'unresolved':
        draft['pending_clarification'] = {'question_id': 'unresolved-feedback-target', 'question': '此事件没有关联的建议，请确认目标。'}
        return 'needs_clarification'
    body = _context(draft, state)
    stage_id = '%s:tool:%d' % (draft['run_id'], draft['step_count'])
    draft['usage']['llm_calls'] += 1
    record = {'step': draft['step_count'], 'decision_mode': 'llm', 'stage_id': stage_id, 'workflow_version': VERSION, 'prompt_version': VERSION, 'observation_hash': digest(body)}
    try:
        value = client.complete(stage_id, PROMPT, body, lambda v: _validate(v, draft['snapshot']))
        value = _validate(deepcopy(value), draft['snapshot'])
    except Exception as exc:
        record.update(decision_mode='llm_failed', error=type(exc).__name__ + ': ' + str(exc))
        draft['step_log'].append(record)
        draft['failures'].append(deepcopy(record))
        return 'failed'
    record.update(tool=value['tool'], raw=deepcopy(value))
    try:
        try:
            outcome, terminal = _execute(draft, state, value, diffs)
        except (ValueError, KeyError, TypeError) as exc:
            outcome, terminal = ({'status': 'rejected', 'error': str(exc)}, None)
    except (ValueError, KeyError, TypeError) as exc:
        outcome, terminal = ({'status': 'rejected', 'error': str(exc)}, None)
    state['observation'] = deepcopy(outcome)
    summary = {k: deepcopy(outcome[k]) for k in ('status', 'error', 'range', 'strategy', 'disposition', 'event_id') if k in outcome}
    if 'candidates' in outcome:
        summary['candidate_count'] = len(outcome['candidates'])
    if 'evaluation' in outcome:
        summary['evaluation'] = {k: outcome['evaluation'].get(k) for k in ('status', 'utility', 'input_hash')}
    if outcome['status'] == 'rejected':
        outcome['error_code'] = 'TOOL_ARGUMENT_OR_CONSTRAINT_REJECTED'
        state.setdefault('tool_errors', []).append({'tool': value['tool'], 'arguments': deepcopy(value['arguments']), 'error': outcome['error'], 'error_code': outcome['error_code']})
        state['tool_errors'] = state['tool_errors'][-12:]
    else:
        state['tool_errors'] = [e for e in state.get('tool_errors', []) if e['tool'] != value['tool']]
    state['observation'] = deepcopy(outcome)
    state.setdefault('tool_history', []).append({'tool': value['tool'], 'result': summary})
    state['tool_history'] = state['tool_history'][-24:]
    record['tool_result'] = deepcopy(outcome)
    draft['step_log'].append(record)
    if terminal:
        return terminal
    if _agent_completion_ready(draft, state, feedback):
        return _finalize(draft)
    return None

def _execute(draft, state, value, diffs):
    tool, args = (value['tool'], value['arguments'])
    contract = TOOL_CONTRACTS[tool]
    required = set(contract['required'])
    if required - set(args) or set(args) - required - set(contract.get('optional', [])):
        raise ValueError(tool + ' arguments must follow tool_contracts: ' + ','.join(contract['required']))
    for key, choices in contract.get('enums', {}).items():
        if key in args and args[key] not in choices:
            raise ValueError(key + ' must be one of ' + ','.join(choices))
    facts = fact_view(draft['snapshot'])
    if tool == 'record_event':
        from zoneinfo import ZoneInfo
        if set(args) != {'start_utc', 'end_utc', 'timezone', 'title', 'source_reference'}:
            raise ValueError('record_event requires start_utc,end_utc,timezone,title,source_reference')
        source = next((m for m in _user_messages(draft) if m['reference'] == args['source_reference']), None)
        if source is None:
            raise ValueError('source_reference must identify an actual user message')
        start, end = (dt(args['start_utc']), dt(args['end_utc']))
        ZoneInfo(args['timezone'])
        if start < dt(draft['snapshot']['now_utc']) or end <= start:
            raise ValueError('record_event must describe a positive future interval')
        if not isinstance(args['title'], str) or not args['title'].strip():
            raise ValueError('record_event title is required')
        if state['trial'] is not None:
            raise ValueError('retain or revert the current trial before recording a fact')
        after = dict(args, start_utc=stamp(start), end_utc=stamp(end), kind='environment', availability='unavailable', confirmed=True, source='agent_user_message', source_text=source['text'])
        event_id = 'user-event-' + digest({'source': source, 'start': stamp(start), 'end': stamp(end)})[:24]
        diffs.append({'target': 'timeline_event', 'op': 'add', 'event_id': event_id, 'before': None, 'after': after, 'affects_committed': False})
        return ({'status': 'recorded', 'event_id': event_id, 'event': after}, None)
    if tool == 'read':
        window = _range(draft, args)
        state['reads'].append(window)
        return ({'status': 'read', 'range': window, 'facts': [deepcopy(e) for e in draft['snapshot']['events'] if _in_range(e, window)], 'blocks': [deepcopy(b) for b in draft['plan']['sleep_blocks'] if _in_range(b, window)], 'constraints': summarize_constraints(constraint_report(facts, draft['plan']['sleep_blocks']))}, None)
    if tool == 'search':
        if state['trial'] is not None:
            raise ValueError('retain or revert the current trial before another search')
        window = _range(draft, args, bounded=True)
        strategy = args.get('strategy', 'habitual')
        if strategy not in ('habitual', 'moves', 'beam'):
            raise ValueError('unknown search strategy')
        segment = dict(window, entry_state=None)
        if strategy == 'habitual':
            proposals = habitual_proposals(facts, segment, _omitted(draft))
        elif strategy == 'moves':
            proposals = move_proposals(facts, draft['plan']['sleep_blocks'], segment)
        else:
            proposals = search_seed_proposals(facts, draft, segment, True)
        candidates = []
        for proposal in proposals:
            candidate, specs = candidate_blocks(draft['plan']['sleep_blocks'], proposal)
            valid, reasons = hard_validate(facts, candidate)
            edits = [_spec_edit(s) for s in specs]
            fingerprint = _edit_fingerprint(edits)
            previous = _matching_attempts(state, fingerprint, draft['facts_digest'], digest(draft['plan']['sleep_blocks']))
            candidates.append({'edits': edits, 'feasible': valid, 'reasons': reasons, 'source': proposal['source']})
            candidates[-1]['prior_attempts'] = previous
        from .incremental_proposals import segment_windows
        return ({'status': 'searched', 'range': window, 'strategy': strategy, 'candidates': candidates, 'empty_reason': None if candidates else 'strategy_produced_no_candidates' if segment_windows(facts, window) else 'no_known_free_window', 'next_options': ['edit exact times', 'try another search strategy'] if not candidates else [], 'approximation': 'search proposals are not physiological evaluations'}, None)
    if tool == 'edit':
        if set(args) != {'edits'}:
            raise ValueError('edit expects edits only')
        if state['trial'] is not None:
            raise ValueError('retain or revert the current trial before another edit')
        fingerprint = _edit_fingerprint(args['edits'])
        previous = _matching_attempts(state, fingerprint, draft['facts_digest'], digest(draft['plan']['sleep_blocks']))
        trial = _stage(draft, args['edits'], value['explanation'])
        trial['fingerprint'] = fingerprint
        trial['prior_attempts'] = previous
        state['trial'] = trial
        return ({'status': 'staged', 'edits': deepcopy(trial['edits']), 'constraints': summarize_constraints(trial['report']), 'prior_attempts': deepcopy(previous)}, None)
    if tool == 'evaluate':
        if set(args) - {'start_utc', 'end_utc'} or bool(set(args)) != ('start_utc' in args and 'end_utc' in args):
            raise ValueError('evaluate accepts both start_utc and end_utc, or neither')
        window = _range(draft, args) if args else _range(draft, {'start_utc': draft['snapshot']['now_utc'], 'end_utc': draft['snapshot']['horizon_end_utc']})
        trial = state['trial']
        blocks = trial['blocks'] if trial else draft['plan']['sleep_blocks']
        full_window = window['start_utc'] == draft['snapshot']['now_utc'] and window['end_utc'] == draft['snapshot']['horizon_end_utc']
        from .incremental import _current_score_entry, _stop_budget_exhausted
        expected_calls = 1 if full_window and (_current_score_entry(draft) or not trial) else 2
        if draft['budget'].get('max_model_calls') is not None and draft['usage']['model_calls'] + expected_calls > draft['budget']['max_model_calls']:
            return ({'status': 'budget_exhausted'}, _stop_budget_exhausted(draft, 'max_model_calls'))
        result, scoreable = run_evaluation(facts, deepcopy(blocks), sleep_count=True, start_utc=window['start_utc'], end_utc=window['end_utc'])
        if full_window:
            cached = _current_score_entry(draft)
            if cached is None:
                baseline_result, baseline_scoreable = run_evaluation(facts, deepcopy(draft['plan']['sleep_blocks']), sleep_count=True, start_utc=window['start_utc'], end_utc=window['end_utc']) if trial else (result, scoreable)
                baseline_evidence = _evidence(baseline_result)
                baseline_report = constraint_report(facts, draft['plan']['sleep_blocks'])
                cached = {'status': baseline_result.get('status'), 'utility': baseline_result.get('utility'), 'losses': deepcopy(baseline_result.get('losses')), 'input_hash': baseline_evidence.get('input_hash'), 'trace_hash': baseline_evidence.get('trace_hash'), 'constraint_failed_ids': [c.get('id') for c in baseline_report if c.get('status') == 'fail'], 'evaluation_start_utc': window['start_utc'], 'evaluation_end_utc': window['end_utc']}
                if baseline_scoreable:
                    from .incremental import _cache_key, _cache_entry, _refresh_entry_states
                    draft['score_cache'][_cache_key(draft, draft['plan']['edit_sequence'])] = _cache_entry(baseline_result, baseline_report)
                    _refresh_entry_states(draft, baseline_result.get('trace'))
                if trial:
                    draft['usage']['model_calls'] += 1
            baseline_evidence = deepcopy(cached)
            baseline_scoreable = bool(cached.get('status') == 'complete' and cached.get('utility') is not None)
            baseline_evidence.update(evaluation_start_utc=window['start_utc'], evaluation_end_utc=window['end_utc'])
        else:
            baseline_result, baseline_scoreable = run_evaluation(facts, deepcopy(draft['plan']['sleep_blocks']), sleep_count=True, start_utc=window['start_utc'], end_utc=window['end_utc'])
            baseline_evidence = _evidence(baseline_result)
            draft['usage']['model_calls'] += 1
        draft['usage']['model_calls'] += 1
        evidence = _evidence(result)
        comparison = {'baseline': baseline_evidence, 'trial': evidence, 'range': window, 'facts_version': draft['facts_digest'], 'plan_version': draft['plan']['edit_sequence']}
        if trial:
            trial.update(evaluation=evidence, scoreable=scoreable and baseline_scoreable, result=deepcopy(result), range=window, comparison=comparison)
        else:
            if full_window and scoreable:
                from .incremental import _cache_key, _cache_entry, _refresh_entry_states
                report = constraint_report(facts, blocks)
                draft['score_cache'][_cache_key(draft, draft['plan']['edit_sequence'])] = _cache_entry(result, report)
                _refresh_entry_states(draft, result.get('trace'))
            state['last_evaluation'] = evidence
            state['last_comparison'] = comparison
            state['last_evaluated_plan_version'] = draft['plan']['edit_sequence']
        return ({'status': 'evaluated', 'trial': bool(trial), 'scoreable': scoreable and baseline_scoreable, 'evaluation': evidence, 'comparison': comparison}, None)
    if tool in ('retain', 'revert'):
        if args:
            raise ValueError(tool + ' takes no arguments')
        trial = state['trial']
        if trial is None:
            raise ValueError('no trial exists')
        if tool == 'revert':
            _record_attempt(state, draft, trial, 'reverted', value['explanation'])
            state['trial'] = None
            return ({'status': 'reverted', 'edits': deepcopy(trial['edits'])}, None)
        if not trial.get('scoreable') or not trial.get('range'):
            raise ValueError('a complete real-model trial evaluation is required before retain')
        if not _range_covers_edits(trial['range'], trial['specs']):
            raise ValueError('evaluation range must cover every edited before and after interval')
        if trial['collateral']:
            _pending_confirmations(draft, trial)
            state['trial'] = None
            return ({'status': 'awaiting_confirmation', 'evaluation': trial['evaluation']}, 'needs_confirmation')
        diffs.extend(_retain(draft, state, trial))
        state['trial'] = None
        return ({'status': 'retained', 'evaluation': deepcopy(state['last_evaluation'])}, None)
    if tool == 'submit':
        window = _range(draft, args)
        disposition = args.get('disposition')
        if disposition not in ('planned', 'preserved', 'no_window'):
            raise ValueError('submit requires planned, preserved or no_window disposition')
        if state['trial'] is not None:
            raise ValueError('unresolved trial cannot be submitted')
        if not _covered(window, state['reads']):
            raise ValueError('read the complete submitted range first')
        evaluation = state.get('last_comparison')
        if evaluation is None:
            from .incremental import _current_score_entry
            cached = _current_score_entry(draft)
            if cached and cached.get('status') == 'complete':
                evaluation = {'range': {'start_utc': draft['snapshot']['now_utc'], 'end_utc': draft['snapshot']['horizon_end_utc']}, 'facts_version': draft['facts_digest'], 'plan_version': draft['plan']['edit_sequence'], 'trial': cached}
                state['last_comparison'] = evaluation
                state['last_evaluation'] = deepcopy(draft.get('model_observation') or {})
                state['last_evaluated_plan_version'] = draft['plan']['edit_sequence']
        if not evaluation:
            raise ValueError('submit requires a completed evaluation of the current plan version')
        if evaluation.get('facts_version') != draft['facts_digest']:
            raise ValueError('submit evaluation is stale against current facts: ' + str(evaluation.get('facts_version')) + ' != ' + str(draft['facts_digest']))
        if state.get('last_evaluated_plan_version') != draft['plan']['edit_sequence'] or evaluation.get('plan_version') != draft['plan']['edit_sequence']:
            raise ValueError('submit evaluation is stale against current plan version')
        if not _covered(window, [evaluation['range']]):
            raise ValueError('submit range is outside the current evaluation range')
        if evaluation['trial'].get('status') != 'complete' or evaluation['trial'].get('utility') is None:
            raise ValueError('current plan range must have a complete model evaluation')
        if not _feedback_satisfied(draft, draft['plan']['sleep_blocks']):
            raise ValueError('feedback target is unchanged or an omitted occurrence was reinserted')
        issues = _planning_evidence(draft, state)
        relevant = [issue for issue in issues if _in_range(issue, window)]
        state['planning_issues'] = issues
        unresolved = [issue for issue in relevant if issue['status'] == 'unresolved']
        if not relevant:
            unresolved = [{'start_utc': window['start_utc'], 'end_utc': window['end_utc'], 'status': 'unresolved', 'reason': 'missing_policy_demand_observation'}]
        if disposition == 'preserved' and any((i['status'] == 'planned' for i in relevant)):
            raise ValueError('preserved must be supported by unchanged evaluated arrangements')
        if disposition == 'no_window':
            from .incremental_proposals import segment_windows
            if segment_windows(facts, window):
                raise ValueError('known-free time exists; choose a sleep suggestion or explain another concrete blocker')
        if unresolved and disposition != 'no_window':
            impossible = any((i.get('reason') in ('constraints_leave_no_free_window', 'current_plan_constraint_conflict', 'missing_policy_demand_observation') for i in unresolved))
            return _unresolved_submission(draft, window, unresolved, clarify=impossible)
        state['submissions'].append(window)
        for segment in draft['segments']:
            if _covered(segment, state['submissions']):
                segment.update(status='processed', decision='agent_reviewed', reason=None)
        return ({'status': 'submitted', 'range': window, 'disposition': disposition}, None)
    if tool == 'clarify':
        if set(args) != {'question'} or not isinstance(args['question'], str) or (not args['question'].strip()):
            raise ValueError('clarify requires one question')
        draft['pending_clarification'] = {'question_id': 'agent-question-%d' % draft['step_count'], 'question': args['question']}
        return ({'status': 'needs_clarification', 'question': args['question']}, 'needs_clarification')
    raise ValueError('unsupported tool')

def _planning_evidence(draft, state):
    """Index coverage evidence by *existing policy* sleep-demand observations.

    These 24h/partial buckets are product scoring units from sleep_targets,
    not physiological day boundaries, mandatory edit units or score gates.
    A multi-day edit can explain many observations. Positive residual deficit
    does not fail completion; unexplained demand without an evaluated plan does.
    """
    from .incremental_proposals import segment_windows
    comparison = state.get('last_comparison') or {}
    if not comparison:
        from .incremental import _current_score_entry
        cached = _current_score_entry(draft)
        if cached and cached.get('status') == 'complete':
            window = {'start_utc': draft['snapshot']['now_utc'], 'end_utc': draft['snapshot']['horizon_end_utc']}
            comparison = {'range': window, 'facts_version': draft['facts_digest'], 'plan_version': draft['plan']['edit_sequence'], 'trial': cached, 'baseline': cached}
            state['last_comparison'] = comparison
            state['last_evaluation'] = deepcopy(draft.get('model_observation') or {})
            state['last_evaluated_plan_version'] = draft['plan']['edit_sequence']
    current = comparison.get('trial')
    evidence = state['last_evaluation'] or draft.get('model_observation') or {}
    if not current or comparison.get('facts_version') != draft['facts_digest'] or comparison.get('plan_version') != draft['plan']['edit_sequence'] or (state.get('last_evaluated_plan_version') != draft['plan']['edit_sequence']) or (evidence.get('input_hash') != current.get('input_hash')):
        return []
    buckets = (evidence.get('physiological_metrics') or {}).get('sleep_target_buckets') or []
    blocks = draft['plan']['sleep_blocks']
    originals = {b['id']: b for b in _original_plan(draft).get('sleep_blocks', [])}
    facts = fact_view(draft['snapshot'])
    issues = []
    for bucket in buckets:
        start = dt(bucket['start_utc'])
        end = start + timedelta(hours=float(bucket['hours']))
        window = {'start_utc': stamp(start), 'end_utc': stamp(end)}
        if float(bucket.get('target_hours', 0)) <= 0:
            continue
        related = [b for b in blocks if _in_range(b, window)]
        row = dict(window, policy='sleep_targets-existing-24h-buckets', target_hours=bucket['target_hours'], predicted_sleep_hours=bucket['sleep_hours'], residual_policy_deficit=bucket.get('deficit'), arrangement_ids=[b['id'] for b in related], model_input_hash=current['input_hash'], status='unresolved')
        failed_checks = [c.get('id') for c in evidence.get('constraint_report', []) if c.get('status') == 'fail']
        if current.get('constraint_failed_ids') or failed_checks:
            row.update(reason='current_plan_constraint_conflict', constraint_ids=deepcopy(current.get('constraint_failed_ids') or failed_checks))
        elif related and float(bucket['sleep_hours']) > 0:
            row.update(status='preserved' if all((originals.get(b['id']) == b for b in related)) else 'planned', reason='arrangement_and_full_horizon_model_evidence')
        elif any((_in_range(omitted, window) for omitted in _omitted(draft))) and _feedback_satisfied(draft, blocks):
            row.update(status='authorized_omit', reason='explicit_omit_this_occurrence', target_event_id=draft['request'].get('target_event_id'), omitted_block_ids=deepcopy((draft.get('feedback') or {}).get('linked_block_ids', [])))
        elif not segment_windows(facts, window):
            row['reason'] = 'constraints_leave_no_free_window'
        else:
            row['reason'] = 'model_predicts_no_sleep_from_arrangements' if related else 'unplanned_sleep_demand'
        issues.append(row)
    return issues

def _agent_completion_ready(draft, state, feedback):
    if state['trial'] is not None or not state.get('last_comparison'):
        return False
    comparison = state['last_comparison']
    if comparison.get('facts_version') != draft['facts_digest'] or comparison.get('plan_version') != draft['plan']['edit_sequence'] or state.get('last_evaluated_plan_version') != draft['plan']['edit_sequence'] or (comparison.get('trial', {}).get('status') != 'complete'):
        return False
    if feedback:
        if not _feedback_satisfied(draft, draft['plan']['sleep_blocks']):
            return False
        originals = {b['id']: b for b in _original_plan(draft).get('sleep_blocks', [])}
        current = {b['id']: b for b in draft['plan']['sleep_blocks']}
        target = [originals[i] for i in feedback.get('linked_block_ids', []) if i in originals]
        target += [current[i] for i in feedback.get('linked_block_ids', []) if i in current]
        if not target:
            return False
        required_ranges = [{'start_utc': b['start_utc'], 'end_utc': b['end_utc']} for b in target]
        required_ranges.extend(deepcopy(state.get('retained_edit_ranges', [])))
    else:
        required_ranges = [{'start_utc': draft['snapshot']['now_utc'], 'end_utc': draft['snapshot']['horizon_end_utc']}]
    return all((_covered(required, state.get('submissions', [])) for required in required_ranges))

def _unresolved_submission(draft, window, issues, *, clarify):
    outcome = {'status': 'needs_clarification' if clarify else 'incomplete', 'range': window, 'unresolved_demand': deepcopy(issues)}
    if not clarify:
        return (outcome, None)
    draft['pending_clarification'] = {'question_id': 'unresolved-planning-%d' % draft['step_count'], 'question': '这段范围仍有无法安排的睡眠需求，请说明需要优先处理的具体冲突。'}
    return (outcome, 'needs_clarification')

def _spec_edit(spec):
    op, event_id, before, after = spec
    if op == 'add':
        return {'op': 'add', 'after': {k: after[k] for k in ('start_utc', 'end_utc', 'timezone', 'role') if k in after}}
    if op == 'remove':
        return {'op': 'remove', 'event_id': event_id}
    return {'op': 'move', 'event_id': event_id, 'after': {k: after[k] for k in ('start_utc', 'end_utc', 'timezone', 'role') if k in after}}

def _immutable(block, snapshot):
    if dt(block['start_utc']) < dt(snapshot['now_utc']):
        return True
    if block.get('protection') in ('actual', 'fixed_external'):
        return True
    ids = {block['id'], block.get('origin_event_id')}
    if any((e['id'] in ids and (e.get('protection') in ('actual', 'fixed_external') or e.get('kind') in ('sleep_actual', 'flight', 'duty')) for e in snapshot['events'])):
        return True
    return any((s[3] == block['id'] for s in ongoing_session_spans(snapshot)))

def _committed(block, snapshot):
    origin = block.get('origin_event_id')
    return block_protected(block, snapshot) or (origin is not None and origin in commitment_protected_block_ids(snapshot)) or any((e['id'] == origin and event_protected(e) for e in snapshot['events']))

def _stage(draft, edits, reason):
    from .incremental import _next_block_id
    if not isinstance(edits, list) or not 1 <= len(edits) <= MAX_EDITS:
        raise ValueError('edit needs 1..8 associated actions')
    snapshot = draft['snapshot']
    working = deepcopy(draft['plan']['sleep_blocks'])
    original = {b['id']: deepcopy(b) for b in working}
    target_ids = set((draft.get('feedback') or {}).get('linked_block_ids') or [])
    feedback_targets = {f['target_event_id'] for f in snapshot.get('user_feedback', [])}
    target_ids.update((b['id'] for b in working if b.get('origin_event_id') in feedback_targets))
    touched, specs, proposals, collateral = (set(), [], [], [])
    for edit in edits:
        if not isinstance(edit, dict) or edit.get('op') not in ('add', 'move', 'remove'):
            raise ValueError('unknown edit operation')
        op = edit['op']
        allowed = {'op', 'after'} if op == 'add' else {'op', 'event_id', 'after'} if op == 'move' else {'op', 'event_id'}
        if set(edit) != allowed:
            raise ValueError('unexpected edit fields')
        before = None
        if op != 'add':
            block_id = edit['event_id']
            if block_id not in original or block_id in touched:
                raise ValueError('unknown or duplicate edit target')
            before = original[block_id]
            if _immutable(before, snapshot):
                raise ValueError('actual, ongoing, past or fixed arrangement is immutable')
            touched.add(block_id)
            if _committed(before, snapshot) and block_id not in target_ids:
                collateral.append(block_id)
        after = None
        if op != 'remove':
            raw = edit['after']
            if not isinstance(raw, dict) or set(raw) - {'start_utc', 'end_utc', 'timezone', 'role'}:
                raise ValueError('after allows only times, timezone and role')
            if not {'start_utc', 'end_utc'} <= set(raw):
                raise ValueError('after requires start_utc and end_utc')
            a, b = (dt(raw['start_utc']), dt(raw['end_utc']))
            if not dt(snapshot['now_utc']) <= a < b <= dt(snapshot['horizon_end_utc']):
                raise ValueError('edited arrangement outside future horizon')
            from zoneinfo import ZoneInfo
            timezone = raw.get('timezone', (before or {}).get('timezone', snapshot['profile']['home_timezone']))
            ZoneInfo(timezone)
            role = raw.get('role', (before or {}).get('role', 'main'))
            if role not in ROLE_VALUES:
                raise ValueError('role must be main or nap')
            after = dict(before or {}, start_utc=stamp(a), end_utc=stamp(b), timezone=timezone, role=role, id=before['id'] if before else _next_block_id(draft))
        proposal = {'kind': op, 'source': reason, 'event_id': before['id'] if before else None, 'before': before, 'after': after, 'blocks': [after] if op == 'add' else []}
        working, generated = candidate_blocks(working, proposal)
        specs.extend(generated)
        proposals.append(proposal)
    if _omitted_violation(draft, working):
        raise ValueError('equivalent sleep in omitted occurrence is forbidden for this revision')
    valid, reasons = hard_validate(fact_view(snapshot), working)
    report = constraint_report(fact_view(snapshot), working)
    if not valid or any((c['status'] == 'fail' for c in report)):
        raise ValueError('constraint rejection: ' + ';'.join(reasons + [c['id'] for c in report if c['status'] == 'fail']))
    return {'edits': deepcopy(edits), 'blocks': working, 'specs': specs, 'proposals': proposals, 'collateral': collateral, 'report': report, 'reason': reason, 'fingerprint': _edit_fingerprint(edits), 'base_plan_hash': digest(draft['plan']['sleep_blocks'])}

def _edit_fingerprint(edits):

    def clean(value):
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items() if k != 'id'}
        if isinstance(value, list):
            return [clean(v) for v in value]
        return value
    return digest(clean(edits))

def _matching_attempts(state, fingerprint, facts_version, base_plan_hash):
    return [deepcopy(a) for a in state.get('candidate_attempts', []) if a.get('fingerprint') == fingerprint and a.get('facts_version') == facts_version and (a.get('base_plan_hash') == base_plan_hash)]

def _record_attempt(state, draft, trial, disposition, explanation):
    state.setdefault('candidate_attempts', []).append({'fingerprint': trial.get('fingerprint'), 'facts_version': draft['facts_digest'], 'base_plan_hash': trial.get('base_plan_hash'), 'range': deepcopy(trial.get('range')), 'evaluation': deepcopy(trial.get('evaluation')), 'disposition': disposition, 'edit_reason': trial.get('reason'), 'revert_reason': explanation if disposition == 'reverted' else None, 'decision_reason': explanation, 'step': draft['step_count']})

def _omitted(draft):
    live = [f for f in draft['snapshot'].get('user_feedback', []) if f['meaning'] == 'omit_occurrence' and f.get('start_utc') and f.get('end_utc')]
    if draft['request'].get('meaning') != 'omit_occurrence':
        return draft.get('omitted_spans', []) + live
    target = set((draft.get('feedback') or {}).get('linked_block_ids') or [])
    return [b for b in _original_plan(draft).get('sleep_blocks', []) if b['id'] in target] + live

def _original_plan(draft):
    return draft.get('review_original_plan', draft['snapshot'].get('active_plan') or {})

def _omitted_violation(draft, blocks):
    target = set((draft.get('feedback') or {}).get('linked_block_ids') or [])
    originals = {b['id']: b for b in (draft['snapshot'].get('active_plan') or {}).get('sleep_blocks', [])}
    for block in blocks:
        if block['id'] in originals and block == originals[block['id']]:
            continue
        if any((_in_range(block, span) for span in _omitted(draft))):
            return True
    return False

def _feedback_satisfied(draft, blocks):
    originals = {b['id']: b for b in _original_plan(draft).get('sleep_blocks', [])}
    current = {b['id']: b for b in blocks}
    if _omitted_violation(draft, blocks):
        return False
    for item in draft['snapshot'].get('user_feedback', []):
        for original in originals.values():
            if original.get('origin_event_id') != item['target_event_id']:
                continue
            updated = current.get(original['id'])
            if updated == original or (item['meaning'] == 'replace' and updated is None):
                return False
    feedback = draft.get('feedback')
    if not feedback:
        return True
    target = set(feedback['linked_block_ids'])
    if not target or any((current.get(i) == originals.get(i) for i in target)):
        return False
    if draft['request']['meaning'] == 'omit_occurrence':
        if any((i in current for i in target)) or any((any((_in_range(b, s) for s in _omitted(draft))) for b in blocks)):
            return False
    elif not all((i in current for i in target)):
        return False
    return True

def _evidence(result):
    from .incremental import _summary_of
    evidence = _summary_of(result)
    evidence.update({k: deepcopy(result.get(k)) for k in ('physiological_metrics', 'preference_metrics', 'final_state', 'assumptions', 'evaluation_start_utc', 'evaluation_end_utc', 'initialization')})
    return evidence

def _retain(draft, state, trial):
    from .incremental import _commit, _cache_entry, _cache_key, _refresh_entry_states
    if digest(draft['plan']['sleep_blocks']) != trial['base_plan_hash']:
        raise ValueError('trial is stale against current draft')
    diffs = []
    for proposal, spec in zip(trial['proposals'], trial['specs']):
        diffs.extend(_commit(draft, proposal, [spec]))
    draft['plan']['edit_sequence'] += 1
    result = trial['result']
    full = trial['range']['start_utc'] == draft['snapshot']['now_utc'] and trial['range']['end_utc'] == draft['snapshot']['horizon_end_utc']
    if full:
        draft['score_cache'][_cache_key(draft, draft['plan']['edit_sequence'])] = _cache_entry(result, trial['report'])
        _refresh_entry_states(draft, result.get('trace'))
    state['last_evaluation'] = deepcopy(trial['evaluation'])
    state['last_comparison'] = deepcopy(trial['comparison'])
    state['last_comparison']['plan_version'] = draft['plan']['edit_sequence']
    state['last_evaluated_plan_version'] = draft['plan']['edit_sequence']
    _record_attempt(state, draft, trial, 'retained', trial.get('reason'))
    state.setdefault('retained_edit_ranges', []).extend(({'start_utc': item['start_utc'], 'end_utc': item['end_utc']} for spec in trial['specs'] for item in spec[2:4] if item is not None))
    earliest = _earliest_edit(trial['specs'])
    state['submissions'] = _subtract_submissions(state.get('submissions', []), earliest)
    state['planning_issues'] = []
    for segment in draft['segments']:
        if _covered(segment, state['submissions']):
            segment.update(status='processed', decision='agent_reviewed', reason=None)
        else:
            segment.update(status='pending', decision=None, reason=None)
    if draft.get('feedback') and _feedback_satisfied(draft, draft['plan']['sleep_blocks']):
        draft['feedback']['status'] = 'applied'
    return diffs

def _earliest_edit(edits):
    times = []
    for edit in edits:
        items = edit[2:4] if isinstance(edit, (list, tuple)) else (edit.get('after'), edit.get('before'))
        for item in items:
            if item and item.get('start_utc'):
                times.append(dt(item['start_utc']))
    return min(times) if times else None

def _range_covers_edits(window, specs):
    a, b = (dt(window['start_utc']), dt(window['end_utc']))
    for spec in specs:
        for item in spec[2:4]:
            if item is not None and (dt(item['start_utc']) < a or dt(item['end_utc']) > b):
                return False
    return True

def _subtract_submissions(ranges, earliest):
    if earliest is None:
        return deepcopy(ranges)
    kept = []
    for item in ranges:
        a, b = (dt(item['start_utc']), dt(item['end_utc']))
        if b <= earliest:
            kept.append(item)
        elif a < earliest:
            kept.append({'start_utc': item['start_utc'], 'end_utc': stamp(earliest)})
    return kept

def _pending_confirmations(draft, trial):
    group_id = 'agent-batch-%d' % draft['step_count']
    for block_id in trial['collateral']:
        spec = next((s for s in trial['specs'] if s[1] == block_id))
        cid = 'conf-%03d' % draft['next_confirmation_id']
        draft['next_confirmation_id'] += 1
        draft['pending_confirmations'].append({'confirmation_id': cid, 'event_id': block_id, 'before': deepcopy(spec[2]), 'after': deepcopy(spec[3]), 'reason': trial['reason'], 'agent_batch_id': group_id, 'proposals': [], 'evaluation': deepcopy(trial['evaluation'])})
    draft['agent_state']['confirmation_trial'] = deepcopy(trial)

def apply_confirmation(draft, entry, decision):
    """Atomic batch confirmation hook used by the unchanged public resume API."""
    state = _state(draft)
    trial = state.get('confirmation_trial')
    if trial is None:
        raise ValueError('missing associated confirmation trial')
    if decision == 'reject':
        draft['pending_confirmations'] = [c for c in draft['pending_confirmations'] if c.get('agent_batch_id') != entry['agent_batch_id']]
        _record_attempt(state, draft, trial, 'reverted', 'collateral_confirmation_rejected')
        state.pop('confirmation_trial', None)
        state['observation'] = {'status': 'confirmation_rejected', 'reason': 'associated batch discarded'}
        return []
    if any((c.get('agent_batch_id') == entry['agent_batch_id'] for c in draft['pending_confirmations'])):
        return []
    result = _retain(draft, state, trial)
    state.pop('confirmation_trial', None)
    state['observation'] = {'status': 'confirmation_applied', 'evaluation': trial['evaluation']}
    return result

def _covered(window, pieces):
    cursor, end = (dt(window['start_utc']), dt(window['end_utc']))
    for piece in sorted(pieces, key=lambda p: p['start_utc']):
        a, b = (dt(piece['start_utc']), dt(piece['end_utc']))
        if a <= cursor < b:
            cursor = b
        if cursor >= end:
            return True
    return cursor >= end
