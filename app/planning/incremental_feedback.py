"""Authority-bound repeated-feedback gate, not model-inferred preference.

The timeline service owns occurrence lineage, deduplication, undo and reason
normalization. Planning validates its internal v1 evidence and pauses before
tools until this exact pending question has an answer in the current run.
"""
from copy import deepcopy
from .state import digest


def validate_context(context):
    if context is None:
        return
    prefix = 'invalid_snapshot: feedback_context'
    if (not isinstance(context, dict) or type(context.get('schema_version')) is not int
            or context['schema_version'] != 1):
        raise ValueError(prefix + '.schema_version must be 1')
    required = {'current_feedback_id', 'target_event_id', 'occurrence_id', 'requires_clarification',
                'reason_code', 'has_material_new_reason', 'history'}
    missing = sorted(required - set(context))
    if missing:
        raise ValueError(prefix + '.' + missing[0] + ' missing')
    for key in ('requires_clarification', 'has_material_new_reason'):
        if not isinstance(context[key], bool):
            raise ValueError(prefix + '.' + key + ' must be boolean')
    for key in ('current_feedback_id', 'occurrence_id', 'reason_code'):
        if context[key] is not None and (not isinstance(context[key], str) or not context[key]):
            raise ValueError(prefix + '.' + key + ' must be nonempty string or null')
    if not isinstance(context['target_event_id'], str) or not context['target_event_id']:
        raise ValueError(prefix + '.target_event_id must be nonempty string')
    if not isinstance(context['history'], list):
        raise ValueError(prefix + '.history must be list')
    seen = set()
    for index, row in enumerate(context['history']):
        path = prefix + '.history[%d]' % index
        if not isinstance(row, dict):
            raise ValueError(path + ' must be object')
        for key in ('feedback_id', 'target_event_id', 'source'):
            if not isinstance(row.get(key), str) or not row[key]:
                raise ValueError(path + '.' + key + ' must be nonempty string')
        if row['feedback_id'] in seen:
            raise ValueError(path + '.feedback_id duplicates an earlier record')
        seen.add(row['feedback_id'])
        for key in ('target_revision', 'feedback_sequence'):
            if not isinstance(row.get(key), int) or isinstance(row[key], bool) or row[key] < 0:
                raise ValueError(path + '.' + key + ' must be nonnegative integer')
        if row.get('meaning') not in ('replace', 'omit_occurrence'):
            raise ValueError(path + '.meaning invalid')
        if row.get('status') not in ('active', 'undone'):
            raise ValueError(path + '.status invalid')
        if row.get('reason') is not None and not isinstance(row['reason'], str):
            raise ValueError(path + '.reason must be string or null')
        if row.get('occurrence_id', context['occurrence_id']) != context['occurrence_id']:
            raise ValueError(path + '.occurrence_id differs from the service-selected chain')
        replacement = row.get('applied_replacement')
        if replacement is not None:
            if not isinstance(replacement, dict) or any(not isinstance(replacement.get(k), str) or not replacement[k]
                    for k in ('job_id', 'run_id', 'before_event_id', 'after_event_id', 'applied_at')):
                raise ValueError(path + '.applied_replacement lacks versioned publication evidence')
    if context['requires_clarification']:
        current_id = context['current_feedback_id']
        current = next((r for r in context['history'] if r['feedback_id'] == current_id), None)
        if (current is None or current['status'] != 'active' or current['meaning'] != 'replace'
                or current['target_event_id'] != context['target_event_id']
                or not context['occurrence_id'] or context['has_material_new_reason']):
            raise ValueError(prefix + ': repeated-rejection gate lacks active current replace evidence')
        if not any(r['feedback_id'] != current_id and r['status'] == 'active' and r['meaning'] == 'replace'
                   and r.get('applied_replacement') and r['feedback_sequence'] < current['feedback_sequence']
                   for r in context['history']):
            raise ValueError(prefix + ': repeated-rejection gate lacks earlier active published replacement')


def bind_guard(draft):
    context = draft['snapshot'].get('feedback_context')
    request = draft['request']
    if not context or request['kind'] != 'feedback':
        return
    if context['target_event_id'] != request['target_event_id']:
        raise ValueError('invalid_snapshot: feedback_context.target_event_id does not match request')
    if not context['requires_clarification']:
        return
    if request['meaning'] != 'replace':
        raise ValueError('invalid_snapshot: repeated-rejection gate requires replace request')
    guard = {'feedback_id': context['current_feedback_id'],
             'question_id': 'repeat-rejection:' + context['current_feedback_id'],
             'context_digest': digest(context), 'answered': False,
             'question': '前一个替代也不合适，你希望这次具体改哪一点？'}
    draft['repeated_feedback_guard'] = guard
    # An unresolved event link has its own more basic question first. It cannot
    # authorize tools merely by answering that unrelated clarification.
    if draft.get('feedback', {}).get('status') != 'unresolved':
        require_answer(draft)


def require_answer(draft):
    guard = draft.get('repeated_feedback_guard')
    if not guard or guard['answered']:
        return False
    context = draft['snapshot'].get('feedback_context') or {}
    if (guard['feedback_id'] != context.get('current_feedback_id') or
            guard['context_digest'] != digest(context)):
        raise ValueError('repeated-feedback context changed; start a new run')
    draft['pending_clarification'] = {'question_id': guard['question_id'], 'question': guard['question']}
    draft['status'] = 'needs_clarification'
    return True


def accept_answer(draft, question_id, answer):
    """Only this pending question/feedback/context may release its own gate."""
    guard = draft.get('repeated_feedback_guard')
    if not guard or question_id != guard['question_id']:
        return
    context = draft['snapshot'].get('feedback_context') or {}
    if (guard['feedback_id'] != context.get('current_feedback_id') or
            guard['context_digest'] != digest(context)):
        raise ValueError('repeated-feedback answer is stale against authoritative history')
    guard.update(answered=True, answer=deepcopy(answer), answered_question_id=question_id)
