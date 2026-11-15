"""Incremental observe/action/tool-evaluation scheduling loop (INCREMENTAL-PLANNING-01).

Pure-Python contract surface frozen with timeline-api across the
``interface-agreement*.md`` chain (latest file wins; 09 is current):

- ``read_plan(snapshot)`` — the public plan read surface:
  ``{'plan_revision': snapshot['revision'], 'blocks': [...]}`` with stable
  block ids and the optional ``origin_event_id`` link (agreement-05 §1).
- ``begin_run(snapshot, *, request, budget=None, run_id=None)`` — start a run;
  ``request.kind`` is ``initial`` or ``feedback`` (the LLM ``text`` intent is
  out of scope in v1, agreement-03 §1). The run state embeds the snapshot and
  is JSON-serializable; planning persists nothing and provides no HTTP.
- ``step(run_state, snapshot=None, *, mode='step', limit=None)`` — perform
  work and return the frozen per-step result ``{step_index, step_status,
  event_diffs[], constraint_summary, score_summary, needs, draft_digest,
  run_state}``. ``mode='step'`` (default) executes exactly one step;
  ``mode='continue'`` runs up to ``limit`` steps (default
  ``CONTINUE_LIMIT``) or until a terminal state.
- ``resume_run(run_state, *, input=None, budget_extension=None,
  snapshot=None)`` — apply a clarification answer, a confirmation decision or
  a budget extension and return the updated run state (agreement-03 §3).

Statuses: ``step | continue | complete | needs_clarification |
needs_confirmation | budget_exhausted | failed``. Per agreement-08 §2 the two
non-terminal statuses are **mode-based**: ``step`` = single-step mode result
(one step executed, run not finished); ``continue`` = continue-mode result
(stopped at a step boundary with uncovered time and budget left, no user
input needed). Terminal states: ``complete`` / ``needs_clarification`` /
``needs_confirmation`` / ``budget_exhausted`` / ``failed``.

``needs`` always carries the two frozen wire keys (agreement-04 §2):
``{'clarification': {'question_id', 'question'} | None, 'confirmations':
[{'confirmation_id', 'event_id', 'before', 'after', 'reason'}] | None}``;
internal ``pending_confirmations[]`` project into ``needs.confirmations``.

Staleness (agreement-09 C2): the draft stores a **facts digest** that excludes
``now_utc``/``horizon_end_utc``. When the caller passes the current snapshot,
a facts change (events / profile / sessions / preferences / scheduling_policy
/ revision / input_revision) raises ``StaleDraftError`` (the service rebases
with a new run); a pure clock advance rebases the run in place — segments are
re-seeded from the new ``now``, plan blocks are kept, and the evaluation is
re-run from the new ``now`` (the model always simulates from ``now``; there is
no calendar reset). The score cache is keyed by facts digest + evaluation
window + edit sequence, never by the sequence alone (C1).

Model-face rule (agreements 06/07/09): every model or engine call uses a fact
view whose event list carries only interval-bearing events and never the
planning-owned kinds ``rest_day``/``sleep_plan``; candidate blocks travel only
as ``compute``'s second parameter and are never written into events. All
physiology comes from the public model composition
(``app.planning.evaluation`` → ``app.sleep_model.api.compute``); the loop
re-derives the state per evaluation — no ``model_checkpoint`` reuse and no
``advance_state`` chain (a stated design choice, agreement-09 §1.3). The
per-segment entry ``S`` is a planning-owned search seed, not a checkpoint.

Feedback targets a timeline event id and is resolved to plan blocks through
the ``origin_event_id`` link (defensively through a commitment's
``target_event_id``); an unresolved target surfaces ``needs_clarification`` —
never a silent no-op. Protection is the frozen union rule plus protection by
reference through commitment records; edits touching protected context are
never applied automatically — they become pending concrete confirmations.
Agent I/O has no propagation/decay/work-progress field. Runtime coverage
metadata remains inside the service-owned draft envelope.
"""
import re
from copy import deepcopy
from datetime import timedelta
from zoneinfo import ZoneInfo
from app.contracts.time import dt, stamp
from app.sleep_model.contracts import VERSION as MODEL_VERSION
from .engine import POLICY_VERSION
from .incremental_proposals import build_proposals, candidate_blocks, constraint_report, fact_view, hard_validate, habitual_proposals, proposal_key, run_evaluation, search_seed_proposal, segment_covers, summarize_constraints, block_protected, event_protected, unmatched_commitment_notes
from .search import VERSION as SEARCH_VERSION
from .state import digest
DRAFT_VERSION = 'incremental-run-v1'
STEP_STATUSES = ('step', 'continue', 'complete', 'needs_clarification', 'needs_confirmation', 'budget_exhausted', 'failed')
REQUEST_KINDS = ('initial', 'feedback')
FEEDBACK_MEANINGS = ('replace', 'omit_occurrence')
CONFIRMATION_DECISIONS = ('apply', 'reject')
SEGMENT_HOURS = 48
PROPOSALS_PER_STEP = 3
CONTINUE_LIMIT = 8
DEFAULT_MAX_LLM_CALLS = 64
TERMINAL_STATUSES = ('complete', 'needs_clarification', 'needs_confirmation', 'budget_exhausted', 'failed')
DECISION_MODES = ('llm', 'deterministic')
from .incremental_agent import PROMPT as DECISION_PROMPT, VERSION as DECISION_PROMPT_VERSION
DECISION_WORKFLOW_VERSION = DECISION_PROMPT_VERSION
_VALID_CLOCK = re.compile('^([01]\\d|2[0-3]):[0-5]\\d$')
BLOCKING_EVENT_KINDS = frozenset({'duty', 'flight', 'sleep_actual', 'commute', 'preparation', 'protected_rest', 'environment'})
_FACT_KEYS = ('events', 'profile', 'sessions', 'active_plan', 'preferences', 'scheduling_policy', 'revision', 'input_revision', 'feedback_context', 'user_feedback')

class IncrementalError(ValueError):
    """Base class of the loop's caller-contract errors (field-path messages)."""

class InvalidSnapshotError(IncrementalError):
    """The snapshot violates the planning read contract (maps to 400)."""

class InvalidRequestError(IncrementalError):
    """The request/input/budget payload violates the frozen contract (maps to 400)."""

class DecisionClientRequiredError(InvalidRequestError):
    """LLM decision mode without a decision client (Lead steering msgs 290/297).

    The production path must explicitly configure and pass the decision
    client; a missing client is an explicit error, never a silent
    deterministic downgrade presented as agent success."""

class StaleDraftError(IncrementalError):
    """The draft does not match the current input facts (maps to 409/conflict)."""

def read_plan(snapshot):
    """Public plan read surface (agreement-05 §1): the current effective plan
    as planning sees it, with its authoritative ``plan_revision``."""
    if not isinstance(snapshot, dict):
        raise InvalidSnapshotError('invalid_snapshot: snapshot must be a dict')
    if 'revision' not in snapshot:
        raise InvalidSnapshotError('invalid_snapshot: revision missing')
    if not isinstance(snapshot['revision'], int) or isinstance(snapshot['revision'], bool):
        raise InvalidSnapshotError('invalid_snapshot: revision must be an integer')
    blocks = deepcopy((snapshot.get('active_plan') or {}).get('sleep_blocks') or [])
    return {'plan_revision': snapshot['revision'], 'blocks': blocks}

def begin_run(snapshot, *, request, budget=None, run_id=None, client=None, decision_mode=None):
    """Start an incremental run and return the JSON-serializable run state.

    ``decision_mode`` defaults to ``'llm'`` (the production semantics): the
    run then requires a decision client at every step, and a missing client is
    an explicit :class:`DecisionClientRequiredError` — never a silent
    deterministic downgrade (Lead steering msgs 290/297). ``'deterministic'``
    is only reachable by explicit request (offline/degraded diagnostics,
    ``degraded=true`` in provenance) and rejects a client."""
    _validate_snapshot(snapshot)
    request = _normalize_request(request, snapshot)
    caps = _normalize_budget(budget, 'budget')
    mode = _resolve_decision_mode(decision_mode, client)
    draft = _new_draft(snapshot, request, caps, run_id, mode)
    _run_baseline(draft)
    if request['kind'] == 'feedback':
        _prepare_feedback(draft)
    draft['notes'].extend(unmatched_commitment_notes(snapshot, {b.get('id') for b in draft['plan']['sleep_blocks']}))
    for event in snapshot.get('events', []):
        if event.get('kind') == 'rest_day':
            draft['notes'].append('rest day fact: ' + str(event.get('date')) + ' (' + str(event.get('timezone')) + '); no duty on that local date, no sleep window, no actual sleep derived')
    _assess_request_guard(draft)
    from .incremental_feedback import bind_guard
    try:
        bind_guard(draft)
    except ValueError as exc:
        raise InvalidSnapshotError(str(exc)) from None
    return draft

def step(run_state, snapshot=None, *, mode='step', limit=None, client=None):
    return _step(run_state, snapshot, mode=mode, limit=limit, client=client)

def _step(run_state, snapshot=None, *, mode='step', limit=None, client=None):
    """Perform work and return the frozen per-step result.

    ``mode='step'`` (default) executes exactly one step; ``mode='continue'``
    executes up to ``limit`` steps (default ``CONTINUE_LIMIT``) or stops at the
    first terminal state. Non-terminal results carry ``step_status='step'``
    (single-step mode) or ``'continue'`` (continue mode) per agreement-08 §2.
    Passing ``snapshot`` enables the staleness/rebase rules of agreement-09 C2.
    In LLM decision mode every step requires the decision client (missing →
    :class:`DecisionClientRequiredError`); in explicit deterministic mode a
    client is rejected.
    """
    if mode not in ('step', 'continue'):
        raise InvalidRequestError("invalid_request: mode must be 'step' or 'continue'")
    if limit is not None and (not isinstance(limit, int) or isinstance(limit, bool) or limit < 1):
        raise InvalidRequestError('invalid_request: limit must be a positive integer')
    draft = _validate_draft(run_state)
    _validate_decision_client(draft['decision_mode'], client)
    if draft['decision_mode'] == 'llm':
        from .incremental_agent import _state
        _state(draft)
    if snapshot is not None:
        _sync_snapshot(draft, snapshot)
    iterations = 1 if mode == 'step' else limit if limit is not None else CONTINUE_LIMIT
    diffs = []
    status = None
    for _ in range(iterations):
        status, step_diffs = _step_once(draft, client)
        diffs.extend(step_diffs)
        if status is not None:
            break
    if status is None:
        status = 'step' if mode == 'step' else 'continue'
    draft['status'] = status
    return _build_result(draft, status, diffs)

def resume_run(run_state, *, input=None, budget_extension=None, snapshot=None, client=None):
    """Apply a clarification answer, a confirmation decision or a budget
    extension and return the updated run state (agreement-03 §3: resume
    returns the run state; the next ``step`` call reports the outcome).
    The decision-mode/client rules match :func:`step`."""
    draft = _validate_draft(run_state)
    _validate_decision_client(draft['decision_mode'], client)
    if snapshot is not None:
        _sync_snapshot(draft, snapshot)
    if budget_extension is not None:
        _apply_budget_extension(draft, budget_extension)
    if input is not None:
        _apply_input(draft, input)
    elif draft['status'] == 'failed' and draft.get('agent_state'):
        draft['status'] = 'pending'
    return draft

def _validate_snapshot(snapshot):
    """Field-path validation of the planning read contract.

    Required profile fields of the new incremental API are never defaulted
    (agreement-05 §2): ``home_timezone`` (IANA), ``preferred_sleep_start_hour``
    (0–23), ``sleep_need_hours`` (in (0, 24]) and ``habit_wake_local``
    ("HH:MM"). Interval-bearing events are required for the blocking kinds;
    ``rest_day`` facts and reference-only commitment markers are legitimately
    interval-less (agreements 06/07).
    """
    if not isinstance(snapshot, dict):
        raise InvalidSnapshotError('invalid_snapshot: snapshot must be a dict')
    from .incremental_feedback import validate_context
    try:
        validate_context(snapshot.get('feedback_context'))
    except ValueError as exc:
        raise InvalidSnapshotError(str(exc)) from None
    for key in ('now_utc', 'horizon_end_utc', 'profile', 'events', 'revision'):
        if key not in snapshot:
            raise InvalidSnapshotError('invalid_snapshot: ' + key + ' missing')
    for key in ('now_utc', 'horizon_end_utc'):
        try:
            dt(snapshot[key])
        except (ValueError, TypeError, KeyError) as exc:
            raise InvalidSnapshotError('invalid_snapshot: ' + key + ' ' + str(exc)) from None
    if not isinstance(snapshot['events'], list):
        raise InvalidSnapshotError('invalid_snapshot: events must be a list')
    for index, event in enumerate(snapshot['events']):
        if not isinstance(event, dict):
            raise InvalidSnapshotError('invalid_snapshot: events[%d] must be a dict' % index)
        for key in ('id', 'kind'):
            if not isinstance(event.get(key), str) or not event[key]:
                raise InvalidSnapshotError('invalid_snapshot: events[%d].%s must be a non-empty string' % (index, key))
        kind = event['kind']
        if kind in BLOCKING_EVENT_KINDS:
            for key in ('start_utc', 'end_utc'):
                if event.get(key) is None:
                    raise InvalidSnapshotError('invalid_snapshot: events[%d] (%s) requires %s' % (index, kind, key))
        for key in ('start_utc', 'end_utc'):
            if event.get(key) is not None:
                try:
                    dt(event[key])
                except (ValueError, TypeError, KeyError) as exc:
                    raise InvalidSnapshotError('invalid_snapshot: events[%d].%s %s' % (index, key, exc)) from None
    profile = snapshot['profile']
    if not isinstance(profile, dict):
        raise InvalidSnapshotError('invalid_snapshot: profile must be a dict')
    timezone_name = profile.get('home_timezone')
    if not isinstance(timezone_name, str) or not timezone_name:
        raise InvalidSnapshotError('invalid_snapshot: profile.home_timezone missing (IANA name required)')
    try:
        ZoneInfo(timezone_name)
    except Exception as exc:
        raise InvalidSnapshotError('invalid_snapshot: profile.home_timezone ' + str(exc)) from None
    hour = profile.get('preferred_sleep_start_hour')
    if not isinstance(hour, (int, float)) or isinstance(hour, bool) or (not 0 <= hour <= 23):
        raise InvalidSnapshotError('invalid_snapshot: profile.preferred_sleep_start_hour must be a number in [0, 23]')
    need = profile.get('sleep_need_hours')
    if not isinstance(need, (int, float)) or isinstance(need, bool) or (not 0 < need <= 24):
        raise InvalidSnapshotError('invalid_snapshot: profile.sleep_need_hours must lie in (0, 24]')
    wake = profile.get('habit_wake_local')
    if not isinstance(wake, str) or not _VALID_CLOCK.match(wake):
        raise InvalidSnapshotError('invalid_snapshot: profile.habit_wake_local must be "HH:MM"')
    if not isinstance(snapshot['revision'], int) or isinstance(snapshot['revision'], bool):
        raise InvalidSnapshotError('invalid_snapshot: revision must be an integer')
    active_plan = snapshot.get('active_plan')
    if active_plan is not None:
        if not isinstance(active_plan, dict):
            raise InvalidSnapshotError('invalid_snapshot: active_plan must be a dict or None')
        _validate_blocks(active_plan.get('sleep_blocks'), 'active_plan.sleep_blocks')
    for key in ('preferences',):
        if snapshot.get(key) is not None and (not isinstance(snapshot[key], list)):
            raise InvalidSnapshotError('invalid_snapshot: ' + key + ' must be a list or None')
    for key in ('sessions', 'scheduling_policy'):
        if snapshot.get(key) is not None and (not isinstance(snapshot[key], dict)):
            raise InvalidSnapshotError('invalid_snapshot: ' + key + ' must be a dict or None')

def _validate_blocks(blocks, path):
    if blocks is None:
        return
    if not isinstance(blocks, list):
        raise InvalidSnapshotError('invalid_snapshot: ' + path + ' must be a list')
    seen = set()
    for index, block in enumerate(blocks):
        if not isinstance(block, dict) or not isinstance(block.get('id'), str) or (not block['id']):
            raise InvalidSnapshotError('invalid_snapshot: %s[%d].id must be a non-empty string' % (path, index))
        if block['id'] in seen:
            raise InvalidSnapshotError('invalid_snapshot: %s[%d].id duplicate' % (path, index))
        seen.add(block['id'])
        for key in ('start_utc', 'end_utc'):
            if key not in block:
                raise InvalidSnapshotError('invalid_snapshot: %s[%d].%s missing' % (path, index, key))
            try:
                dt(block[key])
            except (ValueError, TypeError, KeyError) as exc:
                raise InvalidSnapshotError('invalid_snapshot: %s[%d].%s %s' % (path, index, key, exc)) from None

def _validate_draft(run_state):
    if not isinstance(run_state, dict) or run_state.get('draft_version') != DRAFT_VERSION:
        raise StaleDraftError('stale draft: unknown or missing draft_version (expected ' + DRAFT_VERSION + ')')
    for key in ('run_id', 'snapshot', 'request', 'plan', 'segments', 'budget', 'usage', 'step_count', 'score_cache', 'provenance', 'status', 'step_log', 'pending_confirmations', 'facts_digest'):
        if key not in run_state:
            raise StaleDraftError('stale draft: run state is missing ' + key)
    _validate_snapshot(run_state['snapshot'])
    return run_state

def _normalize_request(request, snapshot):
    if not isinstance(request, dict):
        raise InvalidRequestError('invalid_request: request must be a dict')
    kind = request.get('kind')
    if kind not in REQUEST_KINDS:
        raise InvalidRequestError('invalid_request: request.kind must be one of ' + ', '.join(REQUEST_KINDS))
    if kind == 'initial':
        unknown = sorted(set(request) - {'kind'})
        if unknown:
            raise InvalidRequestError('invalid_request: request.' + unknown[0] + ' is not accepted for kind initial')
        return {'kind': 'initial'}
    target = request.get('target_event_id')
    if not isinstance(target, str) or not target:
        raise InvalidRequestError('invalid_request: request.target_event_id must be a non-empty string')
    meaning = request.get('meaning')
    if meaning not in FEEDBACK_MEANINGS:
        raise InvalidRequestError("invalid_request: request.meaning must be 'replace' or 'omit_occurrence'")
    reason = request.get('reason')
    if reason is not None and (not isinstance(reason, str)):
        raise InvalidRequestError('invalid_request: request.reason must be a string or None')
    unknown = sorted(set(request) - {'kind', 'target_event_id', 'meaning', 'reason'})
    if unknown:
        raise InvalidRequestError('invalid_request: request.' + unknown[0] + ' is not accepted')
    if target not in {e.get('id') for e in snapshot.get('events', [])}:
        raise InvalidRequestError('invalid_request: request.target_event_id ' + target + ' does not match any snapshot event')
    return {'kind': 'feedback', 'target_event_id': target, 'meaning': meaning, 'reason': reason}

def _normalize_budget(budget, field):
    """Frozen three-key budget: ``max_steps`` / ``max_model_calls`` keep their
    D1 semantics (missing = unlimited), ``max_llm_calls`` is new with a finite
    default of 64; an explicit ``None`` means unlimited and is recorded."""
    caps = {'max_steps': None, 'max_model_calls': None, 'max_llm_calls': DEFAULT_MAX_LLM_CALLS}
    if budget is None:
        return caps
    if not isinstance(budget, dict):
        raise InvalidRequestError('invalid_request: ' + field + ' must be a dict or None')
    unknown = sorted(set(budget) - {'max_steps', 'max_model_calls', 'max_llm_calls'})
    if unknown:
        raise InvalidRequestError('invalid_request: ' + field + '.' + unknown[0] + ' is not accepted')
    for key in ('max_steps', 'max_model_calls'):
        value = budget.get(key)
        if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 0):
            raise InvalidRequestError('invalid_request: ' + field + '.' + key + ' must be a non-negative integer or None')
        caps[key] = value
    if 'max_llm_calls' in budget:
        value = budget['max_llm_calls']
        if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 0):
            raise InvalidRequestError('invalid_request: ' + field + '.max_llm_calls must be a non-negative integer or None')
        caps['max_llm_calls'] = value
    return caps

def _resolve_decision_mode(decision_mode, client):
    if decision_mode is None:
        mode = 'llm'
    elif decision_mode in DECISION_MODES:
        mode = decision_mode
    else:
        raise InvalidRequestError("invalid_request: decision_mode must be 'llm' or 'deterministic'")
    _validate_decision_client(mode, client)
    return mode

def _validate_decision_client(mode, client):
    if mode == 'deterministic':
        if client is not None:
            raise InvalidRequestError("invalid_request: client must be None for decision_mode='deterministic'")
        return
    if client is None:
        raise DecisionClientRequiredError("invalid_request: decision client required for decision_mode='llm'")
    if not callable(getattr(client, 'complete', None)):
        raise InvalidRequestError('invalid_request: client must expose complete(stage_id, system, body, validate)')

def _assess_request_guard(draft):
    """Safety screening of the user text/metadata that reaches the decision
    payload (D2 requirement 5; legacy ``run_step`` pattern)."""
    from .safety import assess_request
    snapshot = draft['snapshot']
    request = draft['request']
    user_text = request.get('reason') or '' if request.get('kind') == 'feedback' else ''
    metadata = '\n'.join((str(e.get('title', '')) + ' ' + str(e.get('notes', '')) for e in snapshot.get('events', [])))
    guard = assess_request((user_text + '\n' + metadata).strip(), snapshot)
    draft['request_guard'] = {'blocked': guard['blocked'], 'reason_codes': guard['reason_codes'], 'message': guard['message']}
    if guard['blocked']:
        draft['notes'].append('request guard blocked: ' + ','.join(guard['reason_codes']))

def _new_draft(snapshot, request, caps, run_id, decision_mode='deterministic'):
    blocks = deepcopy((snapshot.get('active_plan') or {}).get('sleep_blocks') or [])
    provenance = {}
    for block in blocks:
        provenance[block['id']] = 'carried'
    degraded = decision_mode == 'deterministic'
    decision_provenance = {'mode': decision_mode, 'degraded': degraded}
    if decision_mode == 'llm':
        decision_provenance.update({'prompt_version': DECISION_PROMPT_VERSION, 'workflow_version': DECISION_WORKFLOW_VERSION})
    return {'draft_version': DRAFT_VERSION, 'run_id': run_id if run_id is not None else 'run-' + digest({'snapshot_hash': digest(snapshot), 'request': request})[:16], 'snapshot': deepcopy(snapshot), 'request': deepcopy(request), 'facts_digest': digest(_facts_of(snapshot)), 'decision_mode': decision_mode, 'plan': {'sleep_blocks': blocks, 'provenance': provenance, 'next_block_id': 1, 'edit_sequence': 0}, 'segments': _segments(snapshot), 'omitted_spans': [], 'budget': caps, 'usage': {'steps': 0, 'model_calls': 0, 'llm_calls': 0}, 'step_count': 0, 'step_log': [], 'feedback': None, 'pending_clarification': None, 'pending_confirmations': [], 'next_confirmation_id': 1, 'queued_diffs': [], 'clarification_answers': [], 'confirmation_decisions': [], 'notes': [], 'failures': [], 'score_cache': {}, 'baseline': None, 'budget_reason': None, 'request_guard': None, 'status': 'pending', 'provenance': {'model_version': MODEL_VERSION, 'search_version': SEARCH_VERSION, 'policy_version': POLICY_VERSION, 'sleep_count': True, 'decision': decision_provenance}}

def _facts_of(snapshot):
    """Facts summary for staleness (agreement-09 C2): everything that changes
    the evaluated schedule or physiology — excluding the clock/window fields."""
    return {key: snapshot[key] for key in _FACT_KEYS if key in snapshot}

def _segments(snapshot):
    """Bounded coverage segments over the recorded interval (work units only —
    a day is a display unit and no physiological state is reset here)."""
    start, end = (dt(snapshot['now_utc']), dt(snapshot['horizon_end_utc']))
    segments = []
    cursor = start
    index = 0
    while cursor < end:
        right = min(end, cursor + timedelta(hours=SEGMENT_HOURS))
        segments.append({'index': index, 'start_utc': stamp(cursor), 'end_utc': stamp(right), 'status': 'pending', 'decision': None, 'reason': None, 'entry_state': None, 'tried': []})
        cursor = right
        index += 1
    return segments

def _run_baseline(draft):
    if draft['budget'].get('max_model_calls') is not None and draft['usage']['model_calls'] >= draft['budget']['max_model_calls']:
        draft['status'] = 'budget_exhausted'
        draft['budget_reason'] = 'max_model_calls'
        return
    facts = fact_view(draft['snapshot'])
    blocks = deepcopy(draft['plan']['sleep_blocks'])
    result, scoreable = run_evaluation(facts, blocks, sleep_count=True)
    draft['usage']['model_calls'] += 1
    draft['baseline'] = _summary_of(result)
    if draft['decision_mode'] == 'llm':
        from .incremental_agent import _evidence
        draft['model_observation'] = _evidence(result)
    if scoreable:
        report = constraint_report(facts, blocks)
        draft['score_cache'][_cache_key(draft, draft['plan']['edit_sequence'])] = _cache_entry(result, report)
        _refresh_entry_states(draft, result.get('trace'))

def _summary_of(result):
    provenance = result.get('provenance') or {}
    return {'status': result.get('status'), 'utility': result.get('utility'), 'losses': result.get('losses'), 'error': result.get('error'), 'model_version': result.get('model_version'), 'input_hash': provenance.get('input_hash'), 'trace_hash': provenance.get('trace_hash')}

def _cache_entry(result, report):
    provenance = result.get('provenance') or {}
    return {'status': result.get('status'), 'utility': result.get('utility'), 'losses': result.get('losses'), 'model_version': result.get('model_version'), 'input_hash': provenance.get('input_hash'), 'trace_hash': provenance.get('trace_hash'), 'constraint_failed_ids': [c.get('id') for c in report if c.get('status') == 'fail']}

def _cache_key(draft, sequence):
    """Score-cache key (agreement-09 C1): facts digest + evaluation window +
    edit sequence — never the sequence alone."""
    return digest({'facts': draft['facts_digest'], 'window': {'now_utc': draft['snapshot']['now_utc'], 'horizon_end_utc': draft['snapshot']['horizon_end_utc']}, 'edit_sequence': sequence})

def _refresh_entry_states(draft, trace):
    """Planning-owned search seeds (one S per segment start) from a trace.

    These are not model checkpoints and not resumable physiology — only a
    seeding convenience for ``search.generate`` (agreement-09 §1.4)."""
    if not trace:
        return
    rows = [(dt(p['time_utc']), p) for p in trace]
    for segment in draft['segments']:
        start = dt(segment['start_utc'])
        row = None
        for instant, point in rows:
            if instant <= start:
                row = point
            else:
                break
        if row is not None and row.get('S') is not None:
            segment['entry_state'] = {'S': row['S'], 'at_utc': segment['start_utc']}

def _prepare_feedback(draft):
    """Resolve the feedback target to plan blocks (agreement-07 §3 / -05 G5).

    Primary link: a plan block's ``origin_event_id``; defensive link: a
    commitment event's ``target_event_id`` naming a block id. Unresolved
    targets surface ``needs_clarification`` later — never a silent no-op."""
    snapshot = draft['snapshot']
    request = draft['request']
    target = request['target_event_id']
    blocks = draft['plan']['sleep_blocks']
    events = {e.get('id'): e for e in snapshot.get('events', [])}
    linked = [b for b in blocks if b.get('origin_event_id') == target]
    if not linked:
        commitment = events.get(target)
        if commitment is not None and commitment.get('kind') == 'commitment':
            referenced = commitment.get('target_event_id')
            linked = [b for b in blocks if b.get('id') == referenced]
    directive = {'target_event_id': target, 'meaning': request['meaning'], 'reason': request.get('reason'), 'status': 'unresolved', 'linked_block_ids': [], 'target_segment_index': None}
    if linked:
        index = max(0, min((_segment_index_of(draft, dt(b['start_utc'])) for b in linked)))
        protected = any((block_protected(b, snapshot) for b in linked)) or (target in events and event_protected(events[target]))
        directive.update(status='pending_confirmation' if protected else 'pending_apply', linked_block_ids=[b['id'] for b in linked], target_segment_index=index)
        for segment in draft['segments']:
            if segment['index'] < index:
                segment.update(status='processed', decision='carried_unaffected', reason=None)
    draft['feedback'] = directive

def _stop_budget_exhausted(draft, key):
    """Truthfully record a budget exhaustion (msg 313: the step_log and the
    run state carry the exhausted key name) and return the resumable status."""
    draft['budget_reason'] = key
    draft['step_log'].append({'step': draft['step_count'] + 1, 'action': 'budget_exhausted', 'budget_reason': key})
    return 'budget_exhausted'

def _step_once(draft, client=None):
    """One unit of work. Returns ``(terminal_status_or_None, diffs)``."""
    if draft['status'] in TERMINAL_STATUSES:
        return (draft['status'], [])
    diffs = list(draft.get('queued_diffs') or [])
    draft['queued_diffs'] = []
    exhausted = _exhausted_cap(draft)
    if exhausted is not None:
        return (_stop_budget_exhausted(draft, exhausted), diffs)
    try:
        return (_run_step(draft, diffs, client), diffs)
    except Exception as exc:
        draft.setdefault('failures', []).append({'step': draft['step_count'] + 1, 'error': type(exc).__name__ + ': ' + str(exc)})
        return ('failed', diffs)

def _run_step(draft, diffs, client=None):
    draft['step_count'] += 1
    draft['usage']['steps'] += 1
    if draft.get('request_guard') and draft['request_guard'].get('blocked'):
        draft['pending_clarification'] = {'question_id': 'safety-refusal', 'question': draft['request_guard']['message']}
        return 'needs_clarification'
    from .incremental_feedback import require_answer
    if require_answer(draft):
        return 'needs_clarification'
    if draft['decision_mode'] == 'llm':
        from .incremental_agent import agent_step
        return agent_step(draft, diffs, client)
    feedback = draft.get('feedback')
    if feedback is not None:
        if feedback['status'] == 'unresolved':
            draft['pending_clarification'] = {'question_id': 'unresolved-feedback-target', 'question': '事件 ' + feedback['target_event_id'] + ' 没有关联的推荐计划块（origin_event_id 未命中，且该承诺未引用计划块）；请确认事件标识，或改用 initial 重新排程。'}
            return 'needs_clarification'
        if feedback['status'] == 'pending_confirmation':
            draft['pending_confirmations'].append(_build_confirmation(draft))
            return 'needs_confirmation'
        if feedback['status'] == 'pending_apply':
            diffs.extend(_apply_feedback(draft))
    return _work_segments(draft, diffs, client)

def _exhausted_cap(draft):
    """Name of the first exhausted budget key, or None (msg 313: budget_reason
    identifies the exhausted key)."""
    caps = draft['budget']
    if caps.get('max_steps') is not None and draft['usage']['steps'] >= caps['max_steps']:
        return 'max_steps'
    if caps.get('max_model_calls') is not None and draft['usage']['model_calls'] >= caps['max_model_calls']:
        return 'max_model_calls'
    return _llm_budget_exhausted(draft)

def _llm_budget_exhausted(draft):
    caps = draft['budget']
    if caps.get('max_llm_calls') is not None and draft['usage'].get('llm_calls', 0) >= caps['max_llm_calls']:
        return 'max_llm_calls'
    return None

def _budget_reached(draft):
    return _exhausted_cap(draft) is not None

def _work_segments(draft, diffs, client=None):
    pending = [s for s in draft['segments'] if s['status'] == 'pending']
    if not pending:
        return _finalize(draft)
    segment = pending[0]
    facts = fact_view(draft['snapshot'])
    proposals = build_proposals(facts, draft, segment)
    tried = segment.setdefault('tried', [])
    queue = _dedupe([p for p in proposals if proposal_key(p) not in tried])
    if not queue and 'search_seed' not in tried and (not segment_covers(draft['plan']['sleep_blocks'], segment)):
        tried.append('search_seed')
        seed = search_seed_proposal(facts, draft, segment, True)
        if seed is not None:
            queue = [seed]
    if not queue:
        _settle_segment(draft, segment)
        return None
    bounded = queue[:PROPOSALS_PER_STEP]
    decision = _decide(draft, facts, segment, bounded, client)
    if decision is not None:
        if decision['mode'] == 'llm_refusal':
            draft['pending_clarification'] = {'question_id': 'llm-refusal', 'question': decision['refusal']}
            return 'needs_clarification'
        if decision['mode'] == 'llm_budget_exhausted':
            draft['budget_reason'] = 'max_llm_calls'
            return 'budget_exhausted'
        if decision['mode'] == 'llm' and decision['selected_id'] == 'keep':
            for proposal in bounded:
                key = proposal_key(proposal)
                if key not in tried:
                    tried.append(key)
        elif decision['mode'] == 'llm':
            chosen = decision['selected_key']
            bounded = [p for p in bounded if proposal_key(p) == chosen]
    retained = False
    id_assigner = lambda: _next_block_id(draft)
    for proposal in bounded:
        key = proposal_key(proposal)
        if key not in tried:
            tried.append(key)
        candidate, specs = candidate_blocks(draft['plan']['sleep_blocks'], proposal, id_assigner)
        feasible, reasons = hard_validate(facts, candidate)
        report = constraint_report(facts, candidate)
        fails = [c for c in report if c.get('status') == 'fail']
        entry = {'step': draft['step_count'], 'segment': segment['index'], 'source': proposal.get('source'), 'kind': proposal.get('kind'), 'decision_mode': (decision or {}).get('mode'), 'llm_selected': bool(decision and decision.get('mode') == 'llm' and (decision.get('selected_id') != 'keep'))}
        if not feasible or fails:
            entry.update(action='rejected', reasons=(reasons + [str(c.get('id')) for c in fails])[:8])
            draft['step_log'].append(entry)
            continue
        if draft['budget'].get('max_model_calls') is not None and draft['usage']['model_calls'] >= draft['budget']['max_model_calls']:
            return _stop_budget_exhausted(draft, 'max_model_calls')
        result, scoreable = run_evaluation(facts, candidate, sleep_count=True)
        draft['usage']['model_calls'] += 1
        entry.update(action='evaluated', status=result.get('status'), utility=result.get('utility'), input_hash=(result.get('provenance') or {}).get('input_hash'), trace_hash=(result.get('provenance') or {}).get('trace_hash'))
        incumbent = draft['score_cache'].get(_cache_key(draft, draft['plan']['edit_sequence']))
        if scoreable and (incumbent is None or incumbent.get('constraint_failed_ids') or (incumbent.get('utility') is not None and result['utility'] > incumbent['utility'])):
            before = draft['plan']['edit_sequence']
            diffs.extend(_commit(draft, proposal, specs))
            draft['plan']['edit_sequence'] = before + 1
            draft['score_cache'][_cache_key(draft, draft['plan']['edit_sequence'])] = _cache_entry(result, report)
            _refresh_entry_states(draft, result.get('trace'))
            entry['action'] = 'retained'
            retained = True
            if proposal['kind'] == 'add':
                segment.update(status='processed', decision='covered', reason=None)
        else:
            entry['action'] = 'reverted'
        draft['step_log'].append(entry)
    if not retained:
        remaining = [p for p in proposals if proposal_key(p) not in tried]
        seed_possible = 'search_seed' not in tried and (not segment_covers(draft['plan']['sleep_blocks'], segment))
        if not remaining and (not seed_possible):
            _settle_segment(draft, segment)
    return None

def _dedupe(proposals):
    seen = set()
    out = []
    for proposal in proposals:
        key = proposal_key(proposal)
        if key in seen:
            continue
        seen.add(key)
        out.append(proposal)
    return out

def _decide(draft, facts, segment, bounded, client):
    """Explicit diagnostic solver; production LLM runs use Agent tools."""
    if draft['decision_mode'] != 'deterministic':
        raise InvalidRequestError('LLM runs must use the Agent tool loop')
    draft['step_log'].append({'step': draft['step_count'], 'segment': segment['index'], 'decision_mode': 'deterministic', 'degraded': True})
    return None

def _current_score_entry(draft):
    return draft['score_cache'].get(_cache_key(draft, draft['plan']['edit_sequence']))

def _settle_segment(draft, segment):
    entry = _current_score_entry(draft)
    scoreable = entry is not None and entry.get('status') == 'complete'
    if not scoreable:
        segment.update(status='blocked', decision=None, reason='unscoreable_state')
    elif segment_covers(draft['plan']['sleep_blocks'], segment):
        segment.update(status='processed', decision='settled', reason=None)
    elif not _windows_available(draft, segment):
        segment.update(status='processed', decision='no_window', reason=None)
    else:
        segment.update(status='processed', decision='settled_uncovered', reason=None)

def _windows_available(draft, segment):
    from .incremental_proposals import segment_windows
    return bool(segment_windows(fact_view(draft['snapshot']), segment))

def _finalize(draft):
    if draft.get('decision_mode') == 'llm' and draft.get('agent_state'):
        from .incremental_agent import _agent_completion_ready
        if _agent_completion_ready(draft, draft['agent_state'], draft.get('feedback')):
            return 'complete'
        return None
    blocked = [s for s in draft['segments'] if s['status'] == 'blocked']
    entry = _current_score_entry(draft)
    scoreable = entry is not None and entry.get('status') == 'complete'
    if blocked or (draft['segments'] and (not scoreable)):
        note = 'run finished without a scoreable plan state'
        if blocked:
            note += '; blocked segments: ' + ','.join((str(s['index']) for s in blocked))
        draft['notes'].append(note)
        return 'failed'
    return 'complete'

def _apply_feedback(draft):
    """User-directed apply for an unprotected target (user authority; the
    change is applied directly with evaluation evidence recorded)."""
    snapshot = draft['snapshot']
    facts = fact_view(snapshot)
    feedback = draft['feedback']
    id_assigner = lambda: _next_block_id(draft)
    diffs = []
    if feedback['meaning'] == 'omit_occurrence':
        for block_id in list(feedback['linked_block_ids']):
            block = next((b for b in draft['plan']['sleep_blocks'] if b.get('id') == block_id), None)
            if block is None:
                continue
            proposal = {'kind': 'remove', 'source': 'feedback:omit_occurrence', 'event_id': block_id, 'before': deepcopy(block)}
            _, specs = candidate_blocks(draft['plan']['sleep_blocks'], proposal, id_assigner)
            diffs.extend(_commit(draft, proposal, specs))
            draft['omitted_spans'].append({'start_utc': block['start_utc'], 'end_utc': block['end_utc']})
        draft['plan']['edit_sequence'] += 1
        result, scoreable = run_evaluation(facts, deepcopy(draft['plan']['sleep_blocks']), sleep_count=True)
        draft['usage']['model_calls'] += 1
        if scoreable:
            report = constraint_report(facts, draft['plan']['sleep_blocks'])
            draft['score_cache'][_cache_key(draft, draft['plan']['edit_sequence'])] = _cache_entry(result, report)
            _refresh_entry_states(draft, result.get('trace'))
    else:
        for block_id in list(feedback['linked_block_ids']):
            block = next((b for b in draft['plan']['sleep_blocks'] if b.get('id') == block_id), None)
            if block is None:
                continue
            draft['omitted_spans'].append({'start_utc': block['start_utc'], 'end_utc': block['end_utc']})
            segment = _segment_for(draft, dt(block['start_utc']))
            best = None
            first_feasible = None
            for alternative in habitual_proposals(facts, segment, draft['omitted_spans'])[:2]:
                proposal = {'kind': 'replace', 'source': 'feedback:replace', 'event_id': block_id, 'before': deepcopy(block), 'blocks': deepcopy(alternative['blocks'])}
                candidate, specs = candidate_blocks(draft['plan']['sleep_blocks'], proposal, id_assigner)
                feasible, reasons = hard_validate(facts, candidate)
                report = constraint_report(facts, candidate)
                if not feasible or any((c.get('status') == 'fail' for c in report)):
                    draft['step_log'].append({'step': draft['step_count'], 'segment': segment['index'], 'source': 'feedback:replace', 'action': 'rejected', 'reasons': reasons[:8]})
                    continue
                result, scoreable = run_evaluation(facts, candidate, sleep_count=True)
                draft['usage']['model_calls'] += 1
                if scoreable and (best is None or result['utility'] > best[0]['utility']):
                    best = (result, proposal, specs, report)
                elif first_feasible is None:
                    first_feasible = (result, proposal, specs, report)
            if best is not None:
                result, proposal, specs, report = best
                diffs.extend(_commit(draft, proposal, specs))
                draft['plan']['edit_sequence'] += 1
                draft['score_cache'][_cache_key(draft, draft['plan']['edit_sequence'])] = _cache_entry(result, report)
                _refresh_entry_states(draft, result.get('trace'))
            elif first_feasible is not None:
                result, proposal, specs, report = first_feasible
                diffs.extend(_commit(draft, proposal, specs))
                draft['plan']['edit_sequence'] += 1
                draft['notes'].append('feedback replace on ' + block_id + ' applied without a scoreable alternative (user authority)')
            else:
                proposal = {'kind': 'remove', 'source': 'feedback:replace:no_feasible_alternative', 'event_id': block_id, 'before': deepcopy(block)}
                _, specs = candidate_blocks(draft['plan']['sleep_blocks'], proposal, id_assigner)
                diffs.extend(_commit(draft, proposal, specs))
                draft['plan']['edit_sequence'] += 1
                draft['notes'].append('feedback replace on ' + block_id + ' found no feasible alternative; the occurrence was removed')
    for segment in draft['segments']:
        if segment['index'] == feedback['target_segment_index']:
            segment.update(status='processed', decision='feedback_applied', reason=None)
    feedback['status'] = 'applied'
    return diffs

def _build_confirmation(draft):
    """Concrete pending confirmation for a protected/committed target: the
    hypothetical edit is evaluated as evidence but never applied
    automatically."""
    snapshot = draft['snapshot']
    facts = fact_view(snapshot)
    feedback = draft['feedback']
    blocks = draft['plan']['sleep_blocks']
    linked = [b for b in blocks if b.get('id') in set(feedback['linked_block_ids'])]
    proposals = []
    if feedback['meaning'] == 'omit_occurrence':
        for block in linked:
            proposals.append({'kind': 'remove', 'source': 'feedback:omit_occurrence', 'event_id': block['id'], 'before': deepcopy(block), 'requires_confirmation': True})
    else:
        for block in linked:
            draft['omitted_spans'].append({'start_utc': block['start_utc'], 'end_utc': block['end_utc']})
        for block in linked:
            segment = _segment_for(draft, dt(block['start_utc']))
            alternatives = habitual_proposals(facts, segment, draft['omitted_spans'])
            if alternatives:
                replacement = deepcopy(alternatives[0]['blocks'])
            else:
                replacement = []
            for item in replacement:
                if item.get('id') is None:
                    item['id'] = _next_block_id(draft)
            source = 'feedback:replace' if replacement else 'feedback:replace:no_feasible_alternative'
            proposals.append({'kind': 'replace' if replacement else 'remove', 'source': source, 'event_id': block['id'], 'before': deepcopy(block), 'blocks': replacement, 'requires_confirmation': True})
    working = list(blocks)
    for proposal in proposals:
        working, _ = candidate_blocks(working, proposal, lambda: _next_block_id(draft))
    result, _scoreable = run_evaluation(facts, working, sleep_count=True)
    draft['usage']['model_calls'] += 1
    replacement_block = None
    for proposal in proposals:
        for item in proposal.get('blocks') or []:
            replacement_block = item
            break
    confirmation_id = 'conf-%03d' % draft['next_confirmation_id']
    draft['next_confirmation_id'] += 1
    reason = feedback['meaning'] + '; protected/committed target requires explicit confirmation'
    if feedback.get('reason'):
        reason += '; user reason: ' + feedback['reason']
    if feedback['meaning'] == 'replace' and replacement_block is None:
        reason += '; no feasible alternative was found'
    return {'confirmation_id': confirmation_id, 'event_id': linked[0]['id'] if linked else feedback['target_event_id'], 'before': deepcopy(linked[0]) if linked else None, 'after': deepcopy(replacement_block), 'reason': reason, 'proposals': proposals, 'evaluation': _summary_of(result)}

def _execute_confirmation(draft, entry):
    facts = fact_view(draft['snapshot'])
    diffs = []
    for proposal in entry['proposals']:
        _, specs = candidate_blocks(draft['plan']['sleep_blocks'], proposal, lambda: _next_block_id(draft))
        diffs.extend(_commit(draft, proposal, specs))
    draft['plan']['edit_sequence'] += 1
    evaluation = entry.get('evaluation') or {}
    report = constraint_report(facts, draft['plan']['sleep_blocks'])
    draft['score_cache'][_cache_key(draft, draft['plan']['edit_sequence'])] = {'status': evaluation.get('status'), 'utility': evaluation.get('utility'), 'losses': evaluation.get('losses'), 'model_version': evaluation.get('model_version'), 'input_hash': evaluation.get('input_hash'), 'trace_hash': evaluation.get('trace_hash'), 'constraint_failed_ids': [c.get('id') for c in report if c.get('status') == 'fail']}
    feedback = draft.get('feedback')
    if feedback is not None and feedback['status'] == 'pending_confirmation':
        feedback['status'] = 'applied'
        for segment in draft['segments']:
            if segment['index'] == feedback['target_segment_index']:
                segment.update(status='processed', decision='feedback_applied', reason=None)
    return diffs

def _next_block_id(draft):
    number = draft['plan']['next_block_id']
    existing = {b['id'] for b in draft['plan']['sleep_blocks']}
    while 'rec-%04d' % number in existing:
        number += 1
    draft['plan']['next_block_id'] = number + 1
    return 'rec-%04d' % number

def _commit(draft, proposal, specs):
    """Apply committed specs to the draft plan and produce the wire diffs.

    Plan-block diffs carry ``target='sleep_block'`` and no revision comparison
    (``from_revision``/``to_revision`` are None — the draft holds the prior
    block; agreement-04 §1). Block ids are stable across moves."""
    blocks = draft['plan']['sleep_blocks']
    diffs = []
    for op, event_id, before, after in specs:
        if op == 'add':
            block_id = after.get('id') or _next_block_id(draft)
            block = dict(after, id=block_id)
            blocks.append(block)
            draft['plan']['provenance'][block_id] = 'confirmed' if proposal.get('requires_confirmation') else 'generated'
            diffs.append(_sleep_block_diff(draft, 'add', block_id, None, block, proposal))
        elif op == 'move':
            block = next((b for b in blocks if b.get('id') == event_id))
            block.update({'start_utc': after['start_utc'], 'end_utc': after['end_utc']})
            if after.get('timezone'):
                block['timezone'] = after['timezone']
            if after.get('role'):
                block['role'] = after['role']
            diffs.append(_sleep_block_diff(draft, 'move', event_id, before, block, proposal))
        elif op == 'remove':
            block = next((b for b in blocks if b.get('id') == event_id))
            blocks.remove(block)
            draft['plan']['provenance'].pop(event_id, None)
            diffs.append(_sleep_block_diff(draft, 'remove', event_id, before, None, proposal))
        else:
            raise ValueError('unknown diff op: ' + str(op))
    return diffs

def _sleep_block_diff(draft, op, event_id, before, after, proposal):
    return {'event_id': event_id, 'op': op, 'from_revision': None, 'to_revision': None, 'before': deepcopy(before), 'after': deepcopy(after), 'target': 'sleep_block', 'affects_committed': bool(proposal.get('requires_confirmation')), 'reason': proposal.get('source')}

def _apply_input(draft, input):
    if not isinstance(input, dict):
        raise InvalidRequestError('invalid_request: input must be a dict')
    unknown = sorted(set(input) - {'kind', 'text', 'confirmation_id', 'decision'})
    if unknown:
        raise InvalidRequestError('invalid_request: input.' + unknown[0] + ' is not accepted')
    kind = input.get('kind')
    if kind == 'answer':
        if draft['status'] != 'needs_clarification' or not draft.get('pending_clarification'):
            raise InvalidRequestError('invalid_request: input.kind answer requires a pending clarification')
        text = input.get('text')
        if not isinstance(text, str) or not text.strip():
            raise InvalidRequestError('invalid_request: input.text must be a non-empty string')
        from .incremental_feedback import accept_answer
        accept_answer(draft, draft['pending_clarification']['question_id'], text)
        draft['clarification_answers'].append({'question_id': draft['pending_clarification']['question_id'], 'text': text})
        draft['pending_clarification'] = None
        question_id = draft['clarification_answers'][-1]['question_id']
        if question_id == 'safety-refusal' or question_id == 'llm-refusal':
            if draft.get('feedback') and draft['feedback']['status'] in ('pending_apply', 'pending_confirmation', 'unresolved', 'applied'):
                draft['feedback']['status'] = 'no_op_refused'
            draft['notes'].append('refusal (question_id=' + question_id + ') closed as a documented no-op after the clarification answer')
        elif draft.get('feedback') and draft['feedback']['status'] == 'unresolved':
            draft['feedback']['status'] = 'no_op_answered'
            draft['notes'].append('feedback target ' + draft['feedback']['target_event_id'] + ' unresolved; closed as a documented no-op after the clarification answer')
        draft['status'] = 'pending'
        return
    if kind == 'confirmation':
        if draft['status'] != 'needs_confirmation' or not draft['pending_confirmations']:
            raise InvalidRequestError('invalid_request: input.kind confirmation requires a pending confirmation')
        confirmation_id = input.get('confirmation_id')
        decision = input.get('decision')
        if not isinstance(confirmation_id, str) or not confirmation_id:
            raise InvalidRequestError('invalid_request: input.confirmation_id must be a non-empty string')
        if decision not in CONFIRMATION_DECISIONS:
            raise InvalidRequestError("invalid_request: input.decision must be 'apply' or 'reject'")
        entry = next((c for c in draft['pending_confirmations'] if c['confirmation_id'] == confirmation_id), None)
        if entry is None:
            raise InvalidRequestError('invalid_request: input.confirmation_id ' + confirmation_id + ' does not match a pending confirmation')
        draft['pending_confirmations'] = [c for c in draft['pending_confirmations'] if c['confirmation_id'] != confirmation_id]
        draft['confirmation_decisions'].append({'confirmation_id': confirmation_id, 'decision': decision})
        if entry.get('agent_batch_id'):
            from .incremental_agent import apply_confirmation
            draft['queued_diffs'].extend(apply_confirmation(draft, entry, decision))
            draft['status'] = 'needs_confirmation' if draft['pending_confirmations'] else 'pending'
            return
        if decision == 'apply':
            draft['queued_diffs'].extend(_execute_confirmation(draft, entry))
        else:
            if draft.get('feedback') and draft['feedback']['status'] == 'pending_confirmation':
                draft['feedback']['status'] = 'no_op_rejected'
            draft['notes'].append('confirmation ' + confirmation_id + ' rejected; the concrete diff was not applied')
        draft['status'] = 'pending'
        return
    raise InvalidRequestError("invalid_request: input.kind must be 'answer' or 'confirmation'")

def _apply_budget_extension(draft, extension):
    if not isinstance(extension, dict):
        raise InvalidRequestError('invalid_request: budget_extension must be a dict')
    unknown = sorted(set(extension) - {'max_steps', 'max_model_calls', 'max_llm_calls'})
    if unknown:
        raise InvalidRequestError('invalid_request: budget_extension.' + unknown[0] + ' is not accepted')
    for key in ('max_steps', 'max_model_calls', 'max_llm_calls'):
        if key not in extension:
            continue
        value = extension[key]
        if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 0):
            raise InvalidRequestError('invalid_request: budget_extension.' + key + ' must be a non-negative integer or None')
        draft['budget'][key] = value
    draft['step_log'].append({'step': draft['step_count'] + 1, 'action': 'budget_extension', 'budget': deepcopy(draft['budget'])})
    if draft['status'] == 'budget_exhausted' and _exhausted_cap(draft) is None:
        draft['budget_reason'] = None
        draft['status'] = 'pending'

def _sync_snapshot(draft, snapshot):
    """Facts-change vs clock-advance staleness (agreement-09 C2)."""
    _validate_snapshot(snapshot)
    if digest(_facts_of(snapshot)) != draft['facts_digest']:
        raise StaleDraftError('stale draft: input facts changed (events/profile/sessions/preferences/scheduling_policy/revision); start a new run for the new input')
    now = dt(snapshot['now_utc'])
    base_now = dt(draft['snapshot']['now_utc'])
    if now < base_now:
        raise StaleDraftError('stale draft: now_utc moved backwards from ' + draft['snapshot']['now_utc'] + ' to ' + snapshot['now_utc'])
    if now == base_now and dt(snapshot['horizon_end_utc']) == dt(draft['snapshot']['horizon_end_utc']):
        return
    _rebase(draft, snapshot)

def _rebase(draft, snapshot):
    """Clock advance: re-seed segments from the new now, keep the plan blocks,
    and re-evaluate from the new now (the model always simulates from now)."""
    old_now = draft['snapshot']['now_utc']
    lost = [s['index'] for s in draft['segments'] if s['status'] == 'pending']
    draft['snapshot'] = deepcopy(snapshot)
    draft['segments'] = _segments(snapshot)
    if draft.get('agent_state'):
        draft['agent_state'].update(trial=None, reads=[], submissions=[], last_evaluation=None, planning_issues=[], observation={'status': 'clock_rebased', 'now_utc': snapshot['now_utc']})
        draft['agent_state']['last_comparison'] = None
        draft['agent_state']['last_evaluated_plan_version'] = None
        draft['agent_state']['retained_edit_ranges'] = []
        draft['agent_state'].pop('confirmation_trial', None)
    if draft['pending_confirmations']:
        draft['pending_confirmations'] = []
        draft['notes'].append('pending confirmations invalidated by the clock rebase; they are re-derived')
    draft['status'] = 'pending'
    if draft['budget'].get('max_model_calls') is not None and draft['usage']['model_calls'] >= draft['budget']['max_model_calls']:
        draft['status'] = 'budget_exhausted'
        draft['budget_reason'] = 'max_model_calls'
        return
    result, scoreable = run_evaluation(fact_view(draft['snapshot']), deepcopy(draft['plan']['sleep_blocks']), sleep_count=True)
    draft['usage']['model_calls'] += 1
    if scoreable:
        report = constraint_report(fact_view(draft['snapshot']), draft['plan']['sleep_blocks'])
        draft['score_cache'][_cache_key(draft, draft['plan']['edit_sequence'])] = _cache_entry(result, report)
        _refresh_entry_states(draft, result.get('trace'))
    if draft['decision_mode'] == 'llm':
        from .incremental_agent import _evidence
        draft['model_observation'] = _evidence(result)
    note = 'clock rebase: now_utc ' + old_now + ' -> ' + snapshot['now_utc'] + '; segments re-seeded from the new now; plan blocks kept'
    if lost:
        note += '; segments elapsed unprocessed: ' + ','.join((str(i) for i in lost))
    draft['notes'].append(note)
    draft['step_log'].append({'action': 'rebase', 'now_utc': snapshot['now_utc'], 'horizon_end_utc': snapshot['horizon_end_utc']})

def _pending_needs(draft):
    clarification = None
    if draft.get('pending_clarification'):
        clarification = {'question_id': draft['pending_clarification']['question_id'], 'question': draft['pending_clarification']['question']}
    confirmations = None
    if draft['pending_confirmations']:
        confirmations = [{'confirmation_id': c['confirmation_id'], 'event_id': c['event_id'], 'before': deepcopy(c['before']), 'after': deepcopy(c['after']), 'reason': c['reason']} for c in draft['pending_confirmations']]
    return {'clarification': clarification, 'confirmations': confirmations}

def _score_summary(draft):
    entry = draft['score_cache'].get(_cache_key(draft, draft['plan']['edit_sequence']))
    if entry is None:
        return None
    return {'utility': entry.get('utility'), 'losses': entry.get('losses'), 'model_version': entry.get('model_version') or draft['provenance']['model_version'], 'policy_version': draft['provenance']['policy_version'], 'search_version': draft['provenance']['search_version']}

def _build_result(draft, step_status, event_diffs):
    report = constraint_report(fact_view(draft['snapshot']), draft['plan']['sleep_blocks'])
    return {'step_index': draft['step_count'], 'step_status': step_status, 'event_diffs': deepcopy(event_diffs), 'constraint_summary': summarize_constraints(report), 'score_summary': _score_summary(draft), 'needs': _pending_needs(draft), 'draft_digest': digest(draft), 'run_state': deepcopy(draft)}

def _segment_index_of(draft, instant):
    segments = draft['segments']
    if not segments:
        return 0
    for segment in segments:
        if dt(segment['start_utc']) <= instant < dt(segment['end_utc']):
            return segment['index']
    return 0 if instant < dt(segments[0]['start_utc']) else segments[-1]['index']

def _segment_for(draft, instant):
    index = _segment_index_of(draft, instant)
    return draft['segments'][index]
