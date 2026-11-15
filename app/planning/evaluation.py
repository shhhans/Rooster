"""Planning-owned scoring and evaluation orchestration.

All biological computation comes from the public model API
(``app.sleep_model.api.compute``); the constraint report, the execution audit,
the preference loss, the fixed need/excess buckets, the product weights, the
utility and the sleep-count compensation are planning policy. No physiology is
copied here and no model-private internal is reached.

The legacy import paths (``app.sleep_model.evaluate``,
``app.sleep_model.evaluation``, ``app.sleep_model.adapter``,
``app.sleep_model.preferences``, ``app.sleep_model.sleep_count``) are lazy
compatibility bridges to this module and its siblings; their formulas,
defaults, output dictionaries, hashes and error strings are unchanged (module
decoupling contract step 3).
"""
from copy import deepcopy

from app.contracts.time import dt, stamp
from app.sleep_model import api
from app.sleep_model.contracts import SCENARIOS, VERSION, digest

from .policy import planning_configuration
from .preferences import evaluate_preferences, sleep_targets
from .sleep_count import sleep_count_cost

# The public model result appends these three documented physiology assumption
# strings; the legacy path carried its own trio (see ``_legacy_assumptions``).
PUBLIC_ASSUMPTIONS = (
    'unobserved history filled with habit prior',
    'unmeasured light uses declared photopic scenario',
    'latency and inertia are engineering assumptions; outputs are not fitness-for-duty judgments',
)
LEGACY_ASSUMPTIONS = (
    'unobserved history filled with habit prior',
    'unmeasured light uses declared photopic scenario',
    'latency and inertia are engineering assumptions; score is not fitness for duty',
)


def _legacy_assumptions(model_assumptions):
    """Rebuild the legacy assumption list from the public model result.

    The public ``compute`` result ends with the three documented physiology
    assumption strings and, when the prior did not converge, a trailing
    ``prior_not_converged``; the legacy list is the core assumptions, then
    ``prior_not_converged`` (when present), then the legacy trio. The three
    public strings are therefore removed by identity wherever they sit and the
    legacy trio is appended, which restores the legacy list and its order.
    """
    kept = [item for item in model_assumptions if item not in PUBLIC_ASSUMPTIONS]
    return kept + list(LEGACY_ASSUMPTIONS)


def check_constraints(snapshot, blocks, start, end):
    """Use the main implementation's authoritative scheduling policy."""
    from .engine import check_plan
    snap = deepcopy(snapshot)
    snap['now_utc'] = stamp(start)
    snap['horizon_end_utc'] = stamp(end)
    return check_plan(snap, blocks)


def _window(snapshot, start_utc, end_utc):
    """Resolve the evaluation window exactly like the legacy path did.

    The public model API validates the snapshot before it runs and reports a
    missing field with its own message; the legacy path raised ``KeyError``
    for the same inputs. Resolving the legacy bounds here keeps the legacy
    error strings (``'now_utc'``, ``'horizon_end_utc'``,
    ``timezone-aware instant required``) and the legacy window semantics.
    """
    now = dt(snapshot['now_utc'])
    start = dt(start_utc or now)
    end = dt(end_utc or snapshot['horizon_end_utc'])
    return now, start, end


def _evaluate(snapshot, blocks, *, start_utc, end_utc, mode, preferences, config, sleep_count=True):
    physio, policy, full = config
    now, start, end = _window(snapshot, start_utc, end_utc)
    core = api.compute(snapshot, blocks, start_utc=stamp(start), end_utc=stamp(end),
                        mode=mode, config=physio)
    snap = core['normalized']['snapshot']
    opportunities = core['normalized']['sleep_blocks']
    actual = core['actual_sleep']
    measured = core['effective_sleep']
    predicted = core['predicted_sleep']
    trace = core['trace']
    state = core['initial_state']
    final = core['final_state']
    metrics = dict(core['physiological_metrics'])
    phys = core['physiological_loss']
    initialization = core['provenance']['initialization']

    requested_preferences = deepcopy(preferences if preferences is not None else snapshot.get('preferences', []))
    full_window = start == now and end == dt(snapshot['horizon_end_utc'])
    if full_window:
        resolved_preferences = requested_preferences
        current_opportunities = opportunities
        previous = (snapshot.get('active_plan') or {}).get('sleep_blocks', [])
    else:
        resolved_preferences = []
        for preference in requested_preferences:
            p_start, p_end = dt(preference['scope_start_utc']), dt(preference['scope_end_utc'])
            if p_start < end and p_end > start:
                preference['scope_start_utc'] = stamp(max(start, p_start))
                preference['scope_end_utc'] = stamp(min(end, p_end))
                resolved_preferences.append(preference)
        current_opportunities = [b for b in opportunities
                                 if dt(b['start_utc']) < end and dt(b['end_utc']) > start]
        previous = [b for b in (snapshot.get('active_plan') or {}).get('sleep_blocks', [])
                    if dt(b['start_utc']) < end and dt(b['end_utc']) > start]
    pref = evaluate_preferences(resolved_preferences, current_opportunities, measured, previous)
    buckets, need, excess = sleep_targets(measured, start, end, snap['profile'].get('sleep_need_hours', 8.))
    metrics['sleep_target_buckets'] = buckets
    count = sleep_count_cost(actual + measured, start, end,
                             threshold=policy['sleep_count_threshold'],
                             coefficient=policy['sleep_count_coefficient'] if sleep_count else 0.)
    metrics['sleep_count_cost'] = count
    # Constraints remain whole-plan checks even when the Agent selects a local
    # scoring range. This prevents out-of-range future blocks from appearing
    # invalid merely because the scoring endpoint is shorter than the horizon.
    checks = check_constraints(snap, opportunities, now, dt(snapshot['horizon_end_utc'])) if mode == 'forecast' else [
        {'id': 'aviation_legal_compliance', 'status': 'not_evaluated'}]
    if mode == 'retrospective':
        from .audit import check_execution
        execution = check_execution(snap, stamp(start), stamp(end))
        checks = execution['checks'] + checks
    w = policy['rank_weights']
    total = w[0] * phys + w[1] * need + w[2] * pref['loss'] + w[3] * excess if phys is not None else None
    if total is not None:
        total += count['points'] / 100
    if not initialization.get('prior_converged', True):
        total = None
    assumptions = _legacy_assumptions(core['assumptions'])
    status = 'complete' if total is not None else 'partial'
    conditional = any(b.get('conditional') for b in opportunities) or bool(snapshot.get('proposed_event_changes'))
    return {
        'status': status, 'scorer_version': VERSION, 'model_version': VERSION,
        'mode': mode,
        'publishability': 'invalid' if any(c['status'] == 'fail' for c in checks) else 'conditional' if conditional else 'valid',
        'utility': 100 * (1 - total) if total is not None else None,
        'display_score': 100 * (1 - total) if total is not None else None,
        'losses': {'physiological': phys, 'need': need, 'preference': pref['loss'], 'excess': excess,
                   'sleep_count': count['points'] / 100, 'total': total},
        'physiological_metrics': metrics, 'preference_metrics': pref,
        'resolved_preferences': resolved_preferences, 'predicted_sleep': predicted,
        'missed_opportunities': [b['id'] for b in opportunities
                                 if not any(p['opportunity_id'] == b['id'] for p in predicted)] if mode == 'forecast' else [],
        'constraint_report': checks, 'trace': trace, 'initial_state': state, 'final_state': final,
        'assumptions': assumptions, 'initialization': initialization,
        'evaluation_start_utc': stamp(start), 'evaluation_end_utc': stamp(end),
        'provenance': {
            'model_version': VERSION, 'configuration': full, 'configuration_hash': digest(full),
            'snapshot_hash': core['provenance']['snapshot_hash'],
            'input_hash': digest({'snapshot': snapshot, 'opportunities': opportunities,
                                  'preferences': resolved_preferences, 'mode': mode,
                                  'start_utc': stamp(start), 'end_utc': stamp(end)}),
            'opportunity_hash': core['provenance']['opportunity_hash'],
            'trace_hash': core['provenance']['trace_hash'],
            'initialization': initialization, 'equation_source': 'sources/circadian/manifest.json'},
    }


def evaluate(snapshot, sleep_blocks=None, *, start_utc=None, end_utc=None, mode='forecast',
             preferences=None, scenarios=True, config=None, sleep_count=None):
    """Score a candidate schedule.

    ``sleep_count`` is the explicit compensation switch: ``None`` (default) keeps
    the legacy behaviour (the sleep-count compensation is enabled), ``False``
    removes only the count contribution from the total loss. The switch is an
    evaluation-time choice, not a configuration change: the reported
    configuration and its hash keep describing the configured parameters.
    """
    try:
        resolved = planning_configuration(config)
        enabled = True if sleep_count is None else bool(sleep_count)
        result = _evaluate(deepcopy(snapshot), deepcopy(sleep_blocks or []), start_utc=start_utc,
                           end_utc=end_utc, mode=mode, preferences=preferences, config=resolved,
                           sleep_count=enabled)
        result['scenarios'] = []
        if scenarios:
            for name, changes in SCENARIOS.items():
                if name == 'nominal':
                    continue
                # Legacy scenario rule: the scenario configuration is built from
                # the resolved caller configuration, so the caller's policy
                # carries into every scenario and each scenario publishes its
                # own configuration hash.
                changed = planning_configuration({**resolved[2], **changes})
                item = _evaluate(deepcopy(snapshot), deepcopy(sleep_blocks or []), start_utc=start_utc,
                                 end_utc=end_utc, mode=mode, preferences=preferences, config=changed,
                                 sleep_count=enabled)
                result['scenarios'].append({'name': name, 'utility': item['utility'], 'losses': item['losses'],
                                            'initial_state': item['initial_state'],
                                            'configuration_hash': digest(changed[2]),
                                            'predicted_sleep': item['predicted_sleep'], 'status': item['status']})
            values = [result['utility']] + [s['utility'] for s in result['scenarios']]
            values = [v for v in values if v is not None]
            result['sensitivity_range'] = [min(values), max(values)] if values else None
        return result
    except (ValueError, KeyError, TypeError) as exc:
        return {'status': 'invalid_input', 'utility': None, 'scorer_version': VERSION, 'error': str(exc),
                'publishability': 'invalid', 'trace': [], 'scenarios': []}
    except ArithmeticError as exc:
        return {'status': 'numerical_failure', 'utility': None, 'scorer_version': VERSION, 'error': str(exc),
                'publishability': 'unknown', 'trace': [], 'scenarios': []}


def evaluate_many(snapshot, candidates, **kwargs):
    return [evaluate(snapshot, blocks, **kwargs) for blocks in candidates]
