"""Explicit opt-in melatonin diagnostics (planning side).

This module is a thin adapter over the public model API only. Planning asks
the model for interval summaries explicitly and propagates the model's answer
verbatim: the modelled plasma, phase and synthesis summaries for the requested
candidate intervals, with their units, validity, assumptions and provenance.

What this module deliberately does NOT do:

* it computes no melatonin physiology and copies no equation (the Breslow13
  transcription belongs to the model group);
* it introduces no physiological policy (no required/soft/hard matrix, no
  clinical threshold, no dosing logic) and no planning-quality claim;
* it never clamps a value and never reinterprets a validity marker: model
  failures are raised to the caller, model-reported domain facts are
  propagated unchanged;
* it never changes the legacy dictionaries, scores, ranking or hashes: the
  diagnostic lives behind its own explicit selection, so the absent path is
  untouched.

The frozen cross-group interface is recorded in
``groups/planning/tasks/DECOUPLE-PLANNING-01/interface-agreement.md``:
``compute(..., components=['melatonin'], melatonin_summaries=[{start_utc,
end_utc, id}], melatonin_crossings=...)`` returns the block in
``result['melatonin']``; the request keywords are isolated in the constants
below so that a coordinated rename touches a single call site.

The model revision this consumption is checked against is
``e1d3ae1f5e34d70a8df1563720d60f0c33d5b9dd`` (branch
``codex/sleep-model/MELATONIN-MODEL-01``). Its delivered semantics:

* every requested summary always carries a ``crossings`` key — an empty list
  when the crossings were not requested or when none fell inside the interval;
* the attribution of a crossing to an interval is model-owned and half-open
  (``interval.start_utc <= at_utc < interval.end_utc``), so a record
  legitimately appears in every requested interval that contains it; planning
  never re-filters or re-selects locally;
* the trajectory-wide ``result['melatonin']['crossings']`` list is purely
  additive and the per-summary lists are filtered copies of it.

Crossings handling in this adapter follows that contract exactly: with the
opt-in **enabled** the per-summary crossings are propagated verbatim —
including an empty list, which is the model's own statement that crossings
were computed and none fell inside that interval — and only an **absent** key
(which the delivered model never produces) is marked explicitly. With the
opt-in **disabled** the model's crossings key is a placeholder, not a
statement: this adapter presents no crossings field at all and records the
opt-in state explicitly (``crossings_requested``), so an empty list can never
be read as "none occurred".

``config`` is the planning-owned configuration surface: it is resolved through
``app.planning.policy.public_planning_configuration`` and only its
physiological subset is forwarded to the model, so a product-policy key can
never surface the model's ``unknown physiological parameters`` message from a
planning-owned API.

Planning passes no model parameter overrides (product policy must never leak
into model parameters) and no sampling, light or model-state arguments: the
model derives the sampling grid, the light samples and the model state itself,
and the snapshot is passed through untouched.

Only the integrated path is used here; the standalone model surface uses
different argument names and is not called from planning.
"""
from app.contracts.time import dt
from app.sleep_model import api

from .policy import public_planning_configuration

DIAGNOSTIC = 'melatonin'

#: Model component selection for this diagnostic.
COMPONENTS = [DIAGNOSTIC]
#: Request keyword carrying the explicitly requested interval summaries.
SUMMARIES_KEYWORD = 'melatonin_summaries'
#: Request keyword for the opt-in synthesis crossings (default off).
CROSSINGS_KEYWORD = 'melatonin_crossings'

NOTES = (
    'modeled melatonin diagnostics only: no physiological policy, clinical threshold, '
    'dosing logic or planning-quality claim; the modelled synthesis crossings are '
    'not a measured DLMO',
)


def _intervals(melatonin_summaries):
    """Validate the explicit selection and echo the requested intervals verbatim."""
    if not isinstance(melatonin_summaries, (list, tuple)) or not melatonin_summaries:
        raise ValueError('explicit melatonin_summaries selection required: a non-empty list of '
                         "{'id', 'start_utc', 'end_utc'} interval requests")
    requests = []
    for item in melatonin_summaries:
        if not isinstance(item, dict):
            raise ValueError('melatonin summary request must be a dict with id, start_utc and end_utc')
        missing = [key for key in ('id', 'start_utc', 'end_utc') if key not in item]
        if missing:
            raise ValueError('melatonin summary request missing: ' + ','.join(missing))
        dt(item['start_utc'])
        dt(item['end_utc'])
        requests.append({'id': item['id'], 'start_utc': item['start_utc'], 'end_utc': item['end_utc']})
    return requests


def run(snapshot, sleep_blocks=None, *, melatonin_summaries=None, melatonin_crossings=False,
        start_utc=None, end_utc=None, mode='forecast', config=None):
    """Return the modelled melatonin diagnostics for explicitly requested intervals.

    ``melatonin_summaries`` is the explicit opt-in: a non-empty list of
    ``{'id', 'start_utc', 'end_utc'}`` interval requests. Planning never
    auto-derives intervals from opportunities; the caller states which
    candidate intervals to summarise. The ``id`` is echoed back verbatim.

    Model failures are raised to the caller unchanged (``ValueError`` /
    ``TypeError`` for caller-contract violations, ``ArithmeticError`` for a
    total solver integration failure); solver-reported domain facts (negative
    or non-finite concentrations, an out-of-range photoreceptor state,
    suppression-guard activations) stay validity markers and are propagated
    verbatim.

    ``config`` accepts the planning configuration (physiological parameters and
    planning policy keys); only its physiological subset reaches the model, so
    a policy key never surfaces the model's ``unknown physiological
    parameters`` message from this planning-owned surface. Unknown keys raise
    ``unknown planning parameters: ...``.
    """
    requests = _intervals(melatonin_summaries)
    physio = None if config is None else public_planning_configuration(config)[0]
    result = api.compute(snapshot, sleep_blocks, start_utc=start_utc, end_utc=end_utc, mode=mode,
                         config=physio, components=list(COMPONENTS),
                         **{SUMMARIES_KEYWORD: requests, CROSSINGS_KEYWORD: bool(melatonin_crossings)})
    block = result.get('melatonin')
    if block is None:
        raise RuntimeError('the public model API returned no melatonin diagnostics for the '
                           'requested selection')
    requested_crossings = bool(melatonin_crossings)
    if not requested_crossings:
        # The model's crossings key is a placeholder when the opt-in is off:
        # planning presents no crossings field in that case and records the
        # opt-in state explicitly in the envelope, so an empty list can never
        # be read as "no crossings occurred". Only the placeholder key is
        # omitted; no value is altered and no empty list is synthesised.
        block = {key: value for key, value in block.items() if key != 'crossings'}
        summaries = block.get('summaries')
        if isinstance(summaries, list):
            block['summaries'] = [{key: value for key, value in item.items() if key != 'crossings'}
                                  if isinstance(item, dict) else item for item in summaries]
    summaries = {item.get('id'): item for item in (block.get('summaries') or [])
                 if isinstance(item, dict)}
    unavailable = []
    if requested_crossings:
        # The per-summary crossings are the primary form and are propagated
        # verbatim, including an empty list: that is the model's own statement
        # ("computed, none inside this interval"). Only an ABSENT key -- which
        # the delivered model never produces -- is marked explicitly. Planning
        # never derives crossings from the additive trajectory-wide list and
        # never re-selects the model's intervals locally.
        unavailable = [request['id'] for request in requests
                       if not isinstance(summaries.get(request['id']), dict)
                       or 'crossings' not in summaries[request['id']]]
    notes = list(NOTES)
    if unavailable:
        notes.append('the model response did not carry per-summary crossings for the requested '
                     'intervals ' + ','.join(str(item) for item in unavailable) + '; planning does '
                     'not derive crossings from the trajectory-wide list')
    envelope = {
        'diagnostic': DIAGNOSTIC,
        'selection': {SUMMARIES_KEYWORD: requests, CROSSINGS_KEYWORD: requested_crossings},
        # The opt-in state is explicit: the model's crossings key is a
        # statement only when the crossings were requested.
        'crossings_requested': requested_crossings,
        'api_version': result.get('api_version'), 'model_version': result.get('model_version'),
        'mode': result.get('mode'), 'status': result.get('status'),
        'knowledge_cutoff_utc': result.get('knowledge_cutoff_utc'),
        'evaluation_start_utc': result.get('evaluation_start_utc'),
        'evaluation_end_utc': result.get('evaluation_end_utc'),
        'melatonin': block,
        'notes': notes,
    }
    if requested_crossings:
        # The additive trajectory-wide crossing list of the model, kept
        # separate from the per-summary crossings (the primary form) and
        # optional: it is propagated unchanged when the model provides it and
        # is absent otherwise (an absent field never implies "no crossings
        # occurred").
        if 'crossings' in block:
            envelope['trajectory_crossings'] = block['crossings']
        # Requested intervals for which the model response carries no
        # per-summary crossings at all (defensive: the delivered model always
        # provides the key).
        envelope['crossings_unavailable'] = unavailable
    return envelope
