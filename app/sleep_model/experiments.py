"""Traceable experiment entry reusing the formal compute path.

Experiments require an explicit identity (``experiment_id``, ``config``,
``components``), reuse ``app.sleep_model.api.compute`` verbatim, and never
mutate production defaults: every run resolves a fresh configuration from
deep-copied defaults. The implementation registry is closed — only the
current production implementation is registered, and unknown implementation
names fail loudly. Experiment results are pure compute results plus an
``experiment`` identity block; no product policy keys are added.
"""
from . import api
from .contracts import VERSION, digest
from .melatonin import _declared_experiment_configuration, melatonin_configuration

IMPLEMENTATIONS = {
    'current': {
        'implementation': 'current',
        'description': 'production physiological model through app.sleep_model.api.compute',
        'model_version': VERSION,
        'api_version': api.API_VERSION,
        'compute': api.compute,
    },
}


def run(experiment_id, snapshot, sleep_blocks=None, *, implementation='current', config,
        components, start_utc=None, end_utc=None, mode='forecast', melatonin_config=None,
        melatonin_summaries=None, melatonin_crossings=False):
    """Run one traceable experiment through the formal compute path.

    ``experiment_id`` must be an explicit non-empty string; ``config`` must be
    an explicit dict of physiological parameters (``{}`` = production
    defaults, never a mutation of them) and ``components`` an explicit
    selection (see ``app.sleep_model.api.resolve_components``). The result is
    a ``compute`` result plus ``experiment``: a deterministic, traceable
    experiment identity (id, implementation, versions, components and hashes).

    ``melatonin_config`` is the explicit, versioned melatonin parameter
    object. It is the **only** channel through which the non-published
    ``a = 0`` analytic-clearance value may flow: the resolved configuration is
    wrapped as an explicit declared-experiment configuration here, so the
    production paths keep their positive default and no user-facing
    zero-synthesis switch exists.

    ``melatonin_summaries`` and ``melatonin_crossings`` are forwarded to the
    compute path unchanged (additive keyword-only, defaults ``None``/``False``),
    so an experiment and the production compute share one request-kwarg
    vocabulary.
    """
    if not isinstance(experiment_id, str) or not experiment_id.strip():
        raise ValueError('explicit non-empty experiment_id required')
    if config is None:
        raise ValueError('explicit experiment config required (pass {} for production defaults)')
    if not isinstance(config, dict):
        raise ValueError('experiment config must be a dict of physiological parameters')
    if components is None:
        raise ValueError('explicit experiment components required')
    if implementation not in IMPLEMENTATIONS:
        raise ValueError('unknown experiment implementation: ' + str(implementation)
                         + '; known: ' + ','.join(sorted(IMPLEMENTATIONS)))
    entry = IMPLEMENTATIONS[implementation]
    resolved_melatonin = None
    if melatonin_config is not None:
        if not isinstance(melatonin_config, dict):
            raise ValueError('melatonin_config must be a dict of melatonin parameters')
        resolved_melatonin = _declared_experiment_configuration(
            melatonin_configuration(melatonin_config))
    result = entry['compute'](snapshot, sleep_blocks, start_utc=start_utc, end_utc=end_utc,
                              mode=mode, config=config, components=components,
                              melatonin_config=resolved_melatonin,
                              melatonin_summaries=melatonin_summaries,
                              melatonin_crossings=melatonin_crossings)
    identity = {
        'experiment_id': experiment_id,
        'implementation': implementation,
        'model_version': entry['model_version'],
        'api_version': entry['api_version'],
        'requested_components': result['components']['requested'],
        'resolved_components': result['components']['resolved'],
        'model_config_hash': result['provenance']['model_config_hash'],
        'input_hash': result['provenance']['input_hash'],
        'experiment_hash': digest({'experiment_id': experiment_id,
                                   'implementation': implementation,
                                   'model_config_hash': result['provenance']['model_config_hash'],
                                   'input_hash': result['provenance']['input_hash'],
                                   'requested_components': result['components']['requested']}),
    }
    if 'melatonin' in result['components']['resolved']:
        identity['melatonin_model_config_hash'] = \
            result['melatonin']['provenance']['melatonin_config_hash']
        identity['experiment_hash'] = digest({'experiment_id': experiment_id,
                                              'implementation': implementation,
                                              'model_config_hash': result['provenance']['model_config_hash'],
                                              'input_hash': result['provenance']['input_hash'],
                                              'requested_components': result['components']['requested'],
                                              'melatonin_model_config_hash':
                                                  identity['melatonin_model_config_hash']})
    result['experiment'] = identity
    return result
