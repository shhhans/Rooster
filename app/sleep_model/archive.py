"""Portable complete model archives and score-only reweighting without an ODE/LLM."""
from copy import deepcopy
import json
from pathlib import Path
import platform
import sys
import numpy
import scipy
from .contracts import VERSION,configuration,digest


def archive_evaluation(path,snapshot,blocks,result):
    root=Path(path)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError('model archive is immutable; choose a new directory: '+str(root))
    root.mkdir(parents=True,exist_ok=True)
    module=Path(__file__).parent
    def write(name,value):
        (root/name).write_text(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False),encoding='utf-8')
    write('model_inputs.json',{'snapshot':snapshot,'opportunities':blocks,
        'preferences':result.get('resolved_preferences',[]),'mode':result.get('mode'),
        'start_utc':result.get('evaluation_start_utc'),'end_utc':result.get('evaluation_end_utc'),
        'config':result.get('provenance',{}).get('configuration')})
    write('evaluation.json',result)
    write('predicted_sleep.json',result.get('predicted_sleep',[]))
    (root/'model_trace.jsonl').write_text(''.join(json.dumps(p,ensure_ascii=False,sort_keys=True,allow_nan=False)+'\n' for p in result.get('trace',[])),encoding='utf-8')
    write('state_checkpoints.json',[result.get('initial_state'),result.get('final_state')])
    sources={p.name:digest(p.read_text(encoding='utf-8')) for p in sorted(module.glob('*.py'))}
    source_root=root/'sources';source_root.mkdir(exist_ok=True)
    for p in module.glob('*.py'):
        (source_root/p.name).write_bytes(p.read_bytes())
    write('model_manifest.json',{'schema_version':'sleep-model-archive-v2','model_version':VERSION,
        'provenance':result.get('provenance'),'input_hash':result.get('provenance',{}).get('input_hash'),
        'snapshot_hash':digest(snapshot),'python':sys.version,
        'numpy':numpy.__version__,'scipy':scipy.__version__,'platform':platform.platform(),
        'sources':sources,'upstream':json.loads((module/'sources/circadian/manifest.json').read_text(encoding='utf-8-sig')),
        'file_hashes':{name:digest((root/name).read_text(encoding='utf-8')) for name in
                      ('model_inputs.json','evaluation.json','predicted_sleep.json','model_trace.jsonl','state_checkpoints.json')}})


def reweight(evaluation,weights):
    """Recompute just ranking weights; biological state and original archive stay intact."""
    result=deepcopy(evaluation)
    if result.get('status')!='complete':
        raise ValueError('cannot rank an incomplete evaluation')
    cfg=configuration(dict(result['provenance']['configuration'],rank_weights=weights))
    loss=result['losses']
    total=sum(w*loss[k] for w,k in zip(weights,('physiological','need','preference','excess')))
    result['losses']['total']=total;result['utility']=result['display_score']=100*(1-total)
    result['provenance'].update(configuration=cfg,configuration_hash=digest(cfg),
                                reweighted_from=digest(evaluation),reused_trace_hash=result['provenance']['trace_hash'])
    # Old sensitivity scores use old weights; transform each from stored component losses.
    for item in result.get('scenarios',[]):
        if item['status']=='complete':
            total=sum(w*item['losses'][k] for w,k in zip(weights,('physiological','need','preference','excess')))
            item['losses']['total']=total;item['utility']=100*(1-total)
    values=[result['utility']]+[v['utility'] for v in result.get('scenarios',[]) if v.get('utility') is not None]
    result['sensitivity_range']=[min(values),max(values)]
    return result
