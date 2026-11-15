"""Execution audit belongs to the application, not to the benchmark's oracle."""
from .state import instant


def check_execution(snapshot, start_utc, end_utc):
    start,end=instant(start_utc),instant(end_utc)
    actual=[e for e in snapshot['events'] if e['kind']=='sleep_actual' and instant(e['end_utc'])>start and instant(e['start_utc'])<end]
    conflicts=[]
    for sleep in actual:
        a,b=instant(sleep['start_utc']),instant(sleep['end_utc'])
        for event in snapshot['events']:
            if event['kind'] not in ('duty','commute','preparation','commitment'):continue
            if event['kind']=='commitment' and event.get('locked',True) is False:continue
            c,d=instant(event['start_utc']),instant(event['end_utc'])
            if max(a,c)>=min(b,d):continue
            aircraft=any(e['kind']=='crew_rest' and e.get('id') == sleep.get('crew_rest_id')
                and e.get('authorized') is True and event['id'] in e.get('authorized_event_ids', [])
                and instant(e['start_utc'])<=a and instant(e['end_utc'])>=b for e in snapshot['events'])
            if event['kind']=='duty' and aircraft:continue
            conflicts.append({'sleep_id':sleep['id'],'event_id':event['id'],'rule':'actual_sleep_overlap',
                              'status':'fail','source':sleep.get('source')})
    return {'checks':conflicts,'violating_sleep_count':len({c['sleep_id'] for c in conflicts}),
            'evaluated_sleep_count':len(actual),'legal_status':'not_evaluated'}
