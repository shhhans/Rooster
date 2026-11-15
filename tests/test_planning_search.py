from copy import deepcopy
from app.planning.engine import plan,check_plan
from app.planning.search import generate
from tests.test_planning_benchmark import snapshot
from app.sleep_model.contracts import VERSION


def test_production_search_is_deterministic_and_uses_model():
    snap=snapshot(horizon='2027-04-04T06:00:00Z')
    before=deepcopy(snap)
    a=plan(snap);b=plan(snap)
    assert a==b and snap==before
    assert a['selected']
    assert a['model_version']==VERSION
    assert all(c['metrics']['model_version']==VERSION and 'sleep_count' in c['metrics']['losses'] for c in a['candidates'])
    scores=[c['metrics']['utility'] for c in a['candidates']]
    assert scores==sorted(scores,reverse=True)
    assert all('heuristic' not in c for c in a['candidates'])


def test_search_never_bridges_separate_environment_resources():
    snap=snapshot(now='2027-04-03T00:00:00Z',horizon='2027-04-03T04:00:00Z')
    windows=[dict(start_utc='2027-04-03T00:00:00Z',end_utc='2027-04-03T02:00:00Z',timezone='UTC'),
             dict(start_utc='2027-04-03T02:00:00Z',end_utc='2027-04-03T04:00:00Z',timezone='Asia/Tokyo')]
    candidates,info=generate(snap,windows)
    assert candidates
    for blocks in candidates:
        for b in blocks:
            assert any(w['start_utc']<=b['start_utc'] and b['end_utc']<=w['end_utc'] for w in windows)


def test_no_environment_facts_still_allow_search_in_known_unoccupied_time():
    snap=snapshot(horizon='2027-04-04T06:00:00Z')
    snap['events']=[]
    result=plan(snap)
    assert result['status']=='ready'
    assert result['candidates']
