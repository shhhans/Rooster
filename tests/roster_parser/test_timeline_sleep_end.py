from fastapi.testclient import TestClient
from app.main import create_app


def test_list_can_return_resolved_sleep_end_without_changing_default_contract(tmp_path):
    with TestClient(create_app(str(tmp_path / 'timeline.db'))) as client:
        response = client.post('/timeline/sleep/events', json={'user_id':'u', 'dedupe_key':'one', 'sleep_start':'2026-11-15T23:00', 'sleep_end':'2026-11-16T07:00', 'timezone':'Asia/Shanghai', 'source':'manual'})
        assert response.status_code == 200
        params = {'user_id':'u', 'window_start':'2026-11-15T00:00:00Z', 'window_end':'2026-11-17T00:00:00Z'}
        assert 'sleep_end_utc' not in client.get('/timeline/sleep/events', params=params).json()[0]
        item = client.get('/timeline/sleep/events', params={**params, 'include_end_utc':'true'}).json()[0]
        assert item['sleep_end_utc'] == '2026-11-15T23:00:00Z'
