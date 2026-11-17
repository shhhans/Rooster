import json
import socket
from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient

from app.main import create_app
from app.planning import timeline_agent
from app.services.reactive_timeline import queue_review, run_job


def test_partial_stream_is_persisted_before_timeout_and_can_be_downloaded(tmp_path, monkeypatch):
    partial = b'data: {"choices":[{"delta":{"reasoning_content":"unfinished'

    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read1(self, size):
            if not hasattr(self, 'sent'):
                self.sent = True
                return partial
            raise socket.timeout('read timed out')

    monkeypatch.setenv('ARK_AGENT_API_KEY', 'test')
    monkeypatch.setattr(timeline_agent.urllib.request, 'urlopen', lambda *args, **kwargs: Response())
    with TestClient(create_app(str(tmp_path / 'timeline.db'))) as client:
        store = client.app.state.timeline_service.store
        now = datetime.now(timezone.utc)
        job = queue_review(store, 'Eva', now, now + timedelta(days=1), trigger='manual')
        store.connection.execute("UPDATE reactive_jobs SET status='queued' WHERE job_id=?", (job,))
        monkeypatch.setattr('app.services.reactive_timeline.snapshot', lambda *args, **kwargs: {'revision': 0})
        monkeypatch.setattr(timeline_agent, 'begin_run', lambda *args, **kwargs: {'step_count': 0})
        monkeypatch.setattr(timeline_agent, 'step', lambda state, client, **kwargs: client([], []))
        run_job(store, job)
        row = store.connection.execute('SELECT status,error FROM reactive_jobs WHERE job_id=?', (job,)).fetchone()
        assert row['status'] == 'failed' and 'timed out' in row['error']
        entries = client.get(f'/api/v1/reactive/jobs/{job}/model-requests', params={'user_id': 'Eva'}).json()['requests']
        assert len(entries) == 1 and 'TimeoutError' in entries[0]['error'] and entries[0]['finished_at']
        url = f"/api/v1/reactive/jobs/{job}/model-requests/{entries[0]['request_id']}/stream"
        assert client.get(url, params={'user_id': 'Eva'}).content == partial
        assert client.get(url, params={'user_id': 'other'}).status_code == 404
        activity = client.get(f'/api/v1/reactive/jobs/{job}/activity', params={'user_id': 'Eva'}).json()
        assert activity['steps'][0]['requests'][0]['bytes_received'] == len(partial)
        assert activity['status'] == 'failed'
        assert client.get(f'/api/v1/reactive/jobs/{job}/messages', params={'user_id': 'Eva'}).json() == {'messages': []}


def test_stream_capture_keeps_completed_and_length_attempts(monkeypatch):
    def response(finish):
        data = ('data: ' + json.dumps({'choices': [{'delta': {'content': 'hello'}, 'finish_reason': finish}]}) + '\n\ndata: [DONE]\n\n').encode()
        class Response:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def read1(self, size):
                nonlocal data
                result, data = data, b''
                return result
        return Response()
    responses = iter([response('length'), response('stop')])
    monkeypatch.setenv('ARK_AGENT_API_KEY', 'test')
    monkeypatch.setattr(timeline_agent.urllib.request, 'urlopen', lambda *args, **kwargs: next(responses))
    events = []
    def capture(event, payload):
        events.append((event, payload))
        return len(events)
    assert timeline_agent.request([], [], on_stream=capture)['content'] == 'hello'
    assert [p['finish_reason'] for e, p in events if e == 'end'] == ['length', 'stop']
    assert len([1 for e, _ in events if e == 'data']) == 2


def test_disabled_reasoning_is_sent_to_model(monkeypatch):
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read1(self, size):
            if getattr(self, 'sent', False):
                return b''
            self.sent = True
            return b'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}\n\n'
    payloads = []
    monkeypatch.setenv('ARK_AGENT_API_KEY', 'test')
    def open_request(request, **kwargs):
        payloads.append(json.loads(request.data))
        return Response()
    monkeypatch.setattr(timeline_agent.urllib.request, 'urlopen', open_request)
    assert timeline_agent.request([], [], reasoning_effort='off')['content'] == 'ok'
    assert payloads[0]['thinking'] == {'type': 'disabled'}
    assert 'reasoning_effort' not in payloads[0]
