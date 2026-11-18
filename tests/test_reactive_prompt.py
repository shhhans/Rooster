from app.planning import timeline_agent


def test_new_runs_read_prompt_without_changing_existing_runs(tmp_path, monkeypatch):
    module = tmp_path / 'timeline_agent.py'
    prompt = tmp_path / 'prompts' / 'reactive_system.md'
    prompt.parent.mkdir()
    prompt.write_text('first prompt', encoding='utf-8')
    monkeypatch.setattr(timeline_agent, '__file__', str(module))
    snapshot = {'now_utc': '2026-11-17T12:00:00Z', 'active_plan': {'sleep_blocks': []}}
    first = timeline_agent.begin_run(snapshot, request={'kind': 'initial'})
    prompt.write_text('second prompt', encoding='utf-8')
    second = timeline_agent.begin_run(snapshot, request={'kind': 'initial'})
    assert first['messages'][0]['content'].startswith('first prompt')
    assert second['messages'][0]['content'].startswith('second prompt')
