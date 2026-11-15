"""Real pipeline progress boundaries with an offline injected vision provider."""
import json

from fastapi.testclient import TestClient

from app.api import roster_routes
from app.main import create_app
from app.roster_parser.errors import ProviderTimeoutError
from app.roster_parser.rendering import RosterLimits
from tests.roster_parser.conftest import FakeProvider, flight_event, make_pdf, make_png, model_output


def frames(response):
    return [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]


def test_stage_is_yielded_before_vision_call(monkeypatch):
    fake = FakeProvider(content=model_output([flight_event()]))
    monkeypatch.setattr(roster_routes, "_provider_factory", lambda: fake)
    steps = roster_routes._preview_steps(make_pdf(), RosterLimits.from_env(), None, None, 2026, 11)
    assert next(steps)["stage"] == "validating"
    assert next(steps) == {"type": "progress", "stage": "rendering", "pages": 1}
    assert next(steps) == {"type": "progress", "stage": "recognizing", "pages": 1}
    assert not fake.calls  # The waiting status reaches the caller before blocking work.
    assert next(steps)["stage"] == "normalizing"
    assert len(fake.calls) == 1
    result = next(steps)
    assert result["type"] == "complete"
    assert len(result["result"]["proposed_events"]) == 1


def test_stream_and_original_preview_share_result(monkeypatch):
    fake = FakeProvider(content=model_output([flight_event()]))
    monkeypatch.setattr(roster_routes, "_provider_factory", lambda: fake)
    files = {"file": ("roster.png", make_png(), "image/png")}
    with TestClient(create_app()) as client:
        streamed = client.post("/api/v1/roster/preview/stream", files=files)
        original = client.post("/api/v1/roster/preview", files=files)
    assert streamed.headers["content-type"].startswith("text/event-stream")
    events = frames(streamed)
    assert [e.get("stage") for e in events[:-1]] == ["validating", "recognizing", "normalizing"]
    assert events[-1] == {"type": "complete", "result": original.json()}
    assert len(fake.calls) == 2  # One per explicitly requested preview, never one per stage.


def test_failed_stream_never_emits_completion(monkeypatch):
    fake = FakeProvider(error=ProviderTimeoutError("model timed out"))
    monkeypatch.setattr(roster_routes, "_provider_factory", lambda: fake)
    with TestClient(create_app()) as client:
        response = client.post("/api/v1/roster/preview/stream",
                               files={"file": ("roster.png", make_png(), "image/png")})
    events = frames(response)
    assert events[-1]["type"] == "error"
    assert events[-1]["message"] == "model timed out"
    assert not any(e["type"] == "complete" for e in events)
    assert len(fake.calls) == 1


def test_bad_file_is_reported_before_model(monkeypatch):
    fake = FakeProvider()
    monkeypatch.setattr(roster_routes, "_provider_factory", lambda: fake)
    with TestClient(create_app()) as client:
        response = client.post("/api/v1/roster/preview/stream",
                               files={"file": ("bad.txt", b"not an image", "text/plain")})
    events = frames(response)
    assert events[0]["stage"] == "validating"
    assert events[-1]["type"] == "error"
    assert not fake.calls
