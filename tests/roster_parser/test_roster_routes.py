"""Route-level tests — TestClient + injected fake provider, zero network.

Covers legal inputs, unsupported types, invalid/oversized PDFs, provider
failure/timeout, invalid model JSON, malformed context and the response
contract. Real-sample verification lives separately in
test_roster_real_sample.py (gated).
"""
from __future__ import annotations

import hashlib
import json

import pytest
from fastapi.testclient import TestClient

from app.api import roster_routes
from app.main import create_app
from app.roster_parser.errors import (
    MissingApiKeyError,
    ProviderError,
    ProviderTimeoutError,
)
from tests.roster_parser.conftest import (
    FakeProvider,
    flight_event,
    make_jpeg,
    make_pdf,
    make_png,
    model_output,
)

VALID_CONTENT = model_output([flight_event()])
CONTEXT = {
    "home_timezone": "Asia/Hong_Kong",
    "airport_timezone_map": json.dumps({"HKG": "Asia/Hong_Kong", "LHR": "Europe/London"}),
    "roster_year": "2026",
    "roster_month": "11",
}


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app(), raise_server_exceptions=False)


def install_fake(monkeypatch: pytest.MonkeyPatch, fake: FakeProvider) -> FakeProvider:
    monkeypatch.setattr(roster_routes, "_provider_factory", lambda: fake)
    return fake


def post_file(client: TestClient, name: str, data: bytes, mime: str, **form):
    files = {"file": (name, data, mime)}
    return client.post("/api/v1/roster/preview", files=files, data=form or CONTEXT)


class TestHappyPath:
    def test_png_upload_returns_preview(self, client, monkeypatch):
        fake = install_fake(monkeypatch, FakeProvider(content=VALID_CONTENT))
        png = make_png()
        resp = post_file(client, "roster.png", png, "image/png")
        assert resp.status_code == 200
        body = resp.json()
        assert set(body) == {"proposed_events", "unresolved_items", "warnings", "meta"}
        assert len(body["proposed_events"]) == 1
        event = body["proposed_events"][0]
        assert event["confirm_required"] is True
        assert event["schedule_payload"]["departure"] == "2026-11-03T23:55:00+08:00"
        meta = body["meta"]
        assert meta["model"] == "MiniMax-M3.1-Flash"
        assert meta["file_sha256"] == hashlib.sha256(png).hexdigest()
        assert meta["file_size_bytes"] == len(png)
        assert meta["content_type"] == "image/png"
        assert meta["http_attempts"] == 1
        assert meta["contract_version"] == "roster-preview-v1"
        assert meta["render"]["pdf_page_count"] is None
        # Exactly one provider call, carrying the image input
        assert len(fake.calls) == 1
        assert fake.calls[0]["images"][0][0] == "image/png"

    def test_jpeg_upload_same_structure(self, client, monkeypatch):
        install_fake(monkeypatch, FakeProvider(content=VALID_CONTENT))
        resp = post_file(client, "roster.jpg", make_jpeg(), "image/jpeg")
        assert resp.status_code == 200
        assert resp.json()["meta"]["content_type"] == "image/jpeg"
        assert len(resp.json()["proposed_events"]) == 1

    def test_pdf_rendered_to_png_for_provider(self, client, monkeypatch):
        fake = install_fake(monkeypatch, FakeProvider(content=VALID_CONTENT))
        resp = post_file(client, "roster.pdf", make_pdf(1), "application/pdf")
        assert resp.status_code == 200
        mime, image = fake.calls[0]["images"][0]
        assert mime == "image/png"
        assert image.startswith(b"\x89PNG\r\n\x1a\n")
        meta = resp.json()["meta"]
        assert meta["render"]["pdf_page_count"] == 1
        assert meta["render"]["pages_rendered"] == 1

    def test_preview_only_no_event_ids(self, client, monkeypatch):
        # Preview contract: nothing is persisted; events carry no storage ids.
        install_fake(monkeypatch, FakeProvider(content=VALID_CONTENT))
        body = post_file(client, "roster.png", make_png(), "image/png").json()
        assert all("id" not in e for e in body["proposed_events"])


class TestInputValidation:
    def test_unsupported_type_415(self, client, monkeypatch):
        install_fake(monkeypatch, FakeProvider(content=VALID_CONTENT))
        resp = post_file(client, "notes.txt", b"hello", "text/plain")
        assert resp.status_code == 415
        assert resp.json()["error_code"] == "unsupported_file_type"
        assert "request_id" in resp.json()

    def test_missing_file_400(self, client, monkeypatch):
        install_fake(monkeypatch, FakeProvider(content=VALID_CONTENT))
        resp = client.post("/api/v1/roster/preview", data=CONTEXT)
        assert resp.status_code == 400
        assert resp.json()["error_code"] == "validation_error"

    def test_oversized_413(self, client, monkeypatch):
        install_fake(monkeypatch, FakeProvider(content=VALID_CONTENT))
        monkeypatch.setenv("ROSTER_MAX_UPLOAD_BYTES", "10")
        resp = post_file(client, "roster.png", b"x" * 11, "image/png")
        assert resp.status_code == 413
        assert resp.json()["error_code"] == "file_too_large"

    def test_garbage_pdf_422(self, client, monkeypatch):
        install_fake(monkeypatch, FakeProvider(content=VALID_CONTENT))
        resp = post_file(client, "broken.pdf", b"%PDF-1.4 garbage", "application/pdf")
        assert resp.status_code == 422
        assert resp.json()["error_code"] == "pdf_render_error"

    def test_too_many_pages_413(self, client, monkeypatch):
        install_fake(monkeypatch, FakeProvider(content=VALID_CONTENT))
        monkeypatch.setenv("ROSTER_MAX_PAGES", "1")
        resp = post_file(client, "roster.pdf", make_pdf(2), "application/pdf")
        assert resp.status_code == 413
        assert resp.json()["error_code"] == "too_many_pages"

    def test_invalid_home_timezone_400(self, client, monkeypatch):
        install_fake(monkeypatch, FakeProvider(content=VALID_CONTENT))
        resp = post_file(client, "roster.png", make_png(), "image/png",
                         home_timezone="Not/AZone")
        assert resp.status_code == 400
        assert resp.json()["error_code"] == "invalid_timezone_context"

    def test_invalid_airport_map_400(self, client, monkeypatch):
        install_fake(monkeypatch, FakeProvider(content=VALID_CONTENT))
        resp = post_file(client, "roster.png", make_png(), "image/png",
                         airport_timezone_map="{not json")
        assert resp.status_code == 400
        assert resp.json()["error_code"] == "invalid_timezone_context"

    def test_invalid_roster_month_400(self, client, monkeypatch):
        install_fake(monkeypatch, FakeProvider(content=VALID_CONTENT))
        resp = post_file(client, "roster.png", make_png(), "image/png",
                         roster_month="13")
        assert resp.status_code == 400
        assert resp.json()["error_code"] == "invalid_roster_context"


class TestProviderFailureModes:
    def test_provider_timeout_504(self, client, monkeypatch):
        install_fake(monkeypatch, FakeProvider(error=ProviderTimeoutError("t")))
        resp = post_file(client, "roster.png", make_png(), "image/png")
        assert resp.status_code == 504
        assert resp.json()["error_code"] == "provider_timeout"

    def test_provider_error_502(self, client, monkeypatch):
        install_fake(monkeypatch, FakeProvider(error=ProviderError("bad upstream")))
        resp = post_file(client, "roster.png", make_png(), "image/png")
        assert resp.status_code == 502
        assert resp.json()["error_code"] == "provider_error"

    def test_missing_api_key_503(self, client, monkeypatch):
        install_fake(monkeypatch, FakeProvider(error=MissingApiKeyError("no key")))
        resp = post_file(client, "roster.png", make_png(), "image/png")
        assert resp.status_code == 503
        assert resp.json()["error_code"] == "missing_api_key"

    def test_invalid_model_json_502(self, client, monkeypatch):
        install_fake(monkeypatch, FakeProvider(content="totally not json"))
        resp = post_file(client, "roster.png", make_png(), "image/png")
        assert resp.status_code == 502
        assert resp.json()["error_code"] == "invalid_model_output"

    def test_no_silent_empty_success_on_model_garbage(self, client, monkeypatch):
        # A failed parse must surface as an error, never as an empty 200.
        install_fake(monkeypatch, FakeProvider(content=""))
        resp = post_file(client, "roster.png", make_png(), "image/png")
        assert resp.status_code == 502


class TestRouteDocs:
    def test_openapi_documents_preview(self, client):
        schema = client.get("/openapi.json").json()
        assert "/api/v1/roster/preview" in schema["paths"]
