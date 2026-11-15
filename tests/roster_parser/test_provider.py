"""MiniMax client tests — httpx MockTransport only, zero real HTTP attempts."""
from __future__ import annotations

import base64
import json

import httpx
import pytest

from app.roster_parser.errors import (
    MissingApiKeyError,
    ProviderConfigError,
    ProviderError,
    ProviderTimeoutError,
)
from app.roster_parser.provider import (
    DEFAULT_MODEL,
    MiniMaxClient,
    ProviderConfig,
)


def make_config(**overrides) -> ProviderConfig:
    values = dict(
        api_key="test-key",
        base_url="https://provider.test/v1",
        model=DEFAULT_MODEL,
        timeout_seconds=5.0,
        max_completion_tokens=128,
        stream=False,
    )
    values.update(overrides)
    return ProviderConfig(**values)


def ok_body(content: str = '{"events": []}') -> dict:
    return {
        "id": "resp-123",
        "model": DEFAULT_MODEL,
        "choices": [
            {
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"total_tokens": 42},
        "base_resp": {"status_code": 0, "status_msg": ""},
    }


def make_client(handler, config: ProviderConfig | None = None) -> MiniMaxClient:
    return MiniMaxClient(
        config or make_config(), transport=httpx.MockTransport(handler)
    )


PNG_BYTES = b"\x89PNG\r\n\x1a\nfakepng"


class TestSuccess:
    def test_parses_content_usage_and_model(self):
        client = make_client(lambda request: httpx.Response(200, json=ok_body()))
        result = client.complete_vision("sys", [("image/png", PNG_BYTES)], "user")
        assert result.content == '{"events": []}'
        assert result.usage == {"total_tokens": 42}
        assert result.model == DEFAULT_MODEL
        assert result.status_code == 200
        assert result.http_attempts == 1
        assert result.streamed is False
        assert result.finish_reason == "stop"
        assert result.trace_id == "resp-123"  # from body id when no header
        assert result.response_header_seconds is not None

    def test_request_shape_and_auth(self):
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            captured["auth"] = request.headers.get("Authorization")
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json=ok_body())

        client = make_client(handler)
        client.complete_vision("system prompt", [("image/png", PNG_BYTES)], "user text")

        assert captured["url"] == "https://provider.test/v1/text/chatcompletion_v2"
        assert captured["auth"] == "Bearer test-key"
        body = captured["body"]
        assert body["model"] == DEFAULT_MODEL
        assert body["max_completion_tokens"] == 128
        assert "max_tokens" not in body  # deprecated parameter must not be sent
        assert "reasoning_effort" not in body  # unset = platform default
        assert body["messages"][0] == {"role": "system", "content": "system prompt"}
        user_content = body["messages"][1]["content"]
        assert user_content[0] == {"type": "text", "text": "user text"}
        image_part = user_content[1]
        assert image_part["type"] == "image_url"
        expected = "data:image/png;base64," + base64.b64encode(PNG_BYTES).decode()
        assert image_part["image_url"]["url"] == expected

    def test_multiple_images(self):
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json=ok_body())

        client = make_client(handler)
        client.complete_vision(
            "s", [("image/png", b"a"), ("image/jpeg", b"b")], "u"
        )
        parts = captured["body"]["messages"][1]["content"]
        assert len(parts) == 3
        assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")
        assert parts[2]["image_url"]["url"].startswith("data:image/jpeg;base64,")


class TestFailureModes:
    def test_http_500_is_provider_error(self):
        client = make_client(lambda request: httpx.Response(500, text="boom"))
        with pytest.raises(ProviderError) as excinfo:
            client.complete_vision("s", [("image/png", PNG_BYTES)], "u")
        assert "500" in str(excinfo.value)

    def test_base_resp_error_is_provider_error(self):
        body = {"base_resp": {"status_code": 1004, "status_msg": "invalid api key"}}
        client = make_client(lambda request: httpx.Response(200, json=body))
        with pytest.raises(ProviderError) as excinfo:
            client.complete_vision("s", [("image/png", PNG_BYTES)], "u")
        assert "1004" in str(excinfo.value)

    def test_base_resp_legacy_status_field_is_provider_error(self):
        body = {"base_resp": {"status": 2049, "status_msg": "invalid api key"}}
        client = make_client(lambda request: httpx.Response(200, json=body))
        with pytest.raises(ProviderError):
            client.complete_vision("s", [("image/png", PNG_BYTES)], "u")

    def test_non_json_body_is_provider_error(self):
        client = make_client(lambda request: httpx.Response(200, text="<html>"))
        with pytest.raises(ProviderError):
            client.complete_vision("s", [("image/png", PNG_BYTES)], "u")

    def test_no_choices_is_provider_error(self):
        client = make_client(lambda request: httpx.Response(200, json={"choices": []}))
        with pytest.raises(ProviderError):
            client.complete_vision("s", [("image/png", PNG_BYTES)], "u")

    def test_empty_content_is_provider_error(self):
        body = {"choices": [{"message": {"role": "assistant", "content": "  "}}]}
        client = make_client(lambda request: httpx.Response(200, json=body))
        with pytest.raises(ProviderError):
            client.complete_vision("s", [("image/png", PNG_BYTES)], "u")

    def test_timeout_is_provider_timeout(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectTimeout("timed out", request=request)

        client = make_client(handler)
        with pytest.raises(ProviderTimeoutError):
            client.complete_vision("s", [("image/png", PNG_BYTES)], "u")

    def test_connect_error_is_provider_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("no route", request=request)

        client = make_client(handler)
        with pytest.raises(ProviderError):
            client.complete_vision("s", [("image/png", PNG_BYTES)], "u")

    def test_missing_api_key_raises_without_http(self):
        client = MiniMaxClient(
            make_config(api_key=None), transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json=ok_body())
            )
        )
        with pytest.raises(MissingApiKeyError):
            client.complete_vision("s", [("image/png", PNG_BYTES)], "u")


class TestStreaming:
    @staticmethod
    def sse_body() -> str:
        chunks = [
            {"model": DEFAULT_MODEL, "choices": [{"delta": {"role": "assistant", "content": "{\"events\""}}]},
            {"model": DEFAULT_MODEL, "choices": [{"delta": {"content": ": []}"}}]},
            {"model": DEFAULT_MODEL, "choices": [{"delta": {}, "finish_reason": "stop"}],
             "usage": {"total_tokens": 55, "prompt_tokens": 40, "completion_tokens": 15}},
        ]
        lines = [f"data: {json.dumps(chunk)}" for chunk in chunks]
        lines.append("data: [DONE]")
        return "\n\n".join(lines) + "\n\n"

    @staticmethod
    def sse_handler(captured: dict | None = None):
        def handler(request: httpx.Request) -> httpx.Response:
            if captured is not None:
                captured["body"] = json.loads(request.content)
            return httpx.Response(
                200,
                content=TestStreaming.sse_body().encode("utf-8"),
                headers={
                    "content-type": "text/event-stream; charset=utf-8",
                    "trace-id": "trace-abc",
                },
            )

        return handler

    def test_stream_success_accumulates_deltas(self):
        captured = {}
        client = make_client(
            self.sse_handler(captured), make_config(stream=True)
        )
        result = client.complete_vision("s", [("image/png", PNG_BYTES)], "u")
        assert result.content == '{"events": []}'
        assert result.streamed is True
        assert result.usage["total_tokens"] == 55
        assert result.model == DEFAULT_MODEL
        assert result.finish_reason == "stop"
        assert result.trace_id == "trace-abc"  # from response header
        assert result.content_type == "text/event-stream; charset=utf-8"
        assert result.data_chunks == 3
        assert result.sse_lines >= 4
        assert result.response_header_seconds is not None
        assert result.first_line_seconds is not None
        assert captured["body"]["stream"] is True

    def test_stream_uses_real_client_stream(self, monkeypatch):
        """The buffering defect (#4–#6): streaming must go through
        client.stream(...) and be consumed inside its context."""
        calls = {}
        original_stream = httpx.Client.stream

        def recording_stream(self, method, url, **kwargs):
            calls["method"] = method
            calls["url"] = str(url)
            return original_stream(self, method, url, **kwargs)

        monkeypatch.setattr(httpx.Client, "stream", recording_stream)
        client = make_client(self.sse_handler(), make_config(stream=True))
        client.complete_vision("s", [("image/png", PNG_BYTES)], "u")
        assert calls["method"] == "POST"
        assert calls["url"].endswith("/text/chatcompletion_v2")

    def test_stream_include_usage_only_on_chat_completions(self):
        captured_v2 = {}
        client = make_client(self.sse_handler(captured_v2), make_config(stream=True))
        client.complete_vision("s", [("image/png", PNG_BYTES)], "u")
        assert "stream_options" not in captured_v2["body"]

        captured_openai = {}
        client = make_client(
            self.sse_handler(captured_openai),
            make_config(stream=True, endpoint="chat_completions"),
        )
        client.complete_vision("s", [("image/png", PNG_BYTES)], "u")
        assert captured_openai["body"]["stream_options"] == {"include_usage": True}
        assert captured_openai["body"]["stream"] is True

    def test_stream_disabled_sends_plain_request(self):
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json=ok_body())

        client = make_client(handler, make_config(stream=False))
        result = client.complete_vision("s", [("image/png", PNG_BYTES)], "u")
        assert result.streamed is False
        assert "stream" not in captured["body"]

    def test_stream_platform_error_raises(self):
        chunk = {"base_resp": {"status_code": 1004, "status_msg": "invalid api key"}}
        body = f"data: {json.dumps(chunk)}\n\n"
        client = make_client(
            lambda request: httpx.Response(
                200, content=body.encode(), headers={"content-type": "text/event-stream"}
            ),
            make_config(stream=True),
        )
        with pytest.raises(ProviderError) as excinfo:
            client.complete_vision("s", [("image/png", PNG_BYTES)], "u")
        assert "1004" in str(excinfo.value)

    def test_stream_json_error_body_not_misreported_as_empty(self):
        """Lead-identified defect #2: HTTP 200 + application/json platform
        error body (e.g. status_code=2049) must surface as the real error,
        never as empty_completion."""
        body = json.dumps(
            {"base_resp": {"status_code": 2049, "status_msg": "invalid api key"}}
        ).encode()
        client = make_client(
            lambda request: httpx.Response(
                200, content=body, headers={"content-type": "application/json"}
            ),
            make_config(stream=True),
        )
        with pytest.raises(ProviderError) as excinfo:
            client.complete_vision("s", [("image/png", PNG_BYTES)], "u")
        message = str(excinfo.value)
        assert "2049" in message
        assert "empty_completion" not in message

    def test_stream_json_success_body_extracted(self):
        """A JSON (non-SSE) answer to a streaming request is still an
        answer — extracted, never reported as empty."""
        client = make_client(
            lambda request: httpx.Response(
                200, json=ok_body('{"events": [1]}'),
                headers={"content-type": "application/json"},
            ),
            make_config(stream=True),
        )
        result = client.complete_vision("s", [("image/png", PNG_BYTES)], "u")
        assert result.content == '{"events": [1]}'
        assert result.streamed is False

    def test_stream_empty_content_raises_with_reason(self):
        chunk = {"model": DEFAULT_MODEL, "choices": [{"delta": {"content": ""}}]}
        body = f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n"
        client = make_client(
            lambda request: httpx.Response(
                200, content=body.encode(), headers={"content-type": "text/event-stream"}
            ),
            make_config(stream=True),
        )
        with pytest.raises(ProviderError):
            client.complete_vision("s", [("image/png", PNG_BYTES)], "u")

    def test_stream_reasoning_only_classified(self):
        chunks = [
            {"choices": [{"delta": {"reasoning_content": "thinking..."}}]},
            {"choices": [{"delta": {"reasoning_content": "more"}, "finish_reason": "length"}]},
        ]
        body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
        client = make_client(
            lambda request: httpx.Response(
                200, content=body.encode(), headers={"content-type": "text/event-stream"}
            ),
            make_config(stream=True),
        )
        with pytest.raises(ProviderError) as excinfo:
            client.complete_vision("s", [("image/png", PNG_BYTES)], "u")
        message = str(excinfo.value)
        assert "reason=reasoning_only_stream" in message
        assert "content_pieces=0" in message
        assert "reasoning_pieces=2" in message
        assert "finish_reason=length" in message

    def test_stream_no_chunks_classified_empty_completion(self):
        client = make_client(
            lambda request: httpx.Response(
                200, content=b"\n\n", headers={"content-type": "text/event-stream"}
            ),
            make_config(stream=True),
        )
        with pytest.raises(ProviderError) as excinfo:
            client.complete_vision("s", [("image/png", PNG_BYTES)], "u")
        message = str(excinfo.value)
        assert "reason=empty_completion" in message
        assert "data_chunks=0" in message

    def test_stream_unparsed_chunks_counted(self):
        body = "data: not-json-at-all\n\ndata: [DONE]\n\n"
        client = make_client(
            lambda request: httpx.Response(
                200, content=body.encode(), headers={"content-type": "text/event-stream"}
            ),
            make_config(stream=True),
        )
        with pytest.raises(ProviderError) as excinfo:
            client.complete_vision("s", [("image/png", PNG_BYTES)], "u")
        assert "unparsed_chunks=1" in str(excinfo.value)

    def test_stream_sse_debug_capture(self, tmp_path, monkeypatch):
        capture = tmp_path / "sse-capture.log"
        monkeypatch.setenv("MINIMAX_SSE_DEBUG_PATH", str(capture))
        client = make_client(self.sse_handler(), make_config(stream=True))
        client.complete_vision("s", [("image/png", PNG_BYTES)], "u")
        raw = capture.read_text(encoding="utf-8")
        assert raw.count("data:") >= 3
        assert "[DONE]" in raw

    def test_stream_sse_debug_path_unwritable_raises(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MINIMAX_SSE_DEBUG_PATH", str(tmp_path / "no-dir" / "x.log"))
        client = make_client(self.sse_handler(), make_config(stream=True))
        with pytest.raises(ProviderError) as excinfo:
            client.complete_vision("s", [("image/png", PNG_BYTES)], "u")
        assert "MINIMAX_SSE_DEBUG_PATH" in str(excinfo.value)

    def test_stream_http_error_raises_provider_error(self):
        client = make_client(
            lambda request: httpx.Response(500, text="boom"),
            make_config(stream=True),
        )
        with pytest.raises(ProviderError):
            client.complete_vision("s", [("image/png", PNG_BYTES)], "u")

    def test_env_stream_toggle(self, monkeypatch):
        monkeypatch.setenv("MINIMAX_STREAM", "0")
        assert ProviderConfig.from_env().stream is False
        monkeypatch.setenv("MINIMAX_STREAM", "1")
        assert ProviderConfig.from_env().stream is True
        monkeypatch.delenv("MINIMAX_STREAM")
        assert ProviderConfig.from_env().stream is True


class TestEndpointVariants:
    """Explicit endpoint/effort selection: default unchanged, no silent
    fallback, model name never auto-changed. The community-sourced 'vlm'
    branch was removed (no official documentation)."""

    def test_chat_completions_variant(self):
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json=ok_body())

        client = make_client(
            handler, make_config(endpoint="chat_completions", reasoning_effort="low")
        )
        client.complete_vision("sys", [("image/png", PNG_BYTES)], "u")
        assert captured["url"].endswith("/chat/completions")
        assert captured["body"]["reasoning_effort"] == "low"
        assert captured["body"]["max_completion_tokens"] == 128
        assert captured["body"]["messages"][0] == {"role": "system", "content": "sys"}

    def test_invalid_endpoint_raises_everywhere(self):
        with pytest.raises(ProviderConfigError):
            make_config(endpoint="vlm")  # removed: no official documentation
        with pytest.raises(ProviderConfigError):
            make_config(endpoint="something-else")

    def test_invalid_reasoning_effort_raises(self):
        with pytest.raises(ProviderConfigError):
            make_config(reasoning_effort="ultra")

    def test_nonpositive_max_completion_tokens_raises(self):
        with pytest.raises(ProviderConfigError):
            make_config(max_completion_tokens=0)

    def test_env_validation_and_defaults(self, monkeypatch):
        for name in (
            "MINIMAX_REASONING_EFFORT",
            "MINIMAX_ENDPOINT",
            "MINIMAX_MAX_COMPLETION_TOKENS",
        ):
            monkeypatch.delenv(name, raising=False)
        config = ProviderConfig.from_env()
        assert config.reasoning_effort is None  # field not sent
        assert config.endpoint == "chatcompletion_v2"
        assert config.max_completion_tokens == 16384

        monkeypatch.setenv("MINIMAX_REASONING_EFFORT", "XHIGH")
        monkeypatch.setenv("MINIMAX_ENDPOINT", "CHAT_COMPLETIONS")
        monkeypatch.setenv("MINIMAX_MAX_COMPLETION_TOKENS", "16384")
        config = ProviderConfig.from_env()
        assert config.reasoning_effort == "xhigh"  # case-insensitive
        assert config.endpoint == "chat_completions"
        assert config.max_completion_tokens == 16384

        monkeypatch.setenv("MINIMAX_REASONING_EFFORT", "bogus")
        with pytest.raises(ProviderConfigError):
            ProviderConfig.from_env()
        monkeypatch.setenv("MINIMAX_REASONING_EFFORT", "low")
        monkeypatch.setenv("MINIMAX_ENDPOINT", "bogus")
        with pytest.raises(ProviderConfigError):
            ProviderConfig.from_env()


class TestConfigFromEnv:
    def test_env_overrides(self, monkeypatch):
        monkeypatch.setenv("MINIMAX_API_KEY", "env-key")
        monkeypatch.setenv("MINIMAX_BASE_URL", "https://api.minimax.cn/v1/")
        monkeypatch.setenv("MINIMAX_MODEL", "custom-model")
        monkeypatch.setenv("MINIMAX_TIMEOUT_SECONDS", "12.5")
        monkeypatch.setenv("MINIMAX_MAX_COMPLETION_TOKENS", "999")
        config = ProviderConfig.from_env()
        assert config.api_key == "env-key"
        assert config.base_url == "https://api.minimax.cn/v1"
        assert config.model == "custom-model"
        assert config.timeout_seconds == 12.5
        assert config.max_completion_tokens == 999

    def test_defaults_without_env(self, monkeypatch):
        for name in (
            "MINIMAX_API_KEY",
            "MINIMAX_BASE_URL",
            "MINIMAX_MODEL",
            "MINIMAX_TIMEOUT_SECONDS",
            "MINIMAX_MAX_COMPLETION_TOKENS",
        ):
            monkeypatch.delenv(name, raising=False)
        config = ProviderConfig.from_env()
        assert config.api_key is None
        assert config.base_url == "https://api.minimaxi.com/v1"
        assert config.model == "MiniMax-M3.1-Flash"
        assert config.timeout_seconds == 60.0
        assert config.max_completion_tokens == 16384
