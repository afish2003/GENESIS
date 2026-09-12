"""Tests for the OpenAI-compatible backend, the mock backend, and selection."""

import json

import httpx
import pytest

from controller.config import Backend, RunConfig
from controller.inference.backend import Message
from controller.inference.factory import create_backend, describe_backend
from controller.inference.mock_backend import MockBackend
from controller.inference.openai_backend import OpenAICompatBackend
from controller.phases.schemas import EvaluationOutput, MemorySummaryOutput


def _chat_response(content: str = "hello", model: str = "fake") -> dict:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "model": model,
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": content},
             "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
    }


def _backend_with(handler, **kwargs) -> OpenAICompatBackend:
    """Build a backend whose HTTP client is driven by `handler`."""
    b = OpenAICompatBackend(base_url="https://example.test/v1", model="fake", **kwargs)
    b._client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), headers=b._client.headers
    )
    return b


class TestOpenAICompatBackend:
    @pytest.mark.asyncio
    async def test_parses_standard_response(self):
        b = _backend_with(lambda r: httpx.Response(200, json=_chat_response("hi there")))
        result = await b.complete([Message(role="user", content="yo")])
        assert result.content == "hi there"
        assert result.prompt_tokens == 11
        assert result.completion_tokens == 7
        await b.close()

    @pytest.mark.asyncio
    async def test_request_shape(self):
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["url"] = str(request.url)
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, json=_chat_response())

        b = _backend_with(handler)
        await b.complete([Message(role="system", content="s"),
                          Message(role="user", content="u")], temperature=0.42)
        assert seen["url"] == "https://example.test/v1/chat/completions"
        assert seen["body"]["model"] == "fake"
        assert seen["body"]["temperature"] == 0.42
        assert seen["body"]["stream"] is False
        assert seen["body"]["messages"] == [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u"},
        ]
        # json_mode is off by default — not every server accepts it
        assert "response_format" not in seen["body"]
        await b.close()

    @pytest.mark.asyncio
    async def test_json_mode_opt_in(self):
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, json=_chat_response())

        b = _backend_with(handler, json_mode=True)
        await b.complete([Message(role="user", content="u")])
        assert seen["body"]["response_format"] == {"type": "json_object"}
        await b.close()

    def test_auth_header_present_when_key_set(self):
        b = OpenAICompatBackend(base_url="https://x/v1", model="m", api_key="sk-abc")
        assert b._client.headers["Authorization"] == "Bearer sk-abc"

    def test_auth_header_absent_when_no_key(self):
        """Local servers reject a bogus Authorization header; omit it entirely."""
        b = OpenAICompatBackend(base_url="http://localhost:1234/v1", model="m")
        assert "authorization" not in {k.lower() for k in b._client.headers}

    def test_base_url_trailing_slash_normalized(self):
        b = OpenAICompatBackend(base_url="http://localhost:1234/v1/", model="m")
        assert b.base_url == "http://localhost:1234/v1"

    @pytest.mark.asyncio
    async def test_4xx_not_retried(self):
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(400, json={"error": "bad request"})

        b = _backend_with(handler)
        with pytest.raises(httpx.HTTPStatusError):
            await b.complete([Message(role="user", content="u")])
        assert calls["n"] == 1, "client errors must not be retried"
        await b.close()

    @pytest.mark.asyncio
    async def test_empty_choices_raises(self):
        b = _backend_with(lambda r: httpx.Response(200, json={"choices": []}))
        with pytest.raises(ValueError):
            await b.complete([Message(role="user", content="u")])
        await b.close()

    @pytest.mark.asyncio
    async def test_health_check(self):
        def handler(request: httpx.Request) -> httpx.Response:
            assert str(request.url).endswith("/v1/models")
            return httpx.Response(200, json={"data": []})

        b = _backend_with(handler)
        assert await b.health_check() is True
        await b.close()


class TestMockBackend:
    @pytest.mark.asyncio
    async def test_free_form_returns_prose(self):
        b = MockBackend()
        r = await b.complete([Message(role="user", content="say something")])
        assert len(r.content) > 0
        assert not r.content.startswith("{")

    @pytest.mark.asyncio
    async def test_satisfies_a_real_phase_schema(self):
        """complete_structured must round-trip without hand-written fixtures."""
        b = MockBackend()
        out = await b.complete_structured(
            messages=[Message(role="user", content="summarize")],
            response_schema=MemorySummaryOutput,
        )
        assert isinstance(out, MemorySummaryOutput)

    @pytest.mark.asyncio
    async def test_satisfies_nested_schema_with_refs(self):
        """EvaluationOutput nests EvaluationScores via $ref."""
        b = MockBackend()
        out = await b.complete_structured(
            messages=[Message(role="user", content="evaluate")],
            response_schema=EvaluationOutput,
        )
        assert isinstance(out, EvaluationOutput)

    @pytest.mark.asyncio
    async def test_counts_calls(self):
        b = MockBackend()
        await b.complete([Message(role="user", content="a")])
        await b.complete([Message(role="user", content="b")])
        assert b.call_count == 2


class TestBackendSelection:
    def _config(self, **kw) -> RunConfig:
        return RunConfig(run_id="R", condition="BASELINE", **kw)

    def test_default_is_ollama(self):
        assert self._config().inference_backend == Backend.OLLAMA

    def test_selects_openai(self):
        cfg = self._config(inference_backend="openai", api_base_url="http://x/v1")
        assert isinstance(create_backend(cfg), OpenAICompatBackend)

    def test_selects_mock(self):
        cfg = self._config(inference_backend="mock")
        assert isinstance(create_backend(cfg), MockBackend)

    def test_describe_never_leaks_key(self):
        cfg = self._config(inference_backend="openai", api_key="sk-secret-value")
        described = describe_backend(cfg)
        assert "sk-secret-value" not in described
        assert "authenticated" in described
