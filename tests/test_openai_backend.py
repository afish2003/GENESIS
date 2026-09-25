"""Tests for the OpenAI-compatible backend, the mock backend, and selection."""

import asyncio
import json

import httpx
import pytest

from controller.config import Backend, RunConfig
from controller.inference.backend import Message
from controller.inference.factory import create_backend, describe_backend
from controller.inference.mock_backend import MockBackend
from controller.inference.openai_backend import OpenAICompatBackend
from controller.phases.schemas import MemorySummaryOutput
from controller.tasks import ProtocolTask


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
    async def test_json_mode_does_not_constrain_a_free_form_call(self):
        """json_mode is a property of the endpoint, not of every call.

        It used to be OR'd straight into complete(), so response_format reached
        every request the backend made — including retrieval's prose
        summariser, which then came back as "{}". Structured calls opt in via
        prefers_json_mode; free-form calls never do.
        """
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, json=_chat_response())

        b = _backend_with(handler, json_mode=True)
        await b.complete([Message(role="user", content="summarise this prose")])
        assert "response_format" not in seen["body"]
        await b.close()

    @pytest.mark.asyncio
    async def test_a_caller_asking_for_json_gets_it(self):
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, json=_chat_response())

        b = _backend_with(handler, json_mode=True)
        await b.complete([Message(role="user", content="u")], force_json=True)
        assert seen["body"]["response_format"] == {"type": "json_object"}
        await b.close()

    def test_structured_calls_opt_in_when_the_endpoint_supports_it(self):
        b = OpenAICompatBackend(base_url="https://x/v1", model="m", json_mode=True)
        assert b.prefers_json_mode is True

    def test_structured_calls_do_not_opt_in_when_json_mode_is_off(self):
        b = OpenAICompatBackend(base_url="https://x/v1", model="m", json_mode=False)
        assert b.prefers_json_mode is False

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
    async def test_health_check_generates_rather_than_listing(self):
        """GET /models proves the gateway is up, not that the model works.

        Measured 2026-09-20: a LiteLLM gateway answered /models with 200 in
        0.3s while every completion for one of the listed models hung until
        timeout. A battery spent 50 minutes on that model before anything
        noticed. The check has to ask the model to produce a token.
        """
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["url"] = str(request.url)
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, json={
                "choices": [{"message": {"content": "o"}, "finish_reason": "stop"}]})

        b = _backend_with(handler)
        assert await b.health_check() is True
        assert seen["url"].endswith("/chat/completions")
        assert seen["body"]["max_tokens"] == 1, "a probe must stay cheap"
        await b.close()

    @pytest.mark.asyncio
    async def test_health_check_false_when_the_model_hangs(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("", request=request)

        b = _backend_with(handler)
        assert await b.health_check() is False
        await b.close()

    @pytest.mark.asyncio
    async def test_a_listed_but_unservable_model_is_not_healthy(self):
        """The exact shape of the outage: /models fine, completions dead."""
        def handler(request: httpx.Request) -> httpx.Response:
            if str(request.url).endswith("/models"):
                return httpx.Response(200, json={"data": [{"id": "m"}]})
            raise httpx.ReadTimeout("", request=request)

        b = _backend_with(handler)
        assert await b.health_check() is False
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
        """A task's evaluation schema nests its scores model via $ref."""
        b = MockBackend()
        out = await b.complete_structured(
            messages=[Message(role="user", content="evaluate")],
            response_schema=ProtocolTask().evaluation_schema(),
        )
        assert set(type(out.scores).model_fields) == set(ProtocolTask().dimensions)

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


class TestOutputIsCapped:
    """Nothing capped generation length, and that is not a tuning detail.

    No max_tokens was ever sent. A model that fails to stop generates until it
    fills the context window: one evaluation call ran 19 MINUTES against a
    32k-context judge before being killed, with 57 successful HTTP calls and
    zero errors in the log — it simply never came back. At ~41 tok/s, 32k
    tokens is 13 minutes, which is the whole of it.

    The second failure mode is quieter. A response cut off at the cap is
    truncated mid-JSON, so a structured call fails to parse — and surfaces as
    "the model is bad at JSON" rather than "the response was cut off". Those
    need different fixes, so the backend says which.
    """

    def test_a_cap_exists_by_default(self):
        assert RunConfig(run_id="C", condition="BASELINE").max_output_tokens == 4096

    def test_the_cap_fits_a_full_doctrine_document(self):
        """Doctrine revisions must emit the COMPLETE revised document, which is
        the largest legitimate output any phase asks for. Capping below it
        would truncate every revision."""
        config = RunConfig(run_id="C", condition="BASELINE")
        longest_legitimate = config.max_protocol_length_tokens
        assert config.max_output_tokens > longest_legitimate

    def test_the_backend_sends_it(self):
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, json=_chat_response())

        b = _backend_with(handler, max_output_tokens=1234)
        import asyncio
        asyncio.run(b.complete([Message(role="user", content="u")]))
        assert seen["body"]["max_tokens"] == 1234
        asyncio.run(b.close())

    def test_truncation_is_reported_not_silent(self, caplog):
        """Otherwise a cut-off response looks like a JSON failure."""
        import asyncio
        import logging

        def handler(request: httpx.Request) -> httpx.Response:
            body = _chat_response()
            body["choices"][0]["finish_reason"] = "length"
            return httpx.Response(200, json=body)

        b = _backend_with(handler)
        with caplog.at_level(logging.WARNING):
            asyncio.run(b.complete([Message(role="user", content="u")]))
        assert any("truncated" in r.message.lower() or "cap" in r.message.lower()
                   for r in caplog.records)
        asyncio.run(b.close())

    def test_a_normal_finish_is_not_reported(self, caplog):
        import asyncio
        import logging

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=_chat_response())

        b = _backend_with(handler)
        with caplog.at_level(logging.WARNING):
            asyncio.run(b.complete([Message(role="user", content="u")]))
        assert not [r for r in caplog.records if "cap" in r.message.lower()]
        asyncio.run(b.close())


class TestThinkingToggle:
    """Reasoning models put their chain in reasoning_content, and max_tokens
    counts it.

    qwen3.6-35b-a3b spent an entire 1500-token budget thinking and returned
    content="" with finish_reason "length". Every "unbounded generation", the
    19-minute hang, and every truncated parse failure traced back to this. The
    same call with thinking off took 0.5s and 27 tokens.
    """

    def _body(self, handler_result=None, **kw):
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["body"] = json.loads(request.content)
            return handler_result or httpx.Response(200, json=_chat_response())

        b = _backend_with(handler, **kw)
        asyncio.run(b.complete([Message(role="user", content="u")]))
        asyncio.run(b.close())
        return seen["body"]

    def test_thinking_is_disabled_by_default(self):
        body = self._body()
        assert body["chat_template_kwargs"] == {"enable_thinking": False}

    def test_nothing_is_sent_when_thinking_is_wanted(self):
        """Opting in means letting the model's own default apply."""
        assert "chat_template_kwargs" not in self._body(enable_thinking=True)

    def test_config_default_is_off(self):
        assert RunConfig(run_id="T", condition="BASELINE").enable_thinking is False

    @pytest.mark.parametrize("body_text", [
        "unknown field chat_template_kwargs", "enable_thinking is not supported"])
    def test_an_endpoint_rejecting_the_extension_is_recognised(self, body_text):
        req = httpx.Request("POST", "http://x/v1/chat/completions")
        err = httpx.HTTPStatusError(
            "x", request=req, response=httpx.Response(400, text=body_text, request=req))
        assert OpenAICompatBackend._is_template_kwargs_rejection(err) is True

    def test_an_unrelated_400_is_not_mistaken_for_one(self):
        """An over-broad version retried EVERY 400 with the parameter stripped,
        which test_4xx_not_retried caught."""
        req = httpx.Request("POST", "http://x/v1/chat/completions")
        err = httpx.HTTPStatusError(
            "x", request=req,
            response=httpx.Response(400, text="model 'nope' not found", request=req))
        assert OpenAICompatBackend._is_template_kwargs_rejection(err) is False

    def test_a_rejecting_endpoint_is_retried_once_without_it(self):
        calls = []

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            calls.append("chat_template_kwargs" in body)
            if calls[-1]:
                return httpx.Response(400, text="unknown field chat_template_kwargs")
            return httpx.Response(200, json=_chat_response())

        b = _backend_with(handler)
        asyncio.run(b.complete([Message(role="user", content="u")]))
        assert calls == [True, False]
        assert b._thinking_toggle_supported is False
        # And not sent again for the rest of the run.
        asyncio.run(b.complete([Message(role="user", content="u")]))
        assert calls == [True, False, False]
        asyncio.run(b.close())

    def test_truncation_naming_thinking_when_that_is_the_cause(self, caplog):
        import logging

        def handler(request: httpx.Request) -> httpx.Response:
            body = _chat_response()
            body["choices"][0]["finish_reason"] = "length"
            body["choices"][0]["message"]["reasoning_content"] = "thought " * 500
            body["choices"][0]["message"]["content"] = ""
            return httpx.Response(200, json=body)

        b = _backend_with(handler)
        with caplog.at_level(logging.WARNING):
            asyncio.run(b.complete([Message(role="user", content="u")]))
        assert any("THINKING" in r.message for r in caplog.records), (
            "a truncation caused by thinking must say so — the fix is to turn "
            "thinking off, not to raise the cap"
        )
        asyncio.run(b.close())


class TestChatTemplateRejectionPhrasings:
    """The matcher must cover how servers actually word this, not one guess.

    vLLM behind LiteLLM, serving a Mistral tokenizer, answers 400 with
    "chat_template is not supported for Mistral tokenizers." — no "_kwargs"
    and no "enable_thinking". The matcher looked for those two strings only,
    so the rejection was treated as a caller error and raised, and every phase
    of every cycle failed within seconds of the run starting.
    """

    def _err(self, body: str, status: int = 400) -> httpx.HTTPStatusError:
        request = httpx.Request("POST", "http://x/v1/chat/completions")
        response = httpx.Response(status, text=body, request=request)
        return httpx.HTTPStatusError("e", request=request, response=response)

    @pytest.mark.parametrize("body", [
        "chat_template is not supported for Mistral tokenizers.",
        "litellm.BadRequestError: OpenAIException - chat_template is not "
        "supported for Mistral tokenizers.. Received Model Group=mistral",
        "unrecognised argument: chat_template_kwargs",
        "enable_thinking is not a valid field",
        "CHAT_TEMPLATE IS NOT SUPPORTED",
    ])
    def test_recognised(self, body):
        from controller.inference.openai_backend import OpenAICompatBackend
        assert OpenAICompatBackend._is_template_kwargs_rejection(self._err(body))

    @pytest.mark.parametrize("body", [
        "model 'nope' does not exist",
        "context length exceeded",
        "invalid api key",
        "response_format json_object is not supported",
    ])
    def test_not_mistaken_for_other_400s(self, body):
        """Still narrow: stripping the parameter must not become the response
        to every client error."""
        from controller.inference.openai_backend import OpenAICompatBackend
        assert not OpenAICompatBackend._is_template_kwargs_rejection(self._err(body))

    def test_a_500_that_names_the_parameter_IS_this(self):
        """This test previously asserted the opposite, on my assumption that a
        5xx is always a server problem. LiteLLM in front of a llama.cpp
        backend answers chat_template_kwargs with its own 500:

          litellm.InternalServerError: AsyncCompletions.create() got an
          unexpected keyword argument 'chat_template_kwargs'

        which is a caller error wearing a server error's status. Retrying it
        resends the bad parameter, so Qwen3.8-Flash-Next was unusable.
        """
        from controller.inference.openai_backend import OpenAICompatBackend
        assert OpenAICompatBackend._is_template_kwargs_rejection(self._err(
            "litellm.InternalServerError: AsyncCompletions.create() got an "
            "unexpected keyword argument 'chat_template_kwargs'", status=500))

    def test_an_ordinary_500_is_still_not_this(self):
        """The body check is what keeps it narrow: a real server fault must
        stay retryable."""
        from controller.inference.openai_backend import OpenAICompatBackend
        for body in ("internal server error", "upstream connect error",
                     "CUDA out of memory"):
            assert not OpenAICompatBackend._is_template_kwargs_rejection(
                self._err(body, status=500)), body

    def test_a_502_from_a_dead_worker_is_still_retryable(self):
        from controller.inference.openai_backend import OpenAICompatBackend
        assert not OpenAICompatBackend._is_template_kwargs_rejection(
            self._err("Bad Gateway", status=502))
