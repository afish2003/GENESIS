"""Tests for getting parseable JSON out of a small model.

The `code` task asks a model to put a complete source file — quotes, newlines,
backslashes — inside a JSON string field. That is the hardest thing you can ask
a 7b model to emit, and on 2026-09-15 it failed a real run outright: qwen2.5:7b
opened the `content` field with a Python triple-quote, which is valid Python and
not JSON, three times running, and `protocol_design` raised.

Measured on the real model before fixing anything:

    temperature 0.7, json_mode off   1/6 parsed
    temperature 0.3, json_mode off   1/6 parsed
    temperature 0.7, json_mode ON    5/6 parsed

Temperature is not the variable. Constraining the decoder is. These tests pin
that conclusion in the code, because the obvious fix — "turn the temperature
down for structured calls" — is wrong and looks right.
"""

from __future__ import annotations

import httpx
import pytest
from pydantic import BaseModel

from controller.config import RunConfig
from controller.inference.backend import InferenceBackend, InferenceResult, Message
from controller.inference.openai_backend import OpenAICompatBackend


class Tiny(BaseModel):
    value: str


class RecordingBackend(InferenceBackend):
    """Replays scripted responses and records how each call was made."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    async def complete(self, messages, temperature=0.7, force_json=False, speaker=None):
        self.calls.append({"temperature": temperature, "force_json": force_json})
        content = self.responses[min(len(self.calls) - 1, len(self.responses) - 1)]
        return InferenceResult(content=content, model="test", total_duration_ms=0,
                               prompt_tokens=0, completion_tokens=0)

    async def health_check(self) -> bool:
        return True

    async def close(self) -> None:
        return None


class TestRetryLadder:
    def test_first_attempt_uses_the_requested_temperature(self):
        """The phases that ask at 0.7 are meant to be the inventive ones."""
        assert InferenceBackend.retry_temperature(0.7, 0) == 0.7

    def test_retries_step_down_and_never_up(self):
        assert InferenceBackend.retry_temperature(0.7, 1) == 0.2
        assert InferenceBackend.retry_temperature(0.7, 2) == 0.0

    def test_a_low_request_is_not_raised_on_retry(self):
        assert InferenceBackend.retry_temperature(0.1, 1) == 0.1

    @pytest.mark.asyncio
    async def test_first_attempt_is_unconstrained(self):
        be = RecordingBackend(['{"value": "ok"}'])
        await be.complete_structured([Message(role="user", content="x")], Tiny)
        assert be.calls == [{"temperature": 0.3, "force_json": False}]

    @pytest.mark.asyncio
    async def test_retries_constrain_the_decoder(self):
        """The measured fix. Without this a retry re-rolls the same dice."""
        be = RecordingBackend(['{"value": """oops"""}', '{"value": "ok"}'])
        out = await be.complete_structured(
            [Message(role="user", content="x")], Tiny, temperature=0.7)
        assert out.value == "ok"
        assert [c["force_json"] for c in be.calls] == [False, True]
        assert [c["temperature"] for c in be.calls] == [0.7, 0.2]

    @pytest.mark.asyncio
    async def test_a_retry_that_succeeds_is_still_counted(self):
        """A run that only parsed on retry is a run the parser shaped."""
        before = InferenceBackend.structured_retries
        be = RecordingBackend(['not json', '{"value": "ok"}'])
        await be.complete_structured([Message(role="user", content="x")], Tiny)
        assert InferenceBackend.structured_retries == before + 1

    @pytest.mark.asyncio
    async def test_exhausting_retries_raises_with_the_raw_response(self):
        be = RecordingBackend(['{"value": """still bad"""}'])
        with pytest.raises(ValueError, match="still bad"):
            await be.complete_structured([Message(role="user", content="x")], Tiny)
        assert len(be.calls) == 3


class TestJsonModeDefault:
    def test_json_mode_is_on_by_default(self):
        """1/6 vs 5/6 on the real model. Off was not a neutral default."""
        assert RunConfig(run_id="J", condition="BASELINE").api_json_mode is True

    def test_a_backend_starts_believing_the_endpoint_supports_it(self):
        be = OpenAICompatBackend(base_url="http://x/v1", model="m", json_mode=True)
        assert be._json_mode_supported is True


class TestEndpointsThatRejectJsonMode:
    """Not every OpenAI-compatible endpoint supports response_format.

    Since it is now ON by default, one that rejects it must degrade rather than
    fail the run — worse results, not absent ones.
    """

    def _error(self, status: int, body: str) -> httpx.HTTPStatusError:
        request = httpx.Request("POST", "http://x/v1/chat/completions")
        response = httpx.Response(status, text=body, request=request)
        return httpx.HTTPStatusError("boom", request=request, response=response)

    @pytest.mark.parametrize("status", [400, 404, 422, 501])
    def test_recognises_a_response_format_rejection(self, status):
        err = self._error(status, "unknown parameter: response_format")
        assert OpenAICompatBackend._is_response_format_rejection(err) is True

    def test_recognises_the_json_object_wording(self):
        err = self._error(400, "json_object is not supported by this model")
        assert OpenAICompatBackend._is_response_format_rejection(err) is True

    def test_an_unrelated_400_is_not_mistaken_for_one(self):
        """A bad model name must not silently switch JSON mode off for the run."""
        err = self._error(400, "model 'qwen2.5:99b' not found")
        assert OpenAICompatBackend._is_response_format_rejection(err) is False

    def test_a_server_error_is_not_mistaken_for_one(self):
        err = self._error(500, "response_format internal error")
        assert OpenAICompatBackend._is_response_format_rejection(err) is False

