"""OpenAI-compatible inference backend.

Speaks the /v1/chat/completions dialect, which is the de facto standard.
The same class therefore covers, by changing only `base_url` and `model`:

    OpenAI          https://api.openai.com/v1
    Ollama          http://<host>:11434/v1
    LM Studio       http://localhost:1234/v1
    llama.cpp       http://localhost:8080/v1
    vLLM            http://<host>:8000/v1
    Groq            https://api.groq.com/openai/v1
    OpenRouter      https://openrouter.ai/api/v1
    Together        https://api.together.xyz/v1

Local servers usually ignore the API key; hosted ones require it. If no key
is configured the Authorization header is omitted entirely.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Optional

import httpx

from controller.inference.backend import InferenceBackend, InferenceResult, Message

logger = logging.getLogger(__name__)

#: Transient transport failures worth retrying. Only ConnectError and
#: ConnectTimeout were caught, but with a 600 s ceiling against a local 32B
#: model the realistic failure is a ReadTimeout — which is not a subclass of
#: either, so it propagated out of the retry loop, out of complete_structured,
#: out of the phase, and was swallowed by _run_phase.
_RETRYABLE = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.ReadTimeout,
    httpx.WriteTimeout,
    httpx.PoolTimeout,
    httpx.ReadError,
    httpx.WriteError,
    httpx.RemoteProtocolError,
)

# Connection retry settings — mirrors OllamaBackend
_RETRY_DELAYS = [5, 15, 45]  # seconds — exponential backoff


class OpenAICompatBackend(InferenceBackend):
    """Concrete backend for any OpenAI-compatible /v1/chat/completions endpoint."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434/v1",
        model: str = "qwen2.5:7b-instruct",
        api_key: Optional[str] = None,
        timeout: float = 600.0,
        max_output_tokens: int = 4096,
        json_mode: bool = False,
        extra_headers: Optional[dict[str, str]] = None,
    ) -> None:
        """
        Args:
            base_url: Endpoint root, including /v1. Trailing slash optional.
            model: Model identifier as the endpoint expects it.
            api_key: Bearer token. Omitted from requests when None/empty.
            timeout: Per-request timeout in seconds.
            json_mode: Send response_format={"type": "json_object"}. Improves
                structured-output reliability where supported, but not every
                server accepts it — leave off unless the endpoint is known good.
            extra_headers: Additional headers (e.g. OpenRouter attribution).
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.api_key = api_key or None
        self.json_mode = json_mode
        #: Cleared permanently the first time the endpoint rejects
        #: response_format, so one 400 costs one extra request, not one per call.
        self._json_mode_supported = True

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if extra_headers:
            headers.update(extra_headers)

        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout, connect=30.0),
            headers=headers,
        )

    @property
    def prefers_json_mode(self) -> bool:
        """Constrain structured calls when the endpoint supports it."""
        return self.json_mode and self._json_mode_supported

    async def complete(
        self,
        messages: list[Message],
        temperature: float = 0.7,
        force_json: bool = False,
    ) -> InferenceResult:
        """Send a chat completion to {base_url}/chat/completions."""
        payload: dict = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "stream": False,
            # Nothing capped this. A model that fails to stop generates until
            # the context window is full: one evaluation call ran 19 minutes
            # against a 32k judge before being killed, and a truncated JSON
            # response then surfaces as an unexplained parse failure.
            "max_tokens": self.max_output_tokens,
        }
        # Only when the CALLER wants JSON. `self.json_mode` used to be OR'd in
        # here, which applied response_format to every call the backend made —
        # including retrieval's prose summariser, which then returned "{}".
        # Structured calls opt in through complete_structured via
        # prefers_json_mode; free-form calls never do.
        want_json = force_json and self._json_mode_supported
        if want_json:
            payload["response_format"] = {"type": "json_object"}

        if self.stream_sink is not None:
            streamed = await self._complete_streaming(payload)
            if streamed is not None:
                return streamed
            # Streaming failed; fall through to the ordinary request rather
            # than failing the phase. A view feature must never cost a run.

        try:
            response = await self._request_with_retry(payload)
        except httpx.HTTPStatusError as e:
            # `response_format` is widely but not universally supported, and it
            # is now on by default because it is what makes small models emit
            # parseable JSON at all. An endpoint that rejects it must degrade to
            # unconstrained decoding rather than failing the run — the results
            # are worse, not absent, and the warning says which.
            if not (want_json and self._is_response_format_rejection(e)):
                raise
            self._json_mode_supported = False
            logger.warning(
                "%s rejected response_format={'type':'json_object'}; continuing "
                "without it for the rest of this run. Structured phases will "
                "rely on the parse retries instead, which is measurably worse "
                "for small models.", self.base_url,
            )
            payload.pop("response_format", None)
            response = await self._request_with_retry(payload)
        data = response.json()

        choices = data.get("choices") or []
        if not choices:
            raise ValueError(
                f"No choices in response from {self.base_url}: {str(data)[:200]}"
            )
        content = (choices[0].get("message") or {}).get("content") or ""

        usage = data.get("usage") or {}

        # Hitting the cap means the response is cut mid-token-stream, so a
        # structured call will fail to parse for a reason that looks like the
        # model being bad at JSON. Say which it is.
        if (choices[0].get("finish_reason") or "") == "length":
            logger.warning(
                "Response hit the %d-token output cap and was truncated. "
                "Structured output will not parse. Raise max_output_tokens if "
                "this phase legitimately needs more.", self.max_output_tokens,
            )

        return InferenceResult(
            content=content,
            model=data.get("model", self.model),
            total_duration_ms=None,  # not reported by the OpenAI schema
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
        )

    async def _complete_streaming(self, payload: dict) -> Optional[InferenceResult]:
        """Stream the completion, feeding deltas to `stream_sink` as they land.

        Returns None if streaming did not work, so the caller can fall back to a
        normal request. Deliberately not retried: this path exists to make a run
        watchable, and a view feature must never be the reason a run fails.

        Note the sink is called from inside the response loop, so it must be
        cheap and must not raise — the orchestrator's sink appends to a file and
        swallows its own errors.
        """
        url = f"{self.base_url}/chat/completions"
        body = {**payload, "stream": True}
        chunks: list[str] = []
        usage: dict = {}
        model = self.model

        try:
            async with self._client.stream("POST", url, json=body) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if not data or data == "[DONE]":
                        continue
                    try:
                        parsed = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    model = parsed.get("model", model)
                    usage = parsed.get("usage") or usage
                    for choice in parsed.get("choices") or []:
                        piece = (choice.get("delta") or {}).get("content")
                        if piece:
                            chunks.append(piece)
                            self._emit(piece)
        except Exception as e:
            logger.warning(
                "Streaming failed against %s (%s: %s); falling back to a "
                "blocking request for this call.", self.base_url, type(e).__name__, e,
            )
            return None

        if not chunks:
            return None

        content = "".join(chunks)
        return InferenceResult(
            content=content,
            model=model,
            total_duration_ms=None,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
        )

    def _emit(self, delta: str) -> None:
        """Hand one delta to the sink, never letting it break generation."""
        sink = self.stream_sink
        if sink is None:
            return
        try:
            sink(delta)
        except Exception:
            logger.debug("stream_sink raised; dropping it for this run",
                         exc_info=True)
            self.stream_sink = None

    @staticmethod
    def _is_response_format_rejection(error: "httpx.HTTPStatusError") -> bool:
        """Whether a 4xx is about response_format rather than anything else.

        Checked by message, because the OpenAI-compatible ecosystem has no
        shared error code for an unsupported parameter. Narrow on purpose: a
        400 for a bad model name must not be mistaken for this and silently
        turn JSON mode off.
        """
        if error.response.status_code not in (400, 404, 422, 501):
            return False
        body = (error.response.text or "").lower()
        return "response_format" in body or "json_object" in body

    async def _request_with_retry(self, payload: dict) -> httpx.Response:
        """POST with exponential backoff on connection errors, 5xx, and 429."""
        url = f"{self.base_url}/chat/completions"
        last_error: Exception | None = None

        for attempt, delay in enumerate([0] + _RETRY_DELAYS):
            if delay > 0:
                logger.warning(
                    "Inference attempt %d failed, retrying in %ds...", attempt, delay
                )
                await asyncio.sleep(delay)

            try:
                response = await self._client.post(url, json=payload)
                response.raise_for_status()
                return response
            except _RETRYABLE as e:
                last_error = e
                logger.warning("Connection error (attempt %d): %s", attempt + 1, e)
            except httpx.HTTPStatusError as e:
                status = e.response.status_code
                # 429 is transient; other 4xx are caller errors and must not retry.
                if status != 429 and status < 500:
                    logger.error(
                        "Non-retryable HTTP %d from %s: %s",
                        status,
                        url,
                        e.response.text[:300],
                    )
                    raise
                last_error = e
                logger.warning("HTTP %d (attempt %d), will retry", status, attempt + 1)

        raise ConnectionError(
            f"Failed to reach {url} after {len(_RETRY_DELAYS) + 1} attempts: {last_error}"
        )

    async def health_check(self) -> bool:
        """Check the endpoint is reachable via GET {base_url}/models."""
        try:
            response = await self._client.get(f"{self.base_url}/models")
            return response.status_code == 200
        except httpx.HTTPError:
            # Any transport failure means 'not reachable', never a raise.
            return False

    async def close(self) -> None:
        await self._client.aclose()
