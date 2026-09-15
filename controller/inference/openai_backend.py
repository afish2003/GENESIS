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
        self.api_key = api_key or None
        self.json_mode = json_mode

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if extra_headers:
            headers.update(extra_headers)

        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout, connect=30.0),
            headers=headers,
        )

    async def complete(
        self,
        messages: list[Message],
        temperature: float = 0.7,
    ) -> InferenceResult:
        """Send a chat completion to {base_url}/chat/completions."""
        payload: dict = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "stream": False,
        }
        if self.json_mode:
            payload["response_format"] = {"type": "json_object"}

        response = await self._request_with_retry(payload)
        data = response.json()

        choices = data.get("choices") or []
        if not choices:
            raise ValueError(
                f"No choices in response from {self.base_url}: {str(data)[:200]}"
            )
        content = (choices[0].get("message") or {}).get("content") or ""

        usage = data.get("usage") or {}

        return InferenceResult(
            content=content,
            model=data.get("model", self.model),
            total_duration_ms=None,  # not reported by the OpenAI schema
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
        )

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
