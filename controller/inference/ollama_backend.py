"""Ollama inference backend — calls /api/chat over HTTP on the LAN."""

from __future__ import annotations

import logging
import time

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

# Connection retry settings
_RETRY_DELAYS = [5, 15, 45]  # seconds — exponential backoff


class OllamaBackend(InferenceBackend):
    """Concrete backend for Ollama served on the HP Omen over LAN.

    Uses /api/chat with the chat-completion message format.
    """

    def __init__(
        self,
        host: str = "http://192.168.1.100:11434",
        model: str = "qwen2.5:32b-instruct-q4_K_M",
        timeout: float = 600.0,
        max_output_tokens: int = 4096,
        enable_thinking: bool = False,
    ) -> None:
        self.host = host.rstrip("/")
        self.model = model
        self.max_output_tokens = max_output_tokens
        # The factory passes this for every backend. Omitting it here was a
        # TypeError at prepare_run, before cycle 0, on the DEFAULT backend —
        # only the openai and mock paths had factory tests.
        self.enable_thinking = enable_thinking
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=30.0))

    async def complete(
        self,
        messages: list[Message],
        temperature: float = 0.7,
        force_json: bool = False,
        speaker: str | None = None,
    ) -> InferenceResult:
        """Send chat completion to Ollama /api/chat."""
        payload = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
            "options": {
                "temperature": temperature,
                # num_predict was absent, so max_output_tokens was stored and
                # never sent: Ollama generation stayed unbounded on the very
                # backend the cap was added for. -1 is Ollama's "no limit".
                "num_predict": self.max_output_tokens,
            },
            # Ollama's own switch for reasoning models. With thinking on, a
            # call spends its whole budget reasoning and returns empty content.
            "think": self.enable_thinking,
        }

        response = await self._request_with_retry(payload)
        data = response.json()

        # Extract from Ollama response format
        content = data.get("message", {}).get("content", "")
        total_duration = data.get("total_duration")  # nanoseconds
        prompt_eval_count = data.get("prompt_eval_count")
        eval_count = data.get("eval_count")

        return InferenceResult(
            content=content,
            model=data.get("model", self.model),
            total_duration_ms=int(total_duration / 1_000_000) if total_duration else None,
            prompt_tokens=prompt_eval_count,
            completion_tokens=eval_count,
        )

    async def _request_with_retry(self, payload: dict) -> httpx.Response:
        """POST to /api/chat with exponential backoff retry on connection failure."""
        url = f"{self.host}/api/chat"
        last_error: Exception | None = None

        for attempt, delay in enumerate([0] + _RETRY_DELAYS):
            if delay > 0:
                logger.warning(
                    "Ollama connection attempt %d failed, retrying in %ds...",
                    attempt,
                    delay,
                )
                # Use synchronous sleep — this is a retry delay, not parallelism
                import asyncio
                await asyncio.sleep(delay)

            try:
                response = await self._client.post(url, json=payload)
                response.raise_for_status()
                return response
            except _RETRYABLE as e:
                last_error = e
                logger.warning("Ollama connection error (attempt %d): %s", attempt + 1, e)
            except httpx.HTTPStatusError as e:
                # Don't retry on 4xx errors
                if e.response.status_code < 500:
                    raise
                last_error = e
                logger.warning("Ollama server error (attempt %d): %s", attempt + 1, e)

        raise ConnectionError(
            f"Failed to connect to Ollama at {url} after {len(_RETRY_DELAYS) + 1} attempts: {last_error}"
        )

    async def health_check(self) -> bool:
        """Check Ollama is reachable via GET /api/tags."""
        try:
            response = await self._client.get(f"{self.host}/api/tags")
            return response.status_code == 200
        except httpx.HTTPError:
            # Any transport failure means 'not reachable', never a raise.
            return False

    async def close(self) -> None:
        await self._client.aclose()
