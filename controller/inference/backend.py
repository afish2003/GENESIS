"""Abstract inference backend interface."""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Type, TypeVar

from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class Message(BaseModel):
    """A single message in a chat conversation."""

    role: str  # "system", "user", "assistant"
    content: str


class InferenceResult(BaseModel):
    """Raw result from an inference call."""

    content: str
    model: str
    total_duration_ms: int | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


_FENCE = re.compile(r"```[a-zA-Z0-9_+-]*\s*\n?(.*?)\n?\s*```", re.DOTALL)


def extract_json(raw: str) -> str:
    """Pull the JSON payload out of a model response.

    The previous logic split on newlines and dropped the first and last. For a
    single-line fenced response — ```{"a":1}``` — the first line both starts
    with a fence and ends with one, so lines[1:-1] was empty and the content
    became "", failing validation on all three attempts with an unhelpful
    error. Prose before the fence defeated stripping entirely, since the check
    was startswith("```").
    """
    text = (raw or "").strip()

    match = _FENCE.search(text)
    if match:
        inner = match.group(1).strip()
        if inner:
            return inner

    # No usable fence. Fall back to the outermost JSON object or array, which
    # also handles a model that wrapped its answer in prose.
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end > start:
            return text[start:end + 1].strip()

    return text


class InferenceBackend(ABC):
    """Abstract interface for LLM inference.

    Version 1: OllamaBackend (HTTP to HP Omen on LAN).
    Future: VLLMBackend for Lambda server with dual RTX 4090s.
    """

    #: Structured calls that needed at least one retry to parse. A run where
    #: this is large is a run whose artifacts were shaped by the parser rather
    #: than by the agents, which is worth knowing before reading the results.
    structured_retries: int = 0
    structured_failures: int = 0

    @property
    def prefers_json_mode(self) -> bool:
        """Whether this backend should constrain decoding on structured calls.

        Separate from `force_json` because the two questions are different:
        "does this endpoint support JSON mode" is a property of the backend,
        "does THIS call want JSON" is a property of the caller. Conflating them
        applied response_format to free-form calls as well — a prose summary
        asked for under json_object comes back as `{}`.
        """
        return False

    #: Set by the orchestrator to receive generation deltas as they arrive.
    #: When it is None the backend makes an ordinary blocking request, so
    #: nothing about a normal run changes. Its only purpose is watchability:
    #: a phase otherwise appears as nine seconds of silence followed by a
    #: finished block, and you never see the agents think.
    #: Receives (delta, speaker). `speaker` is who the text is coming from —
    #: an agent id, or "evaluator", or None where nobody is speaking as
    #: anyone. Without it the live feed is a wall of text during a two-agent
    #: discussion and you cannot tell which of them is talking, which is the
    #: single thing you most want to know while watching them argue.
    stream_sink: "Optional[Callable[[str, Optional[str]], None]]" = None

    @abstractmethod
    async def complete(
        self,
        messages: list[Message],
        temperature: float = 0.7,
        force_json: bool = False,
        speaker: "Optional[str]" = None,
    ) -> InferenceResult:
        """Send a chat completion request and return the raw result.

        `force_json` asks the endpoint to constrain decoding to valid JSON where
        it supports that. Advisory: backends that cannot do it ignore it.

        `speaker` is passed through to `stream_sink` and affects nothing else.
        Threaded explicitly through every call site rather than carried in a
        contextvar: an ambient "current speaker" is exactly the implicit
        control flow this project rules out, and it would silently mislabel
        the feed the first time a phase built one agent's messages and called
        for another.
        """
        ...

    @staticmethod
    def retry_temperature(requested: float, attempt: int) -> float:
        """Temperature for a structured retry. Steps down; never up.

        NOT the fix for unparseable JSON, and the first version of this
        docstring said it was. Measured on qwen2.5:7b asked to put a Python
        source file in a JSON string field: 1/6 parsed at temperature 0.7 and
        1/6 at 0.3. Temperature does not move it. What moves it is constraining
        the decoder — `force_json` below, 5/6.

        Kept anyway, because a retry that changes none of its inputs is just
        re-rolling the same dice, and low temperature is the right setting for
        an attempt whose only job is to be well-formed.
        """
        if attempt <= 0:
            return requested
        if attempt == 1:
            return min(requested, 0.2)
        return 0.0

    async def complete_structured(
        self,
        messages: list[Message],
        response_schema: Type[T],
        temperature: float = 0.3,
        max_retries: int = 2,
        speaker: "Optional[str]" = None,
    ) -> T:
        """Send a chat completion request and parse the response into a Pydantic model.

        Appends a JSON schema instruction to the last message, then validates
        the response. Retries on validation failure up to max_retries times.
        """
        schema_json = response_schema.model_json_schema()
        schema_instruction = (
            "\n\nYou MUST respond with valid JSON matching this exact schema. "
            "Output ONLY the JSON object, no markdown fencing, no extra text.\n"
            f"Schema: {schema_json}"
        )

        # Build messages with schema instruction appended to last user/system message
        augmented = list(messages)
        if augmented:
            last = augmented[-1]
            augmented[-1] = Message(role=last.role, content=last.content + schema_instruction)

        last_error: Exception | None = None
        last_content: str | None = None
        for attempt in range(1 + max_retries):
            result = await self.complete(
                augmented,
                temperature=self.retry_temperature(temperature, attempt),
                # Structured calls are the only ones that want JSON. On a
                # retry, constrain regardless — that is the measured fix for an
                # unparseable response (1/6 -> 5/6 on qwen2.5:7b).
                force_json=self.prefers_json_mode or attempt > 0,
                speaker=speaker,
            )

            last_content = result.content
            content = extract_json(result.content)

            try:
                parsed = response_schema.model_validate_json(content)
                if attempt:
                    InferenceBackend.structured_retries += 1
                    logger.warning(
                        "%s parsed only on attempt %d (temperature %.2f). The "
                        "model's first answer was not valid JSON.",
                        response_schema.__name__, attempt + 1,
                        self.retry_temperature(temperature, attempt),
                    )
                return parsed
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    # Add error feedback for retry
                    augmented.append(Message(role="assistant", content=result.content))
                    augmented.append(Message(
                        role="user",
                        content=(
                            f"Your response failed JSON validation: {e}\n"
                            "Please try again with valid JSON matching the schema exactly."
                        ),
                    ))

        # Include what the model actually said. Without it a parse failure is
        # undiagnosable from the logs.
        InferenceBackend.structured_failures += 1
        raise ValueError(
            f"Failed to parse response as {response_schema.__name__} after "
            f"{1 + max_retries} attempts: {last_error}\n"
            f"Last raw response: {(last_content or '')[:600]!r}"
        )

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the backend is reachable and ready."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Clean up resources (e.g. HTTP client)."""
        ...
