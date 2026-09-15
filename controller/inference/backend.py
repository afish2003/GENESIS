"""Abstract inference backend interface."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any, Type, TypeVar

from pydantic import BaseModel

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

    @abstractmethod
    async def complete(
        self,
        messages: list[Message],
        temperature: float = 0.7,
    ) -> InferenceResult:
        """Send a chat completion request and return the raw result."""
        ...

    async def complete_structured(
        self,
        messages: list[Message],
        response_schema: Type[T],
        temperature: float = 0.3,
        max_retries: int = 2,
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
            result = await self.complete(augmented, temperature=temperature)

            last_content = result.content
            content = extract_json(result.content)

            try:
                return response_schema.model_validate_json(content)
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
