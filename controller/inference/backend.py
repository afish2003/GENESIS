"""Abstract inference backend interface."""

from __future__ import annotations

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
        for attempt in range(1 + max_retries):
            result = await self.complete(augmented, temperature=temperature)

            # Strip markdown fencing if present
            content = result.content.strip()
            if content.startswith("```"):
                lines = content.split("\n")
                # Remove first line (```json) and last line (```) only
                if lines and lines[-1].strip().startswith("```"):
                    lines = lines[1:-1]
                else:
                    lines = lines[1:]
                content = "\n".join(lines)

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

        raise ValueError(
            f"Failed to parse response as {response_schema.__name__} "
            f"after {1 + max_retries} attempts: {last_error}"
        )

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the backend is reachable and ready."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Clean up resources (e.g. HTTP client)."""
        ...
