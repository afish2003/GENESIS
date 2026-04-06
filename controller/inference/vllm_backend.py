"""vLLM inference backend stub — for future Lambda server migration.

Not implemented in version 1. Exists to confirm the InferenceBackend
abstraction supports a drop-in swap.
"""

from __future__ import annotations

from controller.inference.backend import InferenceBackend, InferenceResult, Message


class VLLMBackend(InferenceBackend):
    """Stub backend for vLLM on a remote server (e.g. Lambda with dual RTX 4090s).

    Version 1 uses OllamaBackend. This class exists to validate the
    abstraction boundary and will be implemented when the migration happens.
    """

    def __init__(self, host: str = "http://localhost:8000", model: str = "qwen2.5:32b") -> None:
        self.host = host
        self.model = model

    async def complete(
        self,
        messages: list[Message],
        temperature: float = 0.7,
    ) -> InferenceResult:
        raise NotImplementedError(
            "VLLMBackend is not implemented in version 1. Use OllamaBackend."
        )

    async def health_check(self) -> bool:
        raise NotImplementedError("VLLMBackend is not implemented in version 1.")

    async def close(self) -> None:
        pass
