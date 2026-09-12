"""ExecutionSandbox — the interface agent code must pass through.

Mirrors InferenceBackend: an abstract contract with swappable implementations,
so the isolation mechanism can be changed or tested without touching the cycle
loop. NullSandbox is the default, so execution is opt-in and unreachable by
accident.

See docs/containment_design.md for the threat model and required properties.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from controller.sandbox.schemas import ExecutionRequest, ExecutionResult

# Hard ceilings. Per-request values may be lower, never higher.
MAX_OUTPUT_BYTES = 64 * 1024
MAX_TIMEOUT_SECONDS = 300.0


class ExecutionSandbox(ABC):
    """Runs untrusted agent-authored code under isolation."""

    @abstractmethod
    async def run(self, request: ExecutionRequest) -> ExecutionResult:
        """Execute the request. Must never raise because of what the code did.

        Implementations report failure via ExecutionResult.outcome. Raising is
        reserved for the sandbox itself being broken, not for agent code
        crashing, looping, or exhausting a limit.
        """
        ...

    async def health_check(self) -> bool:
        """Whether this sandbox can currently run anything."""
        return True

    async def close(self) -> None:
        return None

    @staticmethod
    def truncate(text: str, limit: int = MAX_OUTPUT_BYTES) -> tuple[str, bool]:
        """Cap output so a chatty program cannot flood the research corpus."""
        raw = text.encode("utf-8", errors="replace")
        if len(raw) <= limit:
            return text, False
        return raw[:limit].decode("utf-8", errors="ignore") + "\n...[truncated]", True
