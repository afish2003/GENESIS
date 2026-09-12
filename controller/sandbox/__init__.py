"""Execution sandboxes for agent-authored code.

NullSandbox is the default and refuses everything: execution is opt-in and
must never be reachable by accident. See docs/containment_design.md.
"""

from controller.sandbox.base import ExecutionSandbox
from controller.sandbox.null import NullSandbox
from controller.sandbox.schemas import (
    ExecutionOutcome,
    ExecutionRequest,
    ExecutionResult,
)

__all__ = [
    "ExecutionSandbox",
    "NullSandbox",
    "ExecutionRequest",
    "ExecutionResult",
    "ExecutionOutcome",
]


def create_sandbox(config) -> ExecutionSandbox:
    """Build the sandbox named by config.sandbox_backend. Defaults to refusing."""
    from controller.config import SandboxBackend

    if config.sandbox_backend == SandboxBackend.DOCKER:
        from controller.sandbox.docker import DockerSandbox

        return DockerSandbox(
            image=config.sandbox_image,
            runtime=config.sandbox_runtime,
            memory=config.sandbox_memory,
        )
    return NullSandbox()
