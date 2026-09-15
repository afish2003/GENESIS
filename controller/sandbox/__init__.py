"""Execution sandboxes for agent-authored code.

NullSandbox is the default and refuses everything: execution is opt-in and
must never be reachable by accident. See docs/containment_design.md.
"""

from controller.sandbox.base import ExecutionSandbox
from controller.sandbox.null import NullSandbox
from controller.sandbox.project import ProjectVolume, ProjectVolumeError
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
    "ProjectVolume",
    "ProjectVolumeError",
]


def create_sandbox(config) -> ExecutionSandbox:
    """Build the sandbox named by config.sandbox_backend. Defaults to refusing.

    Raises ProjectVolumeError if a project volume is configured and is not a
    safe one. Refusing at startup is the point: a project volume that silently
    fails to mount gives the agents an empty codebase every cycle, which in the
    data is indistinguishable from agents who cannot build anything.
    """
    from controller.config import SandboxBackend
    from controller.sandbox.project import resolve_project_volume

    if config.sandbox_backend == SandboxBackend.DOCKER:
        from controller.sandbox.docker import DockerSandbox

        volume = resolve_project_volume(config)
        return DockerSandbox(
            image=config.sandbox_image,
            runtime=config.sandbox_runtime,
            memory=config.sandbox_memory,
            tmpfs=config.sandbox_tmpfs,
            project_dir=volume.path if volume else None,
        )
    return NullSandbox()
