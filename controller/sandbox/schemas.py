"""Execution request/result types.

Logged like any other artifact: what the agents chose to build is arguably
the most interesting signal this platform could produce.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class ExecutionOutcome(str, Enum):
    OK = "OK"
    NONZERO_EXIT = "NONZERO_EXIT"
    TIMEOUT = "TIMEOUT"
    MEMORY_LIMIT = "MEMORY_LIMIT"
    REFUSED = "REFUSED"           # NullSandbox, or execution disabled
    SANDBOX_ERROR = "SANDBOX_ERROR"


class ExecutionRequest(BaseModel):
    """Code an agent wants run, and how it should be invoked."""

    files: dict[str, str] = Field(
        ..., description="Relative path -> file content. Paths are sanitised by the sandbox."
    )
    entrypoint: str = Field(..., description="Command to run inside the workspace")
    timeout_seconds: float = Field(default=30.0, ge=1.0, le=300.0)
    requested_by: str = Field(default="", description="Agent id, set by the controller")
    cycle_id: int = Field(default=0)


class ExecutionResult(BaseModel):
    """What happened. Always returned — the sandbox never raises on agent code."""

    outcome: ExecutionOutcome
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    duration_seconds: float = 0.0
    truncated: bool = False
    detail: str = Field(default="", description="Why a non-OK outcome occurred")
