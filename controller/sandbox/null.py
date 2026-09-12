"""NullSandbox — refuses everything. The default.

Execution must never be reachable by default. If no sandbox is configured, an
agent asking to run code gets a clean refusal that is logged like any other
outcome, rather than code running unisolated on the researcher's machine.
"""

from __future__ import annotations

import logging

from controller.sandbox.base import ExecutionSandbox
from controller.sandbox.schemas import ExecutionOutcome, ExecutionRequest, ExecutionResult

logger = logging.getLogger(__name__)


class NullSandbox(ExecutionSandbox):
    def __init__(self) -> None:
        self.refused_count = 0

    async def run(self, request: ExecutionRequest) -> ExecutionResult:
        self.refused_count += 1
        logger.info(
            "NullSandbox refused an execution request from %r (cycle %d, %d file(s))",
            request.requested_by, request.cycle_id, len(request.files),
        )
        return ExecutionResult(
            outcome=ExecutionOutcome.REFUSED,
            detail="Code execution is disabled. No sandbox is configured.",
        )

    async def health_check(self) -> bool:
        return False
