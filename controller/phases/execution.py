"""Optional phase: run this cycle's artifact and record what it did.

This is the half of "see what they program" that storing a file does not
deliver. Until it existed, `CodeTask` produced modules that were read and scored
as prose; nothing ever found out whether they ran.

Placement matters. The phase sits between `protocol_design` and `evaluation` so
the evaluator scores `correctness` against the execution result rather than
against how the code reads (`controller/tasks/code.py`). Running it after
evaluation would still log the result, and the score would still be a guess.

Three properties this phase is responsible for:

- **It never refuses silently.** A refusal (no sandbox configured) is a logged
  CODE_EXECUTION outcome like any other, and reaches the evaluator prompt. An
  evaluator told nothing about execution scores as if the code had run and
  passed.
- **Agent code cannot fail the cycle.** `ExecutionSandbox.run` is contracted
  never to raise over what the code did; a crash, a timeout or an OOM comes back
  as an outcome. Anything that does raise here is the sandbox itself being
  broken, and is caught so a wedged container runtime cannot take the run down.
- **The entrypoint comes from the task, not the model.** See
  `Task.execution_request`.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from controller.logging.schemas import EventEnvelope, EventType
from controller.sandbox.schemas import ExecutionOutcome, ExecutionResult
from controller.tasks import create_task

if TYPE_CHECKING:
    from controller.config import RunConfig
    from controller.cycle import CycleState
    from controller.inference.backend import InferenceBackend
    from controller.logging.logger import AppendOnlyJSONLLogger
    from controller.world.state import WorldState

_logger = logging.getLogger(__name__)

#: How much of each stream reaches the evaluator's prompt. The sandbox already
#: caps output at 64 KiB for the log; that is still far too much to paste into
#: a context window beside the source it came from.
MAX_PROMPT_OUTPUT_CHARS = 2000


async def execute(
    config: RunConfig,
    backend: InferenceBackend,
    world: WorldState,
    cycle: CycleState,
    logger: AppendOnlyJSONLLogger,
) -> list[EventEnvelope]:
    """Run the cycle's artifact in the sandbox and record the result."""
    events = cycle.pending_events
    task = create_task(config)

    request = task.execution_request(world, cycle, config.sandbox_timeout_seconds)
    if request is None:
        _logger.debug(
            "Cycle %d: task %r has nothing to execute", cycle.cycle_id, task.name
        )
        return events

    sandbox = cycle.sandbox
    if sandbox is None:
        # Reachable only by constructing an orchestrator without one. Treated as
        # a refusal rather than an error so the shape of the record is the same
        # as a null-backend run.
        result = ExecutionResult(
            outcome=ExecutionOutcome.REFUSED,
            detail="No sandbox was provided to the orchestrator.",
        )
    else:
        try:
            result = await sandbox.run(request)
        except Exception as e:  # the sandbox itself, not the agents' code
            _logger.exception("Cycle %d: sandbox raised", cycle.cycle_id)
            result = ExecutionResult(
                outcome=ExecutionOutcome.SANDBOX_ERROR,
                detail=f"{type(e).__name__}: {e}",
            )

    _logger.info(
        "Cycle %d: ran %s -> %s (exit %s) in %.2fs",
        cycle.cycle_id, request.entrypoint, result.outcome.value,
        result.exit_code, result.duration_seconds,
    )

    # Evaluation reads this. Trimmed here rather than in the task so the full
    # output still reaches executions.jsonl.
    cycle.execution_result = _for_prompt(result)

    events.append(EventEnvelope(
        event_type=EventType.CODE_EXECUTION,
        run_id=config.run_id,
        condition=config.condition.value,
        cycle_id=cycle.cycle_id,
        agent_id=request.requested_by or None,
        payload={
            "task": task.name,
            "artifact_id": (cycle.proposed_protocol or {}).get("protocol_id", ""),
            "entrypoint": request.entrypoint,
            "files": sorted(request.files),
            "timeout_seconds": request.timeout_seconds,
            "sandbox_backend": config.sandbox_backend.value,
            **result.model_dump(mode="json"),
        },
    ))

    # A sandbox that is broken rather than strict is worth surfacing on its own:
    # every execution refused or erroring looks, in the score data, exactly like
    # agents who write code that does not run.
    if result.outcome in (ExecutionOutcome.REFUSED, ExecutionOutcome.SANDBOX_ERROR):
        events.append(EventEnvelope(
            event_type=EventType.NOTABLE_EVENT,
            run_id=config.run_id,
            condition=config.condition.value,
            cycle_id=cycle.cycle_id,
            payload={
                "kind": "execution_unavailable",
                "outcome": result.outcome.value,
                "detail": result.detail,
            },
        ))

    return events


def _for_prompt(result: ExecutionResult) -> dict:
    """The execution result as the evaluator sees it, with output clipped."""
    data = result.model_dump(mode="json")
    for stream in ("stdout", "stderr"):
        text = data.get(stream) or ""
        if len(text) > MAX_PROMPT_OUTPUT_CHARS:
            data[stream] = text[:MAX_PROMPT_OUTPUT_CHARS] + "\n...[clipped]"
            data["truncated"] = True
    return data
