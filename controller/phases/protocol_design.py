"""Phase 7: Sandbox design — the lead agent produces this run's task artifact.

Task-agnostic. What gets built is decided by controller/tasks/, selected with
config.task; this module only orchestrates.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from controller.inference.backend import Message
from controller.logging.schemas import EventEnvelope, EventType
from controller.tasks import create_task

if TYPE_CHECKING:
    from controller.agents.base import AgentContext
    from controller.config import RunConfig
    from controller.cycle import CycleState
    from controller.inference.backend import InferenceBackend
    from controller.logging.logger import AppendOnlyJSONLLogger
    from controller.world.state import WorldState

_logger = logging.getLogger(__name__)


async def execute(
    config: RunConfig,
    backend: InferenceBackend,
    world: WorldState,
    cycle: CycleState,
    contexts: dict[str, AgentContext],
    logger: AppendOnlyJSONLLogger,
) -> list[EventEnvelope]:
    """Have the lead agent produce the task artifact for this cycle."""
    events: list[EventEnvelope] = cycle.pending_events
    task = create_task(config)

    # The first agent on the roster drafts; the others get their say in
    # evaluation and doctrine revision.
    lead_agent = config.agents[0]
    ctx = contexts[lead_agent]

    messages = [ctx.build_system_message()]
    for msg in ctx.get_discussion_messages():
        messages.append(msg)
    messages.append(Message(
        role="user",
        content=task.design_prompt(config, world, cycle),
    ))

    output = await backend.complete_structured(
        messages=messages,
        response_schema=task.output_schema(),
        temperature=config.temperature_discussion,
        max_retries=config.max_retries,
    )
    if hasattr(output, "proposing_agent"):
        output.proposing_agent = lead_agent

    artifact_id = task.apply(world, output, cycle)

    # Evaluation reads this; keep a task-neutral id alongside the raw output.
    cycle.proposed_protocol = {**output.model_dump(), "protocol_id": artifact_id}

    _logger.info("Cycle %d: %s produced %s %r",
                 cycle.cycle_id, lead_agent, task.artifact_noun, artifact_id)

    events.append(EventEnvelope(
        event_type=EventType.PROTOCOL_PROPOSED,
        run_id=config.run_id,
        condition=config.condition.value,
        cycle_id=cycle.cycle_id,
        agent_id=lead_agent,
        payload={"task": task.name, **output.model_dump()},
    ))

    return events
