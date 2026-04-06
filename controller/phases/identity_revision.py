"""Phase 11: Identity Revision — each agent updates their own identity independently."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from controller.inference.backend import Message
from controller.logging.schemas import EventEnvelope, EventType
from controller.phases.schemas import IdentityRevisionOutput

if TYPE_CHECKING:
    from controller.agents.base import AgentContext
    from controller.config import RunConfig
    from controller.cycle import CycleState
    from controller.inference.backend import InferenceBackend
    from controller.logging.logger import AppendOnlyJSONLLogger
    from controller.world.state import WorldState

_logger = logging.getLogger(__name__)

IDENTITY_PROMPT = """This is the identity revision phase. Review your current identity statement and decide whether to update it.

Your current identity statement:

{current_identity}

Based on this cycle's events — discussion, evaluation feedback, doctrine changes, scenario events — does your identity statement still accurately reflect who you are? If so, return it unchanged. If not, provide an updated version with a summary of what changed and why.

Identity revisions should be genuine, not cosmetic. Change your identity when your experience warrants it."""


async def execute(
    config: RunConfig,
    backend: InferenceBackend,
    world: WorldState,
    cycle: CycleState,
    contexts: dict[str, AgentContext],
    logger: AppendOnlyJSONLLogger,
) -> list[EventEnvelope]:
    """Each agent independently revises their identity statement."""
    events = []

    for agent_id in ["axiom", "flux"]:
        ctx = contexts[agent_id]
        current_identity = world.identities[agent_id].content

        messages = [
            ctx.build_system_message(),
        ]
        for msg in ctx.get_discussion_messages():
            messages.append(msg)
        messages.append(Message(
            role="user",
            content=IDENTITY_PROMPT.format(current_identity=current_identity),
        ))

        output = await backend.complete_structured(
            messages=messages,
            response_schema=IdentityRevisionOutput,
            temperature=config.temperature_discussion,
            max_retries=config.max_retries,
        )
        output.agent_id = agent_id

        # Update world state if changed
        if output.updated_identity != current_identity:
            world.identities[agent_id].content = output.updated_identity
            world.identities[agent_id].last_modified_cycle = cycle.cycle_id
            world.identities[agent_id].version += 1

            events.append(EventEnvelope(
                event_type=EventType.IDENTITY_REVISED,
                run_id=config.run_id,
                condition=config.condition.value,
                cycle_id=cycle.cycle_id,
                agent_id=agent_id,
                payload={
                    "changes_summary": output.changes_summary,
                    "version": world.identities[agent_id].version,
                },
            ))

    return events
