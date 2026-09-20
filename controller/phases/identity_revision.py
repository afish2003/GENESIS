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

{evaluation_feedback}Based on this cycle's events, does your identity statement still accurately reflect who you are? If so, return it unchanged. If not, provide an updated version with a summary of what changed and why.

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
    events = cycle.pending_events

    for agent_id in config.agents:
        ctx = contexts[agent_id]
        current_identity = world.identities[agent_id].content

        messages = [
            ctx.build_system_message(),
        ]
        for msg in ctx.get_discussion_messages():
            messages.append(msg)
        messages.append(Message(
            role="user",
            content=IDENTITY_PROMPT.format(
                current_identity=current_identity,
                evaluation_feedback=(
                    f"## This cycle's evaluation\n\n{fb}\n\n"
                    if (fb := cycle.evaluation_feedback()) else ""
                ),
            ),
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

            contexts[agent_id].cycle_events.append(
                f"You revised your identity statement: {output.changes_summary[:160]}"
            )
            events.append(EventEnvelope(
                event_type=EventType.IDENTITY_REVISED,
                run_id=config.run_id,
                condition=config.condition.value,
                cycle_id=cycle.cycle_id,
                agent_id=agent_id,
                payload={
                    "changes_summary": output.changes_summary,
                    "version": world.identities[agent_id].version,
                    # The statement itself, and the one it replaced.
                    #
                    # Without these, the April proposal's two headline identity
                    # metrics cannot be computed from the logs AT ALL — M1
                    # (Identity Continuity, cosine between consecutive identity
                    # embeddings) and M5 (Convergence Index, cosine between the
                    # two agents' statements) are both defined over this text.
                    # Nine runs were collected carrying only a summary and a
                    # version number, so for those runs the measurement is
                    # permanently impossible. Doctrine got this right —
                    # doctrine_diffs.jsonl stores revised_content — and identity
                    # did not.
                    "identity_text": output.updated_identity,
                    "previous_identity_text": current_identity,
                },
            ))

    return events
