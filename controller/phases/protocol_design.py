"""Phase 7: Protocol Design — agents jointly propose a protocol document."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from controller.inference.backend import Message
from controller.logging.schemas import EventEnvelope, EventType
from controller.phases.schemas import ProtocolProposalOutput
from controller.world.artifacts import ProtocolDocument

if TYPE_CHECKING:
    from controller.agents.base import AgentContext
    from controller.config import RunConfig
    from controller.cycle import CycleState
    from controller.inference.backend import InferenceBackend
    from controller.logging.logger import AppendOnlyJSONLLogger
    from controller.world.state import WorldState

_logger = logging.getLogger(__name__)

PROTOCOL_PROMPT = """This is the protocol design phase. You and {partner_name} should collaboratively propose either a new protocol document or a revision to an existing one.

Current active protocols: {protocol_list}

A protocol document should include:
- Purpose: What problem or need does this protocol address?
- Scope: What does this protocol cover and not cover?
- Procedure: Step-by-step operational procedure
- Evaluation criteria: How to assess whether the protocol is being followed effectively
- Known limitations: Honest assessment of gaps or weaknesses

Based on the discussion this cycle, the evaluation feedback from previous cycles, and your priorities, propose a protocol action (create or revise).

If revising, specify which protocol and what changes. If creating, propose a complete new document."""


async def execute(
    config: RunConfig,
    backend: InferenceBackend,
    world: WorldState,
    cycle: CycleState,
    contexts: dict[str, AgentContext],
    logger: AppendOnlyJSONLLogger,
) -> list[EventEnvelope]:
    """Have Axiom propose a protocol, informed by the cycle's discussion."""
    events = []

    # Build protocol list
    active = [f"- {pid}: {p.title}" for pid, p in world.protocols.items() if not p.archived]
    protocol_list = "\n".join(active) if active else "(none yet)"

    # Axiom leads protocol proposals
    ctx = contexts["axiom"]
    messages = [
        ctx.build_system_message(),
    ]
    # Include discussion history for context
    for msg in ctx.get_discussion_messages():
        messages.append(msg)
    messages.append(Message(
        role="user",
        content=PROTOCOL_PROMPT.format(
            partner_name="Flux",
            protocol_list=protocol_list,
        ),
    ))

    output = await backend.complete_structured(
        messages=messages,
        response_schema=ProtocolProposalOutput,
        temperature=config.temperature_discussion,
        max_retries=config.max_retries,
    )
    output.proposing_agent = "axiom"

    # Store in cycle state for evaluation
    cycle.proposed_protocol = output.model_dump()

    # Update world state
    if output.action.value == "create":
        world.protocols[output.protocol_id] = ProtocolDocument(
            protocol_id=output.protocol_id,
            title=output.title,
            content=output.content,
            version=1,
            created_cycle=cycle.cycle_id,
            last_modified_cycle=cycle.cycle_id,
        )
    elif output.protocol_id in world.protocols:
        proto = world.protocols[output.protocol_id]
        proto.content = output.content
        proto.title = output.title
        proto.version += 1
        proto.last_modified_cycle = cycle.cycle_id

    events.append(EventEnvelope(
        event_type=EventType.PROTOCOL_PROPOSED,
        run_id=config.run_id,
        condition=config.condition.value,
        cycle_id=cycle.cycle_id,
        agent_id="axiom",
        payload=output.model_dump(),
    ))

    return events
