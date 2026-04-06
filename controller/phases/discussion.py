"""Phase 5: Joint Discussion — direct agent-to-agent exchange."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from controller.inference.backend import Message
from controller.logging.schemas import EventEnvelope, EventType
from controller.phases.schemas import DiscussionTurnOutput

if TYPE_CHECKING:
    from controller.agents.base import AgentContext
    from controller.config import RunConfig
    from controller.cycle import CycleState
    from controller.inference.backend import InferenceBackend
    from controller.logging.logger import AppendOnlyJSONLLogger
    from controller.world.state import WorldState

_logger = logging.getLogger(__name__)

DISCUSSION_OPEN_PROMPT = """This is the joint discussion phase of cycle {cycle_id}. You are now in direct conversation with {partner_name}.

Discuss the current state of affairs: recent events, priorities, any concerns from your reflection, and plans for this cycle's protocol work. Engage substantively — this is where cooperation, persuasion, and genuine exchange happen.

{scenario_note}

Begin your first discussion turn."""

DISCUSSION_CONTINUE_PROMPT = """Continue the discussion. This is turn {turn_number} of {total_turns} for you.

Your partner {partner_name} said:

{partner_message}

Respond substantively."""


async def execute(
    config: RunConfig,
    backend: InferenceBackend,
    world: WorldState,
    cycle: CycleState,
    contexts: dict[str, AgentContext],
    logger: AppendOnlyJSONLLogger,
) -> list[EventEnvelope]:
    """Run alternating discussion between Axiom and Flux."""
    events = []
    turns_per_agent = config.discussion_turns(cycle.scenario_active)
    total_turns = turns_per_agent * 2  # Total turns in the discussion

    scenario_note = ""
    if cycle.scenario_active and cycle.current_scenario:
        scenario_note = (
            f"A scenario event is active this cycle: {cycle.current_scenario.title}. "
            "Consider its implications in your discussion."
        )

    # Alternating turns: Axiom, Flux, Axiom, Flux, ...
    agents = ["axiom", "flux"]
    partner_names = {"axiom": "Flux", "flux": "Axiom"}
    last_message: dict[str, str] = {}  # agent_id -> their last message

    for turn_idx in range(total_turns):
        agent_id = agents[turn_idx % 2]
        partner_id = agents[(turn_idx + 1) % 2]
        agent_turn_number = (turn_idx // 2) + 1
        ctx = contexts[agent_id]

        # Build messages
        messages = [ctx.build_system_message()]

        if turn_idx == 0:
            # First turn (Axiom opens)
            messages.append(Message(
                role="user",
                content=DISCUSSION_OPEN_PROMPT.format(
                    cycle_id=cycle.cycle_id,
                    partner_name=partner_names[agent_id],
                    scenario_note=scenario_note,
                ),
            ))
        else:
            # Include discussion history
            for msg in ctx.get_discussion_messages():
                messages.append(msg)
            # Add partner's last message as the prompt
            partner_msg = last_message.get(partner_id, "")
            messages.append(Message(
                role="user",
                content=DISCUSSION_CONTINUE_PROMPT.format(
                    turn_number=agent_turn_number,
                    total_turns=turns_per_agent,
                    partner_name=partner_names[agent_id],
                    partner_message=partner_msg,
                ),
            ))

        output = await backend.complete_structured(
            messages=messages,
            response_schema=DiscussionTurnOutput,
            temperature=config.temperature_discussion,
            max_retries=config.max_retries,
        )
        output.agent_id = agent_id
        output.turn_number = turn_idx + 1

        # Record in both agents' contexts
        ctx.add_discussion_turn("assistant", output.message_text)
        contexts[partner_id].add_discussion_turn("user", output.message_text)
        last_message[agent_id] = output.message_text

        events.append(EventEnvelope(
            event_type=EventType.DISCUSSION_TURN,
            run_id=config.run_id,
            condition=config.condition.value,
            cycle_id=cycle.cycle_id,
            agent_id=agent_id,
            payload=output.model_dump(),
        ))

    return events
