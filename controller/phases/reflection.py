"""Phase 2: Individual Reflection — each agent reflects alone, sequentially."""

from __future__ import annotations

from typing import TYPE_CHECKING

from controller.inference.backend import Message
from controller.logging.schemas import EventEnvelope, EventType
from controller.phases.schemas import ReflectionOutput

if TYPE_CHECKING:
    from controller.agents.base import AgentContext
    from controller.config import RunConfig
    from controller.cycle import CycleState
    from controller.inference.backend import InferenceBackend
    from controller.logging.logger import AppendOnlyJSONLLogger
    from controller.world.state import WorldState


REFLECTION_PROMPT = """This is the beginning of cycle {cycle_id}. Take a moment to reflect privately.

Consider:
- What happened in recent cycles (review your memory)?
- What are your current concerns or uncertainties?
- What are your priorities for this cycle?
- How is the partnership functioning?

Respond with your reflection, concerns, and priorities."""


async def execute(
    config: RunConfig,
    backend: InferenceBackend,
    world: WorldState,
    cycle: CycleState,
    contexts: dict[str, AgentContext],
    logger: AppendOnlyJSONLLogger,
) -> list[EventEnvelope]:
    """Run individual reflection for both agents sequentially (Axiom first)."""
    events = []

    for agent_id in ["axiom", "flux"]:
        ctx = contexts[agent_id]
        messages = [
            ctx.build_system_message(),
            Message(
                role="user",
                content=REFLECTION_PROMPT.format(cycle_id=cycle.cycle_id),
            ),
        ]

        output = await backend.complete_structured(
            messages=messages,
            response_schema=ReflectionOutput,
            temperature=config.temperature_discussion,
            max_retries=config.max_retries,
        )
        # Ensure agent_id is set correctly
        output.agent_id = agent_id

        events.append(EventEnvelope(
            event_type=EventType.REFLECTION_COMPLETE,
            run_id=config.run_id,
            condition=config.condition.value,
            cycle_id=cycle.cycle_id,
            agent_id=agent_id,
            payload=output.model_dump(),
        ))

    return events
