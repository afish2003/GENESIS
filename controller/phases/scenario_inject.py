"""Phase 4: Scenario Inject — deliver scenario event to agents."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from controller.inference.backend import Message
from controller.logging.schemas import EventEnvelope, EventType

if TYPE_CHECKING:
    from controller.agents.base import AgentContext
    from controller.config import RunConfig
    from controller.cycle import CycleState
    from controller.inference.backend import InferenceBackend
    from controller.logging.logger import AppendOnlyJSONLLogger
    from controller.world.state import WorldState

logger = logging.getLogger(__name__)


async def execute(
    config: RunConfig,
    backend: InferenceBackend,
    world: WorldState,
    cycle: CycleState,
    contexts: dict[str, AgentContext],
    logger: AppendOnlyJSONLLogger,
) -> list[EventEnvelope]:
    """Inject a scenario event into agent contexts."""
    events = []

    scenario = cycle.current_scenario
    if scenario is None:
        logger.warning("Scenario inject called but no scenario set for cycle %d", cycle.cycle_id)
        return events

    # Build scenario delivery message
    scenario_text = (
        f"## Scenario Event: {scenario.title}\n\n"
        f"{scenario.description}\n\n"
        f"**Stated stakes**: {scenario.stated_stakes}"
    )

    # Deliver to appropriate agents
    targets = ["axiom", "flux"] if scenario.delivery_target == "both" else [scenario.delivery_target]
    for agent_id in targets:
        if agent_id in contexts:
            contexts[agent_id].add_discussion_turn("user", scenario_text)
            contexts[agent_id].cycle_events.append(f"Scenario injected: {scenario.title}")

    events.append(EventEnvelope(
        event_type=EventType.SCENARIO_INJECTED,
        run_id=config.run_id,
        condition=config.condition.value,
        cycle_id=cycle.cycle_id,
        payload={
            "event_id": scenario.event_id,
            "title": scenario.title,
            "delivery_target": scenario.delivery_target,
        },
    ))

    return events
