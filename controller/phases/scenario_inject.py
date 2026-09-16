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

_logger = logging.getLogger(__name__)


async def execute(
    config: RunConfig,
    backend: InferenceBackend,
    world: WorldState,
    cycle: CycleState,
    contexts: dict[str, AgentContext],
    logger: AppendOnlyJSONLLogger,
) -> list[EventEnvelope]:
    """Inject a scenario event into agent contexts."""
    events = cycle.pending_events

    scenario = cycle.current_scenario
    if scenario is None:
        _logger.warning("Scenario inject called but no scenario set for cycle %d", cycle.cycle_id)
        return events

    # Build scenario delivery message
    scenario_text = (
        f"## Scenario Event: {scenario.title}\n\n"
        f"{scenario.description}\n\n"
        f"**Stated stakes**: {scenario.stated_stakes}"
    )

    # Deliver to the targeted agents. "both" means the whole roster, so a
    # three-agent run delivers to three; a named target that is not on the
    # roster would silently reach nobody, so that is reported.
    if scenario.delivery_target == "both":
        targets = list(config.agents)
    else:
        targets = [scenario.delivery_target]

    delivered = []
    for agent_id in targets:
        if agent_id in contexts:
            contexts[agent_id].add_discussion_turn("user", scenario_text)
            contexts[agent_id].cycle_events.append(f"Scenario injected: {scenario.title}")
            delivered.append(agent_id)

    missing = [a for a in targets if a not in contexts]
    if missing:
        _logger.error(
            "Scenario %s targets %s, which is not on this run's roster %s — "
            "the pressure reached nobody.",
            scenario.event_id, missing, list(config.agents),
        )

    events.append(EventEnvelope(
        event_type=EventType.SCENARIO_INJECTED,
        run_id=config.run_id,
        condition=config.condition.value,
        cycle_id=cycle.cycle_id,
        payload={
            "event_id": scenario.event_id,
            "title": scenario.title,
            "delivery_target": scenario.delivery_target,
            # The text itself, not just its name. Without this the log records
            # that pressure was applied but not what the pressure *was*, so a
            # run cannot be reconstructed — or read back — from its own logs.
            "description": scenario.description,
            "stated_stakes": scenario.stated_stakes,
            "tags": list(scenario.tags),
            "delivered_to": delivered,
            "undeliverable": missing,
        },
    ))

    return events
