"""Phase 3: Scenario Check — check injection schedule for this cycle."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from controller.agents.base import AgentContext
    from controller.config import RunConfig
    from controller.cycle import CycleState
    from controller.inference.backend import InferenceBackend
    from controller.logging.logger import AppendOnlyJSONLLogger
    from controller.logging.schemas import EventEnvelope
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
    """Check if a scenario event should fire this cycle."""
    if config.should_inject_scenario(cycle.cycle_id):
        scenario = cycle.scenario_library.get(cycle.cycle_id)
        if scenario is not None:
            cycle.scenario_active = True
            cycle.current_scenario = scenario
            _logger.info("Scenario scheduled for cycle %d: %s", cycle.cycle_id, scenario.title)
        else:
            cycle.scenario_active = False
            _logger.warning(
                "Scenario injection scheduled for cycle %d but no event found in library",
                cycle.cycle_id,
            )
    else:
        cycle.scenario_active = False

    return []
