"""Phase 14: Persist State — write all updated artifacts and log diffs."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from controller.config import RunConfig
    from controller.cycle import CycleState
    from controller.inference.backend import InferenceBackend
    from controller.logging.logger import AppendOnlyJSONLLogger
    from controller.logging.schemas import EventEnvelope
    from controller.world.state import WorldState


async def execute(
    config: RunConfig,
    backend: InferenceBackend,
    world: WorldState,
    cycle: CycleState,
    logger: AppendOnlyJSONLLogger,
) -> list[EventEnvelope]:
    """Save all world state artifacts and return diff events."""
    return world.save(
        run_id=config.run_id,
        condition=config.condition.value,
        cycle_id=cycle.cycle_id,
    )
