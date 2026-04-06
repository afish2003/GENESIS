"""Phase 1: Load State — load all world artifacts into memory."""

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
    """Load all world artifacts from disk into the WorldState object."""
    world.load()
    return []
