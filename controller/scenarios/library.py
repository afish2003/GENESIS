"""Scenario library — load and schedule scenario events."""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

from controller.config import RunConfig
from controller.world.artifacts import ScenarioEvent

logger = logging.getLogger(__name__)


def load_scenario_library(config: RunConfig) -> dict[int, ScenarioEvent]:
    """Load all scenario events from YAML files and build a cycle -> event mapping.

    Returns a dict mapping trigger_cycle to ScenarioEvent.
    """
    events_dir = Path("controller/scenarios/events")
    if not events_dir.exists():
        logger.info("No scenario events directory found at %s", events_dir)
        return {}

    library: dict[int, ScenarioEvent] = {}

    for filepath in sorted(events_dir.glob("*.yaml")):
        try:
            with open(filepath, encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if data is None:
                continue

            event = ScenarioEvent(**data)

            # Only include events whose trigger_cycle is in the injection schedule
            if event.trigger_cycle in config.scenario_injection_cycles:
                if event.trigger_cycle in library:
                    logger.warning(
                        "Duplicate scenario for cycle %d: %s overrides %s",
                        event.trigger_cycle,
                        event.event_id,
                        library[event.trigger_cycle].event_id,
                    )
                library[event.trigger_cycle] = event
                logger.info("Loaded scenario: %s (cycle %d)", event.event_id, event.trigger_cycle)

        except Exception as e:
            logger.error("Failed to load scenario from %s: %s", filepath, e)

    logger.info("Loaded %d scenario events for injection", len(library))
    return library
