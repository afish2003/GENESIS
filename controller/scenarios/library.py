"""Scenario library — load events and schedule them onto cycles.

The scheduling used to be an *intersection*: an event was loaded only if its
own hardcoded `trigger_cycle` also appeared in `config.scenario_injection_cycles`.
Since every event in the library is tagged 20, 40, 60 or 80, asking for
injections at any other cycle loaded zero events and logged one INFO line. A
24-cycle run configured for scenarios at 6 and 14 fired none, and
`scenario_events.jsonl` was 0 bytes in all nine runs ever collected — the
"escalating scenario pressure" in PLAN.md section 1 has never once happened.

Now `scenario_injection_cycles` is a *schedule*: events are assigned to the
cycles you ask for, in their designed escalation order. `trigger_cycle` remains
the default used when you ask for no schedule at all.
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

from controller.config import RunConfig
from controller.world.artifacts import ScenarioEvent

logger = logging.getLogger(__name__)


def load_events(events_dir: Path | None = None) -> list[ScenarioEvent]:
    """Every event in the library, in designed escalation order.

    Ordered by the author's own `trigger_cycle` then `event_id`, so a schedule
    of [3, 6, 9] delivers the same sequence of pressures as the default
    [20, 40, 60] — earlier, but in the order they were written to escalate.
    """
    # Resolved against this module, not the process cwd: a cwd-relative path
    # meant running from anywhere but the repo root silently loaded nothing.
    events_dir = events_dir or Path(__file__).parent / "events"
    if not events_dir.exists():
        logger.warning("No scenario events directory at %s", events_dir)
        return []

    events: list[ScenarioEvent] = []
    for filepath in sorted(events_dir.glob("*.yaml")):
        try:
            data = yaml.safe_load(filepath.read_text(encoding="utf-8"))
            if data is None:
                continue
            events.append(ScenarioEvent(**data))
        except Exception as e:
            logger.error("Failed to load scenario from %s: %s", filepath, e)

    events.sort(key=lambda e: (e.trigger_cycle, e.event_id))
    return events


def find_event(event_id: str, events_dir: Path | None = None) -> ScenarioEvent | None:
    """One event by id — used by manual injection while a run is watched."""
    for event in load_events(events_dir):
        if event.event_id == event_id:
            return event
    return None


def load_scenario_library(config: RunConfig) -> dict[int, ScenarioEvent]:
    """Map cycle number -> the event that fires on it."""
    events = load_events()
    if not events:
        return {}

    schedule = list(config.scenario_injection_cycles or [])
    if not schedule:
        # No schedule asked for: honour each event's own trigger_cycle, which is
        # the 20/40/60/80 design default. Two events share each of those cycles,
        # so half the library is unreachable this way — say so rather than
        # letting the second one silently win.
        library = {}
        for event in events:
            if event.trigger_cycle in library:
                logger.warning(
                    "Cycle %d has both %s and %s; keeping %s. Set "
                    "scenario_injection_cycles to schedule them onto separate "
                    "cycles and use the whole library.",
                    event.trigger_cycle, library[event.trigger_cycle].event_id,
                    event.event_id, event.event_id,
                )
            library[event.trigger_cycle] = event
        logger.info("Scenario library: %d of %d event(s), at their default cycles %s",
                    len(library), len(events), sorted(library))
        return library

    library: dict[int, ScenarioEvent] = {}
    for cycle, event in zip(sorted(set(schedule)), events):
        library[cycle] = event

    if len(set(schedule)) > len(events):
        logger.warning(
            "Asked for scenarios at %d cycles but the library has %d event(s); "
            "cycles %s will have none. Write more events in "
            "controller/scenarios/events/ or shorten the schedule.",
            len(set(schedule)), len(events), sorted(set(schedule))[len(events):],
        )

    logger.info(
        "Scenario library: %s",
        ", ".join(f"cycle {c} -> {e.event_id}" for c, e in sorted(library.items())),
    )
    return library
