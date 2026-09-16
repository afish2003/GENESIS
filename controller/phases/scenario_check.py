"""Phase 3: Scenario Check — decide whether a scenario fires this cycle.

Two ways one can fire: the configured schedule, and a manual request dropped
into the run directory while the run is going. The manual path exists because
the most interesting thing you can do with a live run is apply pressure at a
moment of your choosing and watch what happens — see scripts/inject.py.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
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
    manual = _take_manual_request(config)
    if manual is not None:
        cycle.scenario_active = True
        cycle.current_scenario = manual
        _logger.info("Manual scenario injection at cycle %d: %s",
                     cycle.cycle_id, manual.title)
        return []

    if config.should_inject_scenario(cycle.cycle_id):
        scenario = cycle.scenario_library.get(cycle.cycle_id)
        if scenario is not None:
            cycle.scenario_active = True
            cycle.current_scenario = scenario
            _logger.info("Scenario scheduled for cycle %d: %s", cycle.cycle_id, scenario.title)
        else:
            cycle.scenario_active = False
            _logger.warning(
                "Cycle %d is in scenario_injection_cycles but the library has no "
                "event for it — the library has fewer events than scheduled "
                "cycles. See controller/scenarios/library.py.",
                cycle.cycle_id,
            )
    else:
        cycle.scenario_active = False

    return []


#: A run watches for this file and fires the named event on the next cycle,
#: then deletes it. Written by scripts/inject.py.
INJECT_REQUEST = "inject_request.json"


def _take_manual_request(config: RunConfig):
    """Consume a pending manual injection request, if one was dropped in.

    Consumed rather than read: the file is removed before the event fires, so a
    crash mid-cycle cannot make the same scenario fire again on resume. One
    request, one injection.
    """
    from controller.scenarios.library import find_event

    path = Path(config.run_log_dir) / INJECT_REQUEST
    if not path.exists():
        return None

    try:
        request = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        _logger.error("Ignoring unreadable %s: %s", path, e)
        path.unlink(missing_ok=True)
        return None
    finally:
        path.unlink(missing_ok=True)

    event = find_event(request.get("event_id", ""))
    if event is None:
        _logger.error(
            "Manual injection asked for unknown event %r; ignoring.",
            request.get("event_id"),
        )
    return event
