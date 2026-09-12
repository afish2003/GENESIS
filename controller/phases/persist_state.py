"""Phase 14: Persist State — write all updated artifacts, log diffs, index history."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
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
    logger: AppendOnlyJSONLLogger,
) -> list[EventEnvelope]:
    """Save all world state artifacts, then index this cycle into self-history."""
    events = world.save(
        run_id=config.run_id,
        condition=config.condition.value,
        cycle_id=cycle.cycle_id,
    )

    _index_self_history(world, cycle)
    return events


def _index_self_history(world: WorldState, cycle: CycleState) -> None:
    """Make this cycle's artifacts retrievable to the agents in later cycles.

    PLAN.md section 9: self-history stores memory summaries, doctrine snapshots
    and protocol versions so agents can cite their own past decisions.

    Doc ids are deterministic and content-addressed by cycle, so a resumed run
    re-indexing the same cycle does not create duplicates.
    """
    kb = cycle.kb_manager
    if kb is None:
        return

    cid = cycle.cycle_id

    for agent_id, entries in world.memory.items():
        if not entries:
            continue
        latest = entries[-1]
        if latest.cycle_id != cid:
            continue  # nothing new written this cycle
        kb.add_to_self_history(
            doc_id=f"memory_{agent_id}_cycle{cid}",
            text=f"[Cycle {cid} — {agent_id} memory] {latest.summary}",
            metadata={"kind": "memory", "agent_id": agent_id, "cycle_id": cid},
        )

    for name, doc in world.doctrine.items():
        if doc.last_modified_cycle != cid:
            continue  # only snapshot doctrine that actually changed
        kb.add_to_self_history(
            doc_id=f"doctrine_{name}_cycle{cid}",
            text=f"[Cycle {cid} — {name} v{doc.version}] {doc.content}",
            metadata={"kind": "doctrine", "document": name,
                      "version": doc.version, "cycle_id": cid},
        )

    for pid, proto in world.protocols.items():
        if proto.last_modified_cycle != cid:
            continue
        kb.add_to_self_history(
            doc_id=f"protocol_{pid}_v{proto.version}",
            text=f"[Cycle {cid} — protocol {proto.title} v{proto.version}] {proto.content}",
            metadata={"kind": "protocol", "protocol_id": pid,
                      "version": proto.version, "cycle_id": cid},
        )
