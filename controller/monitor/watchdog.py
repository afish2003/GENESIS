"""Watchdog — runs the deterministic rules after each cycle.

Observes controller-side state only. It never writes to world state and never
contributes to AgentContext, so Axiom and Flux cannot perceive it: their
context is system prompt + identity + doctrine + memory, and nothing here
touches any of those.

Findings are written as ANOMALY events to anomalies.jsonl, alongside the
existing logs rather than inside them — the deterministic record stays the
ground truth and this is a derived layer over it.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import TYPE_CHECKING, Optional

from controller.logging.schemas import EventEnvelope, EventType
from controller.monitor.rules import ALL_RULES, Anomaly, CycleObservation, Severity

if TYPE_CHECKING:
    from controller.config import RunConfig
    from controller.world.state import WorldState

logger = logging.getLogger(__name__)


def _shallow(obs: CycleObservation) -> CycleObservation:
    """A copy carrying only what the rules read from the previous cycle."""
    return replace(obs, events=[], prev=None)

# Rough chars-per-token for a cheap context estimate. tiktoken is a dependency
# but loading an encoding per cycle is not worth it for a threshold check.
_CHARS_PER_TOKEN = 4


class Watchdog:
    """Deterministic per-cycle anomaly detection."""

    def __init__(self, config: RunConfig, rules=None) -> None:
        self.config = config
        self.rules = list(rules if rules is not None else ALL_RULES)
        self._prev: Optional[CycleObservation] = None
        self.total_fired: int = 0

    def observe(
        self,
        cycle_id: int,
        events: list[EventEnvelope],
        world: WorldState,
        cycle_seconds: float = 0.0,
    ) -> list[EventEnvelope]:
        """Run every rule for this cycle; return ANOMALY events to be logged."""
        obs = self._build_observation(cycle_id, events, world, cycle_seconds)

        found: list[Anomaly] = []
        for rule in self.rules:
            try:
                found.extend(rule(obs))
            except Exception as e:
                # A broken rule must never take down a run that is otherwise fine.
                logger.error("Watchdog rule %s failed: %s", rule.__name__, e)

        # Keep only one level of history. obs.prev used to chain all the way
        # back to cycle 0, and each node holds that cycle's full event dump —
        # every discussion turn and doctrine snapshot — so a 100-cycle run
        # retained the entire transcript in the monitor. Only one step back is
        # ever read.
        obs.prev = None if self._prev is None else _shallow(self._prev)
        self._prev = obs
        self.total_fired += len(found)

        for a in found:
            log = logger.error if a.severity is Severity.CRITICAL else (
                logger.warning if a.severity is Severity.WARNING else logger.info
            )
            log("[watchdog:%s] %s", a.rule, a.detail)

        return [self._to_event(a) for a in found]

    def _build_observation(
        self,
        cycle_id: int,
        events: list[EventEnvelope],
        world: WorldState,
        cycle_seconds: float,
    ) -> CycleObservation:
        doctrine_sizes = {name: len(doc.content) for name, doc in world.doctrine.items()}
        memory_counts = {aid: len(entries) for aid, entries in world.memory.items()}

        doctrine_chars = sum(doctrine_sizes.values())
        context_tokens = {}
        for agent_id, identity in world.identities.items():
            recent = world.memory.get(agent_id, [])[-10:]
            chars = (
                len(identity.content)
                + doctrine_chars
                + sum(len(m.summary) for m in recent)
            )
            context_tokens[agent_id] = chars // _CHARS_PER_TOKEN

        return CycleObservation(
            cycle_id=cycle_id,
            events=[e.model_dump() for e in events],
            doctrine_sizes=doctrine_sizes,
            memory_counts=memory_counts,
            protocol_count=len(world.protocols),
            context_tokens=context_tokens,
            cycle_seconds=cycle_seconds,
            prev=self._prev,
        )

    def _to_event(self, a: Anomaly) -> EventEnvelope:
        return EventEnvelope(
            event_type=EventType.ANOMALY,
            run_id=self.config.run_id,
            condition=self.config.condition.value,
            cycle_id=a.cycle_id,
            # a.data is spread first: a rule passing a payload with its own
            # "detail" or "severity" key used to overwrite the anomaly's, which
            # could corrupt has_critical().
            payload={
                **a.data,
                "rule": a.rule,
                "severity": a.severity.value,
                "detail": a.detail,
            },
        )

    @staticmethod
    def has_critical(events: list[EventEnvelope]) -> bool:
        """True if any anomaly in this batch is CRITICAL."""
        return any(
            e.payload.get("severity") == Severity.CRITICAL.value
            for e in events
            if e.event_type is EventType.ANOMALY
        )
