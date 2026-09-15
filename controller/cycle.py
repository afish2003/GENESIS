"""Cycle orchestrator — executes the 14-phase cycle loop.

Each cycle runs all phases in strict order. Phase modules are
independently testable. Each phase receives the current WorldState
and AgentContexts and returns events to log.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
import time
from datetime import datetime, timezone
from typing import Optional

from controller.agents.base import AgentContext, build_agent_context
from controller.config import RunConfig
from controller.inference.backend import InferenceBackend
from controller.logging.logger import AppendOnlyJSONLLogger
from controller.logging.schemas import EventEnvelope, EventType
from controller.phases import (
    discussion,
    doctrine_revision,
    ethical_log,
    evaluation,
    identity_revision,
    interpretation,
    load_state,
    memory_summarize,
    persist_state,
    protocol_design,
    reflection,
    retrieval,
    scenario_check,
    scenario_inject,
)
from controller.retrieval.databases import KnowledgeBaseManager
from controller.world.artifacts import ScenarioEvent
from controller.monitor.watchdog import Watchdog
from controller.phases.sequence import build_sequence
from controller.world.reset import write_checkpoint
from controller.world.state import WorldState

logger = logging.getLogger(__name__)


@dataclass
class CycleState:
    """Mutable state passed through phases within a single cycle."""

    cycle_id: int
    scenario_active: bool = False
    current_scenario: Optional[ScenarioEvent] = None
    scenario_library: dict[int, ScenarioEvent] = field(default_factory=dict)
    kb_manager: Optional[KnowledgeBaseManager] = field(default=None)
    retrieval_results: dict[str, list] = field(default_factory=dict)  # agent_id -> results
    proposed_protocol: Optional[dict] = None
    evaluation_result: Optional[dict] = None
    events: list[EventEnvelope] = field(default_factory=list)


class CycleOrchestrator:
    """Runs the 14-phase cycle loop for a complete experimental run."""

    def __init__(
        self,
        config: RunConfig,
        backend: InferenceBackend,
        world: WorldState,
        log: AppendOnlyJSONLLogger,
        scenario_library: Optional[dict[int, ScenarioEvent]] = None,
        kb_manager: Optional[KnowledgeBaseManager] = None,
    ) -> None:
        self.config = config
        self.backend = backend
        self.world = world
        self.logger = log
        self.scenario_library = scenario_library or {}
        self.kb_manager = kb_manager
        # Deterministic monitoring. Observes controller-side state only and
        # never contributes to AgentContext, so the agents cannot perceive it.
        self.watchdog = Watchdog(config) if config.watchdog_enabled else None
        self.sequence = build_sequence(config.phase_sequence)
        # Everything logged during the current cycle, so the watchdog can
        # observe the cycle exactly as the logs record it.
        self._cycle_events: list[EventEnvelope] = []

    async def run_all_cycles(self, start_cycle: int = 0) -> None:
        """Execute all cycles from start_cycle to total_cycles."""
        # Log run start
        self._log_event(EventType.RUN_START, -1, payload={
            "total_cycles": self.config.total_cycles,
            "condition": self.config.condition.value,
            "model": self.config.model_name,
        })

        for cycle_id in range(start_cycle, self.config.total_cycles):
            await self.run_cycle(cycle_id)

            # Checkpoint after each cycle
            world_hash = self.world.compute_hash()
            write_checkpoint(
                self.config.run_log_dir,
                self.config.run_id,
                cycle_id,
                world_hash,
            )

            # Check for pause
            if self.config.pause_after_cycle is not None and cycle_id == self.config.pause_after_cycle:
                logger.info("Pausing after cycle %d as requested.", cycle_id)
                break

        # Log run end
        self._log_event(EventType.RUN_END, self.config.total_cycles - 1, payload={
            "completed_cycles": self.config.total_cycles,
        })

    async def run_cycle(self, cycle_id: int) -> None:
        """Execute a single cycle (all 14 phases)."""
        cycle_started = time.monotonic()
        self._cycle_events = []
        logger.info("=== Cycle %d ===", cycle_id)
        self._log_event(EventType.CYCLE_START, cycle_id)

        cycle = CycleState(
            cycle_id=cycle_id,
            scenario_library=self.scenario_library,
            kb_manager=self.kb_manager,
        )

        # Handle memory reset for MEM_RESET condition
        if self.config.should_reset_memory(cycle_id):
            logger.info("Memory reset at cycle %d", cycle_id)
            for agent_id in self.config.agents:
                self.world.reset_memory(
                    agent_id, cycle_id, self.config.memory_reset_bootstrap
                )
            # Self-history must go too. Wiping the memory journal while leaving
            # the full past retrievable would not remove memory, only change how
            # it is reached — which would confound the contrast against BASELINE
            # that PLAN.md section 11 exists to measure.
            removed = self.kb_manager.clear_self_history() if self.kb_manager else 0
            self._log_event(EventType.NOTABLE_EVENT, cycle_id, payload={
                "kind": "memory_reset",
                "agents": list(self.config.agents),
                "self_history_documents_cleared": removed,
            })

        # Walk the sequence. Order is data — see controller/phases/sequence.py.
        contexts: dict[str, AgentContext] = {}
        for phase in self.sequence:
            if not phase.should_run(cycle):
                continue
            await self._run_phase(
                phase.name, cycle_id, cycle, phase.fn,
                contexts if phase.needs_ctx else None,
            )
            if phase.builds_ctx:
                contexts = self._build_contexts()

        self._log_event(EventType.CYCLE_END, cycle_id)

        # Watchdog runs last, after state is persisted, so it observes the
        # cycle exactly as the logs record it.
        if self.watchdog is not None:
            elapsed = time.monotonic() - cycle_started
            anomalies = self.watchdog.observe(
                cycle_id=cycle_id,
                events=self._cycle_events,
                world=self.world,
                cycle_seconds=elapsed,
            )
            if anomalies:
                self.logger.log_events(anomalies)
            if self.config.halt_on_critical_anomaly and Watchdog.has_critical(anomalies):
                raise RuntimeError(
                    f"Cycle {cycle_id}: watchdog reported a CRITICAL anomaly and "
                    f"halt_on_critical_anomaly is set. See anomalies.jsonl."
                )

    async def _run_phase(
        self,
        phase_name: str,
        cycle_id: int,
        cycle: CycleState,
        phase_fn,
        contexts: Optional[dict[str, AgentContext]] = None,
    ) -> None:
        """Run a single phase with logging."""
        self._log_event(EventType.PHASE_START, cycle_id, payload={"phase": phase_name})

        try:
            if contexts is not None:
                events = await phase_fn(
                    config=self.config,
                    backend=self.backend,
                    world=self.world,
                    cycle=cycle,
                    contexts=contexts,
                    logger=self.logger,
                )
            else:
                events = await phase_fn(
                    config=self.config,
                    backend=self.backend,
                    world=self.world,
                    cycle=cycle,
                    logger=self.logger,
                )

            if events:
                self.logger.log_events(events)
                self._cycle_events.extend(events)

        except Exception as e:
            logger.error("Phase %s failed at cycle %d: %s", phase_name, cycle_id, e, exc_info=True)
            self._log_event(EventType.NOTABLE_EVENT, cycle_id, payload={
                "phase": phase_name,
                "error": str(e),
                "type": "PHASE_ERROR",
            })

        self._log_event(EventType.PHASE_END, cycle_id, payload={"phase": phase_name})

    def _build_contexts(self) -> dict[str, AgentContext]:
        """Build fresh agent contexts from current world state."""
        doctrine_texts = {
            name: doc.content for name, doc in self.world.doctrine.items()
        }
        contexts = {}
        for agent_id in self.config.agents:
            contexts[agent_id] = build_agent_context(
                agent_id=agent_id,
                prompts_dir=self.config.run_prompts_dir,
                identity=self.world.identities[agent_id],
                memory=self.world.memory[agent_id],
                doctrine_texts=doctrine_texts,
            )
        return contexts

    def _log_event(
        self,
        event_type: EventType,
        cycle_id: int,
        agent_id: Optional[str] = None,
        payload: Optional[dict] = None,
    ) -> None:
        """Helper to log a single event."""
        event = EventEnvelope(
            event_type=event_type,
            run_id=self.config.run_id,
            condition=self.config.condition.value,
            cycle_id=cycle_id,
            agent_id=agent_id,
            payload=payload or {},
        )
        self.logger.log_event(event)
        self._cycle_events.append(event)
