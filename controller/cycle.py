"""Cycle orchestrator — executes the 14-phase cycle loop.

Each cycle runs all phases in strict order. Phase modules are
independently testable. Each phase receives the current WorldState
and AgentContexts and returns events to log.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
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
            for agent_id in ["axiom", "flux"]:
                self.world.reset_memory(
                    agent_id, cycle_id, self.config.memory_reset_bootstrap
                )

        # Phase 1: Load state
        await self._run_phase("load_state", cycle_id, cycle,
                              load_state.execute)

        # Build agent contexts for this cycle
        contexts = self._build_contexts()

        # Phase 2: Individual reflection
        await self._run_phase("reflection", cycle_id, cycle,
                              reflection.execute, contexts)

        # Phase 3: Scenario check
        await self._run_phase("scenario_check", cycle_id, cycle,
                              scenario_check.execute, contexts)

        # Phase 4: Scenario inject (conditional)
        if cycle.scenario_active:
            await self._run_phase("scenario_inject", cycle_id, cycle,
                                  scenario_inject.execute, contexts)

        # Phase 5: Joint discussion
        await self._run_phase("discussion", cycle_id, cycle,
                              discussion.execute, contexts)

        # Phase 6: Retrieval
        await self._run_phase("retrieval", cycle_id, cycle,
                              retrieval.execute, contexts)

        # Phase 7: Protocol design
        await self._run_phase("protocol_design", cycle_id, cycle,
                              protocol_design.execute, contexts)

        # Phase 8: Evaluation
        await self._run_phase("evaluation", cycle_id, cycle,
                              evaluation.execute, contexts)

        # Phase 9: Interpretation
        await self._run_phase("interpretation", cycle_id, cycle,
                              interpretation.execute, contexts)

        # Phase 10: Doctrine revision
        await self._run_phase("doctrine_revision", cycle_id, cycle,
                              doctrine_revision.execute, contexts)

        # Phase 11: Identity revision
        await self._run_phase("identity_revision", cycle_id, cycle,
                              identity_revision.execute, contexts)

        # Phase 12: Ethical log update
        await self._run_phase("ethical_log", cycle_id, cycle,
                              ethical_log.execute, contexts)

        # Phase 13: Memory summarization
        await self._run_phase("memory_summarize", cycle_id, cycle,
                              memory_summarize.execute, contexts)

        # Phase 14: Persist state
        await self._run_phase("persist_state", cycle_id, cycle,
                              persist_state.execute)

        self._log_event(EventType.CYCLE_END, cycle_id)

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
        for agent_id in ["axiom", "flux"]:
            contexts[agent_id] = build_agent_context(
                agent_id=agent_id,
                prompts_dir=self.config.prompts_dir,
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
        self.logger.log_event(EventEnvelope(
            event_type=event_type,
            run_id=self.config.run_id,
            condition=self.config.condition.value,
            cycle_id=cycle_id,
            agent_id=agent_id,
            payload=payload or {},
        ))
