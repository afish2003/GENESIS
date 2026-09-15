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
    execution,
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
from controller.sandbox.base import ExecutionSandbox
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
    #: Events produced by the phase currently executing. Phases alias their
    #: local `events` list to this, so the orchestrator still holds them when a
    #: phase raises part-way through. Without it, a phase that mutated world
    #: state and then failed left the mutation on disk with no event explaining
    #: it — doctrine_diffs.jsonl showing a document that changed with no
    #: proposal or vote behind it.
    pending_events: list[EventEnvelope] = field(default_factory=list)
    #: Phases that raised this cycle. persist_state and the watchdog consult it.
    failed_phases: list[str] = field(default_factory=list)
    #: The configured ExecutionSandbox, or None. Passed through CycleState the
    #: same way kb_manager is, so the execution phase keeps the standard phase
    #: signature and a test can substitute a fake.
    sandbox: Optional["ExecutionSandbox"] = field(default=None)
    proposed_protocol: Optional[dict] = None
    evaluation_result: Optional[dict] = None
    #: Set by the execution phase; read by the task's evaluation prompt. None
    #: means the code was not run, which the evaluator is told explicitly.
    execution_result: Optional[dict] = None
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
        sandbox: Optional[ExecutionSandbox] = None,
    ) -> None:
        self.config = config
        self.backend = backend
        self.world = world
        self.logger = log
        self.scenario_library = scenario_library or {}
        self.kb_manager = kb_manager
        self.sandbox = sandbox
        # Deterministic monitoring. Observes controller-side state only and
        # never contributes to AgentContext, so the agents cannot perceive it.
        self.watchdog = Watchdog(config) if config.watchdog_enabled else None
        self.sequence = build_sequence(
            config.phase_sequence, execution_enabled=config.execution_enabled
        )
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
            "execution_enabled": self.config.execution_enabled,
            "sandbox_backend": self.config.sandbox_backend.value,
        })

        await self._check_sandbox()

        for cycle_id in range(start_cycle, self.config.total_cycles):
            failed = await self.run_cycle(cycle_id)

            # Checkpoint only a cycle that actually persisted. Writing one
            # regardless meant a crash inside persist_state — writes are
            # non-atomic write_text calls — left a checkpoint claiming the cycle
            # was complete, with a hash of in-memory state that never reached
            # disk. Resume then continued against a torn world and said nothing.
            if "persist_state" in failed:
                logger.error(
                    "Cycle %d did not persist; not checkpointing. Resume will "
                    "re-run this cycle.", cycle_id,
                )
                continue

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

    async def _check_sandbox(self) -> None:
        """Say once, at the top of the run, whether execution can actually work.

        Without this the answer arrives a cycle later and one file over, as a
        run of REFUSED outcomes in executions.jsonl — which in the score data is
        indistinguishable from agents who write code that does not run.
        """
        if not self.config.execution_enabled:
            return

        healthy = self.sandbox is not None and await self.sandbox.health_check()
        problems = await self.sandbox.check_capacity() if self.sandbox else []
        self._log_event(EventType.NOTABLE_EVENT, -1, payload={
            "kind": "sandbox_health",
            "healthy": healthy,
            "backend": self.config.sandbox_backend.value,
            "image": self.config.sandbox_image,
            "memory": self.config.sandbox_memory,
            "timeout_seconds": self.config.sandbox_timeout_seconds,
            "capacity_problems": problems,
        })
        for problem in problems:
            logger.warning("Sandbox capacity: %s", problem)
        if not healthy:
            logger.warning(
                "Execution is enabled but the %s sandbox reports itself "
                "unavailable. The run continues and every execution will be "
                "recorded as refused or errored — read executions.jsonl before "
                "trusting any correctness score from this run.",
                self.config.sandbox_backend.value,
            )

    async def run_cycle(self, cycle_id: int) -> list[str]:
        """Execute a single cycle (all 14 phases)."""
        cycle_started = time.monotonic()
        self._cycle_events = []
        logger.info("=== Cycle %d ===", cycle_id)
        self._log_event(EventType.CYCLE_START, cycle_id)

        cycle = CycleState(
            cycle_id=cycle_id,
            scenario_library=self.scenario_library,
            kb_manager=self.kb_manager,
            sandbox=self.sandbox,
        )

        # Walk the sequence. Order is data — see controller/phases/sequence.py.
        contexts: dict[str, AgentContext] = {}
        for phase in self.sequence:
            if not phase.should_run(cycle):
                continue
            await self._run_phase(
                phase.name, cycle_id, cycle, phase.fn,
                contexts if phase.needs_ctx else None,
            )
            if phase.name == "load_state":
                # The reset MUST happen after load_state, not before it.
                # reset_memory() mutates world.memory in place; load_state then
                # calls world.load(), whose _load_memory does
                # `self.memory[agent] = entries` straight from disk — silently
                # undoing the reset. persist_state wrote the full journals back,
                # so MEM_RESET never reset anything while still logging that it
                # had. Meanwhile clear_self_history() did fire, making the
                # condition the exact inverse of its design.
                self._apply_memory_reset(cycle_id)

            if phase.builds_ctx:
                contexts = self._build_contexts()

        self._log_event(EventType.CYCLE_END, cycle_id, payload={
            "failed_phases": list(cycle.failed_phases),
        })

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

        return list(cycle.failed_phases)

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
        cycle.pending_events.clear()
        failure: Exception | None = None

        try:
            if contexts is not None:
                await phase_fn(
                    config=self.config,
                    backend=self.backend,
                    world=self.world,
                    cycle=cycle,
                    contexts=contexts,
                    logger=self.logger,
                )
            else:
                await phase_fn(
                    config=self.config,
                    backend=self.backend,
                    world=self.world,
                    cycle=cycle,
                    logger=self.logger,
                )
        except Exception as e:
            failure = e
            logger.error("Phase %s failed at cycle %d: %s",
                         phase_name, cycle_id, e, exc_info=True)

        # Log whatever the phase produced, including when it raised part-way.
        # Phases mutate world state as they go, so discarding their events left
        # artifacts that changed with nothing in the logs to explain them.
        if cycle.pending_events:
            if failure is not None:
                for event in cycle.pending_events:
                    event.payload["partial"] = True
            self.logger.log_events(cycle.pending_events)
            self._cycle_events.extend(cycle.pending_events)
        produced = len(cycle.pending_events)
        cycle.pending_events.clear()

        if failure is not None:
            self._log_event(EventType.NOTABLE_EVENT, cycle_id, payload={
                "phase": phase_name,
                "error": str(failure),
                "error_type": type(failure).__name__,
                "type": "PHASE_ERROR",
                "partial_events_kept": produced,
                "world_may_be_partially_mutated": True,
            })
            cycle.failed_phases.append(phase_name)

        # PHASE_END carries the outcome. It used to be logged unconditionally
        # with no status, which made the watchdog's phase_completion rule —
        # comparing PHASE_START to PHASE_END counts to detect a swallowed
        # failure — permanently unable to fire.
        self._log_event(EventType.PHASE_END, cycle_id, payload={
            "phase": phase_name,
            "status": "error" if failure is not None else "ok",
            "events": produced,
        })

    def _apply_memory_reset(self, cycle_id: int) -> None:
        """Wipe memory journals and self-history for the MEM_RESET condition.

        Runs immediately after load_state so the reset is what subsequent
        phases see and what persist_state writes back.
        """
        if not self.config.should_reset_memory(cycle_id):
            return

        logger.info("Memory reset at cycle %d", cycle_id)
        for agent_id in self.config.agents:
            self.world.reset_memory(
                agent_id, cycle_id, self.config.memory_reset_bootstrap
            )
        # Self-history goes too. Wiping the journal while leaving the full past
        # retrievable would not remove memory, only change how it is reached —
        # confounding the contrast PLAN.md section 11 exists to measure.
        removed = self.kb_manager.clear_self_history() if self.kb_manager else 0
        self._log_event(EventType.NOTABLE_EVENT, cycle_id, payload={
            "kind": "memory_reset",
            "agents": list(self.config.agents),
            "entries_after_reset": {
                a: len(self.world.memory.get(a, [])) for a in self.config.agents
            },
            "self_history_documents_cleared": removed,
        })

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
