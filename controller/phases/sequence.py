"""The cycle sequence, as data.

The 14 phases used to be 15 hardcoded `await self._run_phase(...)` calls, so
reordering, skipping or adding a phase meant editing the orchestrator. The
sequence is now a list the orchestrator walks.

Each entry declares:

    name        stable id used in logs and config
    fn          the phase's execute()
    needs_ctx   whether it receives the per-agent AgentContext map
    when        optional predicate on CycleState — skip when it returns False
    builds_ctx  rebuild agent contexts immediately after this phase

Changing the loop is now a list edit. Note what this does NOT check: phases
communicate through CycleState, so an order that puts evaluation before the
artifact exists produces a skipped phase rather than a crash. `validate_sequence`
catches the known dependencies; it is not a general dataflow analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

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


@dataclass(frozen=True)
class Phase:
    name: str
    fn: Callable
    needs_ctx: bool = True
    when: Optional[Callable] = None
    builds_ctx: bool = False

    def should_run(self, cycle) -> bool:
        return self.when is None or self.when(cycle)


#: The v1 sequence (PLAN.md section 8). Identical in order and behaviour to the
#: hardcoded version it replaces.
DEFAULT_SEQUENCE: tuple[Phase, ...] = (
    # Contexts are built from world state, so this must precede anything that
    # needs them — hence builds_ctx here rather than before the loop.
    Phase("load_state", load_state.execute, needs_ctx=False, builds_ctx=True),
    Phase("reflection", reflection.execute),
    Phase("scenario_check", scenario_check.execute),
    Phase("scenario_inject", scenario_inject.execute,
          when=lambda c: c.scenario_active),
    Phase("discussion", discussion.execute),
    Phase("retrieval", retrieval.execute),
    Phase("protocol_design", protocol_design.execute),
    Phase("evaluation", evaluation.execute),
    Phase("interpretation", interpretation.execute),
    Phase("doctrine_revision", doctrine_revision.execute),
    Phase("identity_revision", identity_revision.execute),
    Phase("ethical_log", ethical_log.execute),
    Phase("memory_summarize", memory_summarize.execute),
    Phase("persist_state", persist_state.execute, needs_ctx=False),
)

ALL_PHASES: dict[str, Phase] = {p.name: p for p in DEFAULT_SEQUENCE}

#: Phases that must come after another, because they read what it writes
#: through CycleState. Not exhaustive — these are the ones whose violation is
#: silent rather than loud.
_ORDERING_RULES: tuple[tuple[str, str], ...] = (
    ("scenario_check", "scenario_inject"),   # sets cycle.scenario_active
    ("protocol_design", "evaluation"),       # sets cycle.proposed_protocol
    ("evaluation", "interpretation"),        # sets cycle.evaluation_result
    # Not a CycleState dependency but a destructive one: persist_state writes
    # every artifact from memory, so running it without load_state truncates
    # the journals and both logs on the first cycle.
    ("load_state", "persist_state"),
)


class SequenceError(ValueError):
    """A phase sequence that would silently misbehave."""


def build_sequence(names: list[str] | None) -> tuple[Phase, ...]:
    """Resolve phase names into a sequence, or return the default."""
    if not names:
        return DEFAULT_SEQUENCE
    unknown = [n for n in names if n not in ALL_PHASES]
    if unknown:
        raise SequenceError(
            f"Unknown phase(s): {unknown}. Available: {sorted(ALL_PHASES)}"
        )
    if len(set(names)) != len(names):
        raise SequenceError(f"Phase sequence contains duplicates: {names}")
    sequence = tuple(ALL_PHASES[n] for n in names)
    validate_sequence(sequence)
    return sequence


def validate_sequence(sequence: tuple[Phase, ...]) -> None:
    """Reject orders whose failure mode would be a silent no-op."""
    order = {p.name: i for i, p in enumerate(sequence)}

    for before, after in _ORDERING_RULES:
        if before in order and after in order and order[before] > order[after]:
            raise SequenceError(
                f"{after!r} runs before {before!r}, which produces the state it "
                f"reads. The phase would silently do nothing rather than fail."
            )
        if after in order and before not in order:
            raise SequenceError(
                f"{after!r} is present but {before!r} is not; it has nothing to act on."
            )

    if not any(p.builds_ctx for p in sequence) and any(p.needs_ctx for p in sequence):
        raise SequenceError(
            "Sequence has phases needing agent contexts but nothing that builds them "
            "(load_state is missing)."
        )

    if "persist_state" not in order:
        raise SequenceError(
            "persist_state is missing — the cycle's work would never reach disk."
        )
    if order["persist_state"] != len(sequence) - 1:
        raise SequenceError("persist_state must be last; anything after it is not saved.")
