"""The sandbox task — what the agents actually build each cycle.

GENESIS v1 hardcoded one task: write a protocol document. "Protocol" appeared
across 17 files, so changing what the agents make meant editing the design
phase, the evaluation phase, the evaluator prompt, the artifact model and the
world-state writer together.

A Task gathers all of that in one place:

    what to ask for      design_prompt() and output_schema
    how to store it      apply()
    how to judge it      evaluation_prompt() and dimensions

The cycle loop stays identical. Swapping the task changes what gets built and
how it is scored, without touching orchestration, logging, doctrine, identity,
memory or retrieval.

Tasks produce *artifacts*, not side effects. A task whose artifact is code does
not execute it — execution is a separate capability behind ExecutionSandbox,
which refuses by default (see docs/containment_design.md).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Type

from pydantic import BaseModel

if TYPE_CHECKING:
    from controller.config import RunConfig
    from controller.cycle import CycleState
    from controller.world.state import WorldState


class Task(ABC):
    """One thing the agents build and are scored on, each cycle."""

    #: Stable identifier, used in config and logs.
    name: str = "task"

    #: Human-readable noun for the artifact, e.g. "protocol document".
    artifact_noun: str = "document"

    #: Scoring dimensions. Each is scored 0-10 by a fresh-context evaluator.
    #: Changing these changes EvaluationScores, so a task owns its rubric.
    dimensions: tuple[str, ...] = ()

    @abstractmethod
    def design_prompt(self, config: RunConfig, world: WorldState, cycle: CycleState) -> str:
        """The instruction given to the lead agent in the design phase."""
        ...

    @abstractmethod
    def output_schema(self) -> Type[BaseModel]:
        """Pydantic schema the design phase parses the agent's output into."""
        ...

    @abstractmethod
    def apply(self, world: WorldState, output: BaseModel, cycle: CycleState,
              max_tokens: int | None = None) -> str:
        """Persist the produced artifact into world state.

        Returns the artifact id, which is logged and passed to evaluation.
        Implementations MUST sanitise any model-supplied identifier that
        becomes a filename — see controller/world/paths.py.
        """
        ...

    @abstractmethod
    def evaluation_prompt(
        self, world: WorldState, cycle: CycleState, doctrine_context: str
    ) -> str | None:
        """The evaluator's instruction, or None when there is nothing to score."""
        ...

    def scores_schema(self) -> Type[BaseModel]:
        """Build a scores model from `dimensions`.

        Generated rather than hand-written so a task cannot declare dimensions
        its rubric does not actually score.
        """
        from pydantic import Field, create_model

        fields = {d: (int, Field(..., ge=0, le=10)) for d in self.dimensions}
        return create_model(f"{self.name.title()}Scores", **fields)  # type: ignore[call-overload]

    def evaluation_schema(self) -> Type[BaseModel]:
        """Full evaluator output model for this task's rubric.

        Built from `dimensions` so a task cannot declare one rubric and be
        scored on another — which is exactly what happened when EvaluationOutput
        hardcoded the protocol dimensions.
        """
        from pydantic import Field, create_model

        return create_model(  # type: ignore[call-overload]
            f"{self.name.title()}EvaluationOutput",
            protocol_id=(str, Field(default="", description="Set by controller after parsing")),
            scores=(self.scores_schema(), ...),
            justifications=(dict[str, str], Field(
                ..., description="One-sentence justification per dimension")),
            total_score=(int, Field(..., ge=0, le=self.max_score)),
            assessment=(str, Field(..., description="Overall assessment paragraph")),
        )

    @staticmethod
    def enforce_length(content: str, max_tokens: int) -> tuple[str, bool]:
        """Cap an artifact's length. Returns (content, was_truncated).

        config.max_protocol_length_tokens was declared and never read, so an
        agent could write an arbitrarily long document and inflate every
        subsequent prompt for the rest of the run. Approximated at 4 chars per
        token rather than pulling in tiktoken for a threshold check.
        """
        limit = max_tokens * 4
        if len(content) <= limit:
            return content, False
        return content[:limit] + "\n\n[truncated at the configured length limit]", True

    @property
    def max_score(self) -> int:
        return 10 * len(self.dimensions)
