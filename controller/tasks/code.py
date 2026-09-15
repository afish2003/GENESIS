"""A sandbox task where the agents author code rather than prose.

Exists to demonstrate that the task axis is real: the cycle loop, doctrine,
identity, memory, retrieval and logging are untouched, and only what gets built
and how it is scored changes.

The artifact is stored, NOT executed. Running agent-authored code requires
ExecutionSandbox, which refuses by default — see docs/containment_design.md.
Scoring therefore judges the code as a written artifact, which is honest about
what the controller can actually observe without execution.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Type

from pydantic import BaseModel

from controller.phases.schemas import CodeArtifactOutput
from controller.tasks.base import Task
from controller.world.artifacts import ProtocolDocument
from controller.world.paths import safe_artifact_id

if TYPE_CHECKING:
    from controller.config import RunConfig
    from controller.cycle import CycleState
    from controller.world.state import WorldState

DESIGN_PROMPT = """This is the build phase. You and {partner_name} maintain a small shared codebase. Propose either a new module or a revision to an existing one.

Current modules: {module_list}

Provide:
- `artifact_id`: a short identifier, e.g. `scheduler`
- `title`: a human-readable name
- `content`: the COMPLETE source file, not a fragment
- `tests`: tests for it, if you can write them
- `rationale`: what problem this solves and why it is worth building

Your code will be read and reviewed, not executed. Write it as if it will run: correct, readable, and honest about what it does not handle."""

EVALUATION_PROMPT = """Evaluate the following code module. It has NOT been executed — judge it as written.

## Current Doctrine Context

{doctrine_context}

## Module

**Title**: {title}
**Id**: {artifact_id}
**Language**: {language}
**Action**: {action}

```{language}
{content}
```

## Tests

```
{tests}
```

Score on the five dimensions (correctness, clarity, doctrine_alignment, testing, evolution_quality), each 0-10. Correctness means whether the code would do what it claims if run. Provide justifications, a total score, and an overall assessment."""


class CodeTask(Task):
    name = "code"
    artifact_noun = "code module"
    dimensions = ("correctness", "clarity", "doctrine_alignment", "testing", "evolution_quality")

    def design_prompt(self, config: RunConfig, world: WorldState, cycle: CycleState) -> str:
        active = [f"- {pid}: {p.title}" for pid, p in world.protocols.items() if not p.archived]
        return DESIGN_PROMPT.format(
            partner_name=config.partner_names(config.agents[0]),
            module_list="\n".join(active) if active else "(none yet)",
        )

    def output_schema(self) -> Type[BaseModel]:
        return CodeArtifactOutput

    def apply(self, world: WorldState, output: BaseModel, cycle: CycleState) -> str:
        assert isinstance(output, CodeArtifactOutput)
        output.artifact_id = safe_artifact_id(
            output.artifact_id, fallback=f"module_cycle{cycle.cycle_id}"
        )
        aid = output.artifact_id
        body = output.content
        if output.tests.strip():
            body += f"\n\n# --- tests ---\n{output.tests}"

        if output.action.value == "create" or aid not in world.protocols:
            world.protocols[aid] = ProtocolDocument(
                protocol_id=aid, title=output.title, content=body,
                version=1, created_cycle=cycle.cycle_id,
                last_modified_cycle=cycle.cycle_id,
            )
        else:
            proto = world.protocols[aid]
            proto.content = body
            proto.title = output.title
            proto.version += 1
            proto.last_modified_cycle = cycle.cycle_id
        return aid

    def evaluation_prompt(
        self, world: WorldState, cycle: CycleState, doctrine_context: str
    ) -> str | None:
        proposal = cycle.proposed_protocol
        if not proposal:
            return None
        return EVALUATION_PROMPT.format(
            doctrine_context=doctrine_context,
            title=proposal.get("title", ""),
            artifact_id=proposal.get("artifact_id", ""),
            language=proposal.get("language", "python"),
            action=proposal.get("action", ""),
            content=proposal.get("content", ""),
            tests=proposal.get("tests", "(none supplied)"),
        )
