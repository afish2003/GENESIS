"""A sandbox task where the agents author code rather than prose.

Exists to demonstrate that the task axis is real: the cycle loop, doctrine,
identity, memory, retrieval and logging are untouched, and only what gets built
and how it is scored changes.

By default the artifact is stored and NOT executed: running agent-authored code
requires a sandbox, which refuses by default (docs/containment_design.md), so
scoring judges the code as a written artifact — honest about what the controller
can observe without execution.

When the researcher enables execution, the same module is run in a throwaway
container and the result is fed to the evaluator, so `correctness` is scored
against what the code *did* rather than how it reads. Both the design prompt and
the evaluation prompt say which of the two is happening; an agent told its code
is only read, whose code is then run, is being lied to about its own task.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Type

from pydantic import BaseModel

from controller.phases.schemas import CodeArtifactOutput
from controller.sandbox.schemas import ExecutionRequest
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

{execution_note}"""

#: What the agent is told about the fate of its code. These must stay truthful:
#: `correctness` is scored very differently depending on which one applies.
NOT_EXECUTED_NOTE = """Your code will be read and reviewed, not executed. Write it as if it will run: correct, readable, and honest about what it does not handle."""

EXECUTED_NOTE = """Your code WILL BE RUN. It executes in an isolated container with no network access, a short time limit, and nothing from outside this workspace: standard library only, no package installs, no files but the ones you write here. Its output — and any traceback — is shown to the evaluator and becomes part of the record.

Your source is saved as `module.py`. If you supply tests they are saved as `test_module.py` and run INSTEAD of the module, so they must `import module` (or `from module import ...`) themselves. Use plain asserts or `unittest`; nothing else is installed.

The workspace is read-only. If you need scratch space, write under `/tmp`: it is RAM-backed, shares the container's memory budget, and is thrown away when the run ends.

Make the run say something a reader can learn from — print what the code does, not that it ran. Prefer code that demonstrates itself over code that merely defines itself."""

EVALUATION_PROMPT = """Evaluate the following code module. {preamble}

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
{execution_report}
Score on the five dimensions (correctness, clarity, doctrine_alignment, testing, evolution_quality), each 0-10. {correctness_note} Provide justifications, a total score, and an overall assessment."""

NOT_EXECUTED_PREAMBLE = "It has NOT been executed — judge it as written."
NOT_EXECUTED_CORRECTNESS = "Correctness means whether the code would do what it claims if run."

EXECUTED_PREAMBLE = "It was executed in an isolated container; the result is below."
EXECUTED_CORRECTNESS = (
    "Correctness must be judged against the execution result below, not against "
    "how the code reads. Code that looks right and crashed is not correct. Code "
    "that ran cleanly but only because it asserts nothing is not correct either "
    "— say so rather than rewarding an empty run."
)

#: The execution result, as the evaluator sees it. A refusal is reported rather
#: than hidden: an evaluator told nothing about execution would otherwise score
#: `correctness` as if the code had run and passed.
EXECUTION_REPORT = """
## Execution result

**Outcome**: {outcome}{exit_code} in {duration:.2f}s{limits}

### stdout
```
{stdout}
```

### stderr
```
{stderr}
```
"""


class CodeTask(Task):
    name = "code"
    artifact_noun = "code module"
    dimensions = ("correctness", "clarity", "doctrine_alignment", "testing", "evolution_quality")

    def design_prompt(self, config: RunConfig, world: WorldState, cycle: CycleState) -> str:
        active = [f"- {pid}: {p.title}" for pid, p in world.protocols.items() if not p.archived]
        return DESIGN_PROMPT.format(
            partner_name=config.partner_names(config.agents[0]),
            module_list="\n".join(active) if active else "(none yet)",
            execution_note=(
                EXECUTED_NOTE if getattr(config, "execution_enabled", False)
                else NOT_EXECUTED_NOTE
            ),
        )

    def output_schema(self) -> Type[BaseModel]:
        return CodeArtifactOutput

    def apply(self, world: WorldState, output: BaseModel, cycle: CycleState,
              max_tokens: int | None = None) -> str:
        assert isinstance(output, CodeArtifactOutput)
        output.artifact_id = safe_artifact_id(
            output.artifact_id, fallback=f"module_cycle{cycle.cycle_id}"
        )
        aid = output.artifact_id
        body = output.content
        if output.tests.strip():
            body += f"\n\n# --- tests ---\n{output.tests}"

        if max_tokens:
            body, truncated = self.enforce_length(body, max_tokens)
            if truncated:
                import logging
                logging.getLogger(__name__).warning(
                    "Cycle %d: %s truncated at the configured length limit",
                    cycle.cycle_id, aid,
                )

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
        ran = getattr(cycle, "execution_result", None)
        return EVALUATION_PROMPT.format(
            preamble=EXECUTED_PREAMBLE if ran else NOT_EXECUTED_PREAMBLE,
            correctness_note=EXECUTED_CORRECTNESS if ran else NOT_EXECUTED_CORRECTNESS,
            execution_report=self._execution_report(ran),
            doctrine_context=doctrine_context,
            title=proposal.get("title", ""),
            artifact_id=proposal.get("artifact_id", ""),
            language=proposal.get("language", "python"),
            action=proposal.get("action", ""),
            content=proposal.get("content", ""),
            tests=proposal.get("tests", "(none supplied)"),
        )

    @staticmethod
    def _execution_report(result: dict | None) -> str:
        if not result:
            return ""
        exit_code = result.get("exit_code")
        limits = " (output truncated)" if result.get("truncated") else ""
        detail = result.get("detail") or ""
        if detail:
            limits += f" — {detail}"
        return EXECUTION_REPORT.format(
            outcome=result.get("outcome", "?"),
            exit_code="" if exit_code is None else f", exit code {exit_code},",
            duration=result.get("duration_seconds", 0.0),
            limits=limits,
            stdout=(result.get("stdout") or "").strip() or "(no output)",
            stderr=(result.get("stderr") or "").strip() or "(no output)",
        )

    def execution_request(
        self, world: WorldState, cycle: CycleState, timeout_seconds: float
    ) -> ExecutionRequest | None:
        """Run this cycle's module, or its tests if it has any.

        Filenames are fixed rather than derived from `artifact_id`: an id like
        `3d-grid` sanitises to a valid filename and an invalid module name, so
        tests importing it would fail for a reason that has nothing to do with
        the code the agents wrote.
        """
        proposal = cycle.proposed_protocol
        if not proposal or not (proposal.get("content") or "").strip():
            return None

        files = {"module.py": proposal["content"]}
        entrypoint = "python module.py"
        tests = (proposal.get("tests") or "").strip()
        if tests:
            files["test_module.py"] = tests
            entrypoint = "python test_module.py"

        return ExecutionRequest(
            files=files,
            entrypoint=entrypoint,
            timeout_seconds=timeout_seconds,
            requested_by=proposal.get("proposing_agent", ""),
            cycle_id=cycle.cycle_id,
        )
