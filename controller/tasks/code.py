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

from pathlib import Path
from typing import TYPE_CHECKING, Type

from pydantic import BaseModel

from controller.phases.schemas import CodeArtifactOutput
from controller.sandbox.schemas import ExecutionRequest
from controller.tasks.base import Task
from controller.tasks.protocol import prior_version_section
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

Your source and your tests are saved together as a single file, `module.py`, and run as one — so your tests can call your functions directly, with no import. Put any test calls at the bottom so they actually execute when the file runs. Use plain asserts or `unittest`; nothing else is installed.

The workspace is read-only. If you need scratch space, write under `/tmp`: it is RAM-backed, shares the container's memory budget, and is thrown away when the run ends.

Make the run say something a reader can learn from — print what the code does, not that it ran. Prefer code that demonstrates itself over code that merely defines itself."""

PROJECT_NOTE = """

## Your project

`/project` is a persistent directory that SURVIVES between cycles and is your working directory when your code runs. It is the only thing you have that outlasts your own memory. Whatever you write there is still there next cycle and every cycle after.

**This cycle's program should move the project forward — not start over.** Read what is already there, then add to it, fix it, refactor it, or test it. A cycle spent writing a self-contained file that touches nothing is a cycle the project does not grow.

Concretely, your program can `open(...).write(...)` new source files under `/project`, edit existing ones, and import them to check they work. Build something across cycles that you could not build in one.

### What is in `/project` right now

{tree}
"""

PREVIOUS_RUN_NOTE = """

### What your last program did

It exited {exit_code} ({outcome}) after {duration:.1f}s.

```
{output}
```
{verdict}"""

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
{prior_version}
## Tests

```
{tests}
```
{execution_report}
Score on these dimensions:

{rubric}

{correctness_note} Provide a one-sentence justification per dimension, a total score that is the sum of them, and an overall assessment."""

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
    dimension_guidance = {
        "correctness": "Does the code do what it claims? Judge against the "
                       "execution result when one is shown, not against how it "
                       "reads.",
        "clarity": "Could another author work on this? Are names, structure "
                   "and control flow easy to follow?",
        "doctrine_alignment": "Is it consistent with the shared doctrine quoted "
                              "above? The agents wrote that doctrine, so reward "
                              "genuine consistency rather than restatement.",
        "testing": "Do the tests exercise what matters, including the cases "
                   "likely to break? Tests that assert nothing score low.",
        "evolution_quality": "If a prior version is shown, is this a real "
                             "improvement on it? If new, does it earn its place "
                             "in the codebase? Longer is not better.",
    }

    def design_prompt(self, config: RunConfig, world: WorldState, cycle: CycleState) -> str:
        active = [f"- {pid}: {p.title}" for pid, p in world.protocols.items() if not p.archived]
        note = (EXECUTED_NOTE if getattr(config, "execution_enabled", False)
                else NOT_EXECUTED_NOTE)
        tree = self._project_tree(config)
        if tree:
            note += PROJECT_NOTE.format(tree=tree)
        note += self._previous_run_note(cycle)
        return DESIGN_PROMPT.format(
            partner_name=config.partner_names(config.agents[0]),
            module_list="\n".join(active) if active else "(none yet)",
            execution_note=note,
        )

    @staticmethod
    def _previous_run_note(cycle) -> str:
        """Show the agents what their last program actually did.

        Without this the feedback loop is open: a traceback goes to the
        evaluator, who marks the cycle down, and the agents write the next
        module having never seen the error. Closing it is what makes a run a
        development process rather than a sequence of unrelated files.
        """
        result = getattr(cycle, "previous_execution", None) if cycle else None
        if not result:
            return ""

        stdout = (result.get("stdout") or "").strip()
        stderr = (result.get("stderr") or "").strip()
        output = "\n".join(filter(None, [stdout, stderr])) or "(no output)"
        if len(output) > 1500:
            output = output[:1500] + "\n...[clipped]"

        ok = result.get("outcome") == "OK"
        verdict = ("\nIt ran cleanly. Build on it."
                   if ok else
                   "\nIt did not run cleanly. Fixing this is a legitimate and "
                   "useful thing to do with this cycle.")
        return PREVIOUS_RUN_NOTE.format(
            exit_code=result.get("exit_code", "?"),
            outcome=result.get("outcome", "?"),
            duration=result.get("duration_seconds", 0.0),
            output=output,
            verdict=verdict,
        )

    @staticmethod
    def _project_tree(config, limit: int = 200) -> str:
        """What the agents have built so far, as they would see it.

        Read from the host side of the same directory the container mounts.
        Without this the persistent volume is invisible: the agents would have
        to remember their own codebase from memory summaries, which is exactly
        the channel MEM_RESET removes.
        """
        root = getattr(config, "sandbox_project_dir", None)
        if not root or not getattr(config, "execution_enabled", False):
            return ""
        root = Path(root)
        if not root.is_dir():
            return ""
        entries = []
        for path in sorted(root.rglob("*")):
            if any(part.startswith(".") for part in path.relative_to(root).parts):
                continue
            rel = path.relative_to(root)
            if path.is_dir():
                entries.append(f"  {rel}/")
            else:
                entries.append(f"  {rel}  ({path.stat().st_size} bytes)")
            if len(entries) >= limit:
                entries.append(f"  ... (listing truncated at {limit} entries)")
                break
        return "\n".join(entries) if entries else "  (empty — nothing built yet)"

    def output_schema(self) -> Type[BaseModel]:
        return CodeArtifactOutput

    @staticmethod
    def full_source(content: str, tests: str) -> str:
        """The module and its tests as one file — what is stored AND what runs.

        These used to diverge: `apply` concatenated them into a single stored
        artifact while `execution_request` wrote them as module.py and
        test_module.py and ran the second. So the evaluator scored one thing and
        the container ran another, and the split version only worked if the
        agent wrote an `import module` line that nothing in its own artifact
        needed. It reliably did not, and every execution died with a NameError
        that was this packaging, not the agents' code.

        One file removes the import requirement entirely — tests share the
        module's namespace — and makes "what ran" and "what was scored"
        the same text by construction.
        """
        body = content
        if tests.strip():
            body += f"\n\n# --- tests ---\n{tests}"
        return body

    def apply(self, world: WorldState, output: BaseModel, cycle: CycleState,
              max_tokens: int | None = None) -> str:
        assert isinstance(output, CodeArtifactOutput)
        output.artifact_id = safe_artifact_id(
            output.artifact_id, fallback=f"module_cycle{cycle.cycle_id}"
        )
        aid = output.artifact_id
        body = self.full_source(output.content, output.tests)

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
            cycle.previous_artifact_content = proto.content
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
            prior_version=prior_version_section(cycle),
            rubric=self.rubric(),
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

        return ExecutionRequest(
            files={"module.py": self.full_source(
                proposal["content"], proposal.get("tests") or "")},
            entrypoint="python /workspace/module.py",
            timeout_seconds=timeout_seconds,
            requested_by=proposal.get("proposing_agent", ""),
            cycle_id=cycle.cycle_id,
        )
