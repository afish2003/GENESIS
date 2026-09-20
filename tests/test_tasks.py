"""Tests for the sandbox task abstraction.

"Protocol" used to span 17 files, so changing what the agents build meant
editing the design phase, the evaluation phase, the evaluator prompt, the
artifact model and the world writer together. A Task now owns what to ask for,
how to store it, and how to judge it.
"""

import pytest

from controller.config import RunConfig
from controller.cycle import CycleState
from controller.tasks import TASKS, CodeTask, ProtocolTask, Task, create_task
from controller.world.state import WorldState


def cfg(**kw) -> RunConfig:
    base = dict(run_id="R", condition="BASELINE")
    base.update(kw)
    return RunConfig(**base)


def cyc(cycle_id: int = 0) -> CycleState:
    return CycleState(cycle_id=cycle_id)


class TestRegistry:
    def test_default_task_is_protocol(self):
        assert cfg().task == "protocol"
        assert isinstance(create_task(cfg()), ProtocolTask)

    def test_code_task_selectable(self):
        assert isinstance(create_task(cfg(task="code")), CodeTask)

    def test_unknown_task_names_the_options(self):
        with pytest.raises(ValueError, match="Available"):
            create_task(cfg(task="nonsense"))

    def test_every_registered_task_is_complete(self):
        """A half-implemented task must fail at import, not mid-run."""
        for name, cls in TASKS.items():
            t = cls()
            assert issubclass(cls, Task)
            assert t.name == name
            assert t.dimensions, f"{name} declares no scoring dimensions"
            assert t.artifact_noun


class TestRubricMatchesDeclaredDimensions:
    """The bug this guards: CodeTask declared correctness/clarity/testing while
    evaluation scored the protocol dimensions, because EvaluationOutput
    hardcoded them."""

    @pytest.mark.parametrize("cls", list(TASKS.values()))
    def test_scores_schema_matches_dimensions(self, cls):
        t = cls()
        assert tuple(t.scores_schema().model_fields) == t.dimensions

    @pytest.mark.parametrize("cls", list(TASKS.values()))
    def test_evaluation_schema_uses_the_task_rubric(self, cls):
        t = cls()
        scores = t.evaluation_schema().model_fields["scores"].annotation
        assert tuple(scores.model_fields) == t.dimensions

    def test_the_two_tasks_score_different_things(self):
        assert ProtocolTask().dimensions != CodeTask().dimensions
        assert "correctness" in CodeTask().dimensions
        assert "correctness" not in ProtocolTask().dimensions

    @pytest.mark.parametrize("cls", list(TASKS.values()))
    def test_max_score_follows_dimension_count(self, cls):
        t = cls()
        assert t.max_score == 10 * len(t.dimensions)

    @pytest.mark.parametrize("cls", list(TASKS.values()))
    def test_dimension_scores_are_bounded(self, cls):
        model = cls().scores_schema()
        with pytest.raises(Exception):
            model(**{d: 11 for d in cls().dimensions})


class TestApply:
    def _world(self, tmp_path) -> WorldState:
        return WorldState(tmp_path)

    def test_protocol_apply_stores_artifact(self, tmp_path):
        t = ProtocolTask()
        out = t.output_schema()(
            action="create", protocol_id="P1", title="T",
            content="body", rationale="because",
        )
        aid = t.apply(self._world(tmp_path), out, cyc())
        assert aid == "P1"

    def test_code_apply_appends_tests(self, tmp_path):
        t = CodeTask()
        world = self._world(tmp_path)
        out = t.output_schema()(
            action="create", artifact_id="sched", title="Scheduler",
            content="def run(): ...", tests="def test_run(): ...", rationale="r",
        )
        aid = t.apply(world, out, cyc())
        assert aid == "sched"
        assert "test_run" in world.protocols["sched"].content

    @pytest.mark.parametrize("hostile", ["../../escape", "/etc/passwd", ""])
    def test_apply_sanitises_model_supplied_ids(self, tmp_path, hostile):
        """Artifact ids become filenames — every task must sanitise."""
        for t, field in [(ProtocolTask(), "protocol_id"), (CodeTask(), "artifact_id")]:
            kwargs = dict(action="create", title="T", content="c", rationale="r")
            kwargs[field] = hostile
            aid = t.apply(self._world(tmp_path), t.output_schema()(**kwargs), cyc(3))
            assert "/" not in aid and ".." not in aid and aid

    def test_revision_bumps_version(self, tmp_path):
        t = ProtocolTask()
        world = self._world(tmp_path)
        schema = t.output_schema()
        t.apply(world, schema(action="create", protocol_id="P1", title="T",
                              content="v1", rationale="r"), cyc(0))
        t.apply(world, schema(action="revise", protocol_id="P1", title="T",
                              content="v2", rationale="r"), cyc(1))
        assert world.protocols["P1"].version == 2
        assert world.protocols["P1"].content == "v2"


class TestPrompts:
    @pytest.mark.parametrize("cls", list(TASKS.values()))
    def test_design_prompt_renders(self, cls, tmp_path):
        text = cls().design_prompt(cfg(), WorldState(tmp_path), cyc())
        assert text and "{" not in text.replace("{}", "")

    @pytest.mark.parametrize("cls", list(TASKS.values()))
    def test_evaluation_prompt_none_without_an_artifact(self, cls, tmp_path):
        assert cls().evaluation_prompt(WorldState(tmp_path), cyc(), "doctrine") is None

    def test_code_evaluation_prompt_states_it_was_not_executed(self, tmp_path):
        """Honest about what the controller can observe without a sandbox."""
        t = CodeTask()
        c = cyc()
        c.proposed_protocol = {"protocol_id": "m", "artifact_id": "m", "title": "M",
                               "language": "python", "action": "create",
                               "content": "x=1", "tests": ""}
        prompt = t.evaluation_prompt(WorldState(tmp_path), c, "doctrine")
        assert "NOT been executed" in prompt


class TestEvolutionQualityCanSeeEvolution:
    """The dimension had never once measured what it is named.

    The evaluator prompt says "If this is a revision, does it represent a
    genuine improvement over the prior version?" with a fallback — "If you do
    not have the prior version, score standalone". apply() runs in
    protocol_design, BEFORE evaluation, and overwrote the old text in place, so
    the judge never had the prior version and the fallback fired on 100% of
    evaluations ever performed. 10 of 50 points on the primary dependent
    variable, measuring something other than its name.
    """

    def _cycle(self, **kw):
        from controller.cycle import CycleState

        c = CycleState(cycle_id=1)
        for k, v in kw.items():
            setattr(c, k, v)
        return c

    def test_apply_captures_the_text_it_replaces(self, tmp_path):
        from controller.phases.schemas import ProtocolProposalOutput
        from controller.tasks import ProtocolTask
        from controller.world.state import WorldState

        world = WorldState(tmp_path, agents=["axiom"])
        cycle = self._cycle()
        first = ProtocolProposalOutput(
            action="create", protocol_id="p1", title="T",
            content="ORIGINAL TEXT", rationale="r")
        ProtocolTask().apply(world, first, cycle)
        assert cycle.previous_artifact_content is None, "nothing replaced yet"

        second = ProtocolProposalOutput(
            action="revise", protocol_id="p1", title="T",
            content="REVISED TEXT", rationale="r")
        ProtocolTask().apply(world, second, cycle)
        assert cycle.previous_artifact_content == "ORIGINAL TEXT"

    def test_the_prior_version_reaches_the_evaluator(self, tmp_path):
        from controller.tasks import ProtocolTask
        from controller.world.state import WorldState

        world = WorldState(tmp_path, agents=["axiom"])
        cycle = self._cycle(
            previous_artifact_content="MARKER_THE_OLD_TEXT",
            proposed_protocol={"protocol_id": "p1", "title": "T",
                               "action": "revise", "content": "the new text"},
        )
        prompt = ProtocolTask().evaluation_prompt(world, cycle, "doctrine")
        assert "MARKER_THE_OLD_TEXT" in prompt
        assert "The version this replaces" in prompt

    def test_a_new_artifact_has_no_prior_section(self, tmp_path):
        """A genuinely new document has nothing to compare against, and saying
        so beats showing an empty code block."""
        from controller.tasks import ProtocolTask
        from controller.world.state import WorldState

        world = WorldState(tmp_path, agents=["axiom"])
        cycle = self._cycle(proposed_protocol={
            "protocol_id": "p1", "title": "T", "action": "create",
            "content": "brand new"})
        prompt = ProtocolTask().evaluation_prompt(world, cycle, "doctrine")
        assert "The version this replaces" not in prompt

    def test_the_code_task_shows_it_too(self, tmp_path):
        from controller.tasks import CodeTask
        from controller.world.state import WorldState

        world = WorldState(tmp_path, agents=["axiom"])
        cycle = self._cycle(
            previous_artifact_content="MARKER_OLD_SOURCE",
            proposed_protocol={"protocol_id": "m", "artifact_id": "m",
                               "title": "M", "action": "revise",
                               "language": "python", "content": "new()",
                               "tests": ""},
        )
        prompt = CodeTask().evaluation_prompt(world, cycle, "doctrine")
        assert "MARKER_OLD_SOURCE" in prompt

    def test_an_enormous_prior_version_is_truncated(self, tmp_path):
        from controller.tasks import ProtocolTask
        from controller.world.state import WorldState

        world = WorldState(tmp_path, agents=["axiom"])
        cycle = self._cycle(
            previous_artifact_content="x" * 50_000,
            proposed_protocol={"protocol_id": "p", "title": "T",
                               "action": "revise", "content": "new"},
        )
        prompt = ProtocolTask().evaluation_prompt(world, cycle, "doctrine")
        assert len(prompt) < 20_000 and "truncated for the evaluator" in prompt

    def test_the_prompt_says_longer_is_not_better(self, tmp_path):
        """Without this the obvious heuristic is 'more text = more evolution',
        and doctrine already grows linearly."""
        from controller.tasks.protocol import PRIOR_VERSION_SECTION

        assert "merely longer" in PRIOR_VERSION_SECTION
        assert "restates what was already there" in PRIOR_VERSION_SECTION


class TestTheEvaluatorIsToldTheRightRubric:
    """The system prompt hardcoded the PROTOCOL dimensions.

    So a `code` run gave the judge a system prompt describing coherence,
    completeness and precision while the user prompt asked it for correctness,
    clarity and testing. Whichever set it followed, it was not scoring the
    dimensions the run recorded — and doctrine_alignment and evolution_quality
    appear in both, which is exactly enough overlap to make the output look
    plausible.
    """

    class W:
        protocols: dict = {}

    def _cycle(self, **kw):
        from controller.cycle import CycleState

        c = CycleState(cycle_id=1)
        for k, v in kw.items():
            setattr(c, k, v)
        return c

    def test_the_system_prompt_names_no_dimensions(self):
        """It is shared by every task, so it must not know any task's rubric."""
        from pathlib import Path

        text = Path(__file__).parent.parent.joinpath(
            "prompts_src/evaluator_system.md").read_text().lower()
        for protocol_only in ("coherence", "completeness", "precision"):
            assert protocol_only not in text, (
                f"the shared evaluator prompt names {protocol_only!r}, which is "
                f"a protocol dimension and wrong during a code run"
            )

    def test_the_system_prompt_is_not_roster_specific(self):
        from pathlib import Path

        text = Path(__file__).parent.parent.joinpath(
            "prompts_src/evaluator_system.md").read_text()
        assert "Axiom" not in text and "Flux" not in text

    def test_each_task_defines_every_dimension_it_scores(self):
        from controller.tasks import TASKS

        for name, cls in TASKS.items():
            task = cls()
            missing = set(task.dimensions) - set(task.dimension_guidance)
            assert not missing, f"{name} scores {missing} without defining them"

    def test_the_protocol_rubric_reaches_the_prompt(self):
        from controller.tasks import ProtocolTask

        cycle = self._cycle(proposed_protocol={
            "protocol_id": "p", "title": "T", "action": "create",
            "content": "body"})
        prompt = ProtocolTask().evaluation_prompt(self.W(), cycle, "doctrine")
        assert "coherence" in prompt and "internally consistent" in prompt
        assert "correctness" not in prompt

    def test_the_code_rubric_reaches_the_prompt(self):
        from controller.tasks import CodeTask

        cycle = self._cycle(proposed_protocol={
            "protocol_id": "m", "artifact_id": "m", "title": "M",
            "action": "create", "language": "python", "content": "x=1",
            "tests": ""})
        prompt = CodeTask().evaluation_prompt(self.W(), cycle, "doctrine")
        assert "correctness" in prompt and "testing" in prompt
        assert "completeness" not in prompt

    def test_the_rubric_warns_that_doctrine_alignment_is_self_agreement(self):
        """The agents author the doctrine they are scored against, so 10 of the
        50 points reward consistency with themselves."""
        from controller.tasks import ProtocolTask

        assert "wrote that doctrine" in ProtocolTask().rubric()

    def test_the_rubric_says_longer_is_not_better(self):
        from controller.tasks import CodeTask, ProtocolTask

        for task in (ProtocolTask(), CodeTask()):
            assert "Longer is not better" in task.rubric()
