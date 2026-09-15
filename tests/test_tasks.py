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
