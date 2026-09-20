"""The judge variants, and the one mechanism in them that is checkable.

Background: the 2026-09-20 battery produced 71 evaluations in which
completeness scored 9 in 69 of them, doctrine_alignment 7 in 68, coherence 8
in 68 — while the documents being scored had pairwise text similarity of
0.070. The evaluator prompt already told it to use the whole range. So the
fix has to be structural, and these tests pin the structure.
"""

import pytest

from controller.config import RunConfig
from controller.judge import (
    MINOR_DEFECT_MAX,
    NO_DEFECT_MIN,
    SCORING_VARIANTS,
    VARIANTS,
    anchored_violations,
    build_evaluation_prompt,
    evaluation_schema_for,
    load_judge_system_prompt,
)
from controller.tasks import create_task


@pytest.fixture(scope="module")
def task():
    return create_task(RunConfig(run_id="T", condition="BASELINE", task="protocol"))


class TestFieldOrderIsTheMechanism:
    """A structured decoder emits fields in declaration order, so whichever
    comes first is what the model reasons with. `current` puts scores first,
    which means every justification in the battery was written to fit a number
    already chosen."""

    def test_current_reproduces_the_battery_exactly(self, task):
        """The baseline must be the real thing, or the comparison is against
        a remembered version of it."""
        fields = list(evaluation_schema_for(task, "current").model_fields)
        assert fields == list(task.evaluation_schema().model_fields)
        assert fields.index("scores") < fields.index("justifications")

    @pytest.mark.parametrize("variant,reason_field", [
        ("evidence_first", "justifications"),
        ("anchored", "defects"),
    ])
    def test_the_reasoning_field_precedes_the_scores(self, task, variant, reason_field):
        fields = list(evaluation_schema_for(task, variant).model_fields)
        assert fields.index(reason_field) < fields.index("scores"), (
            f"{variant} lets the model pick a number before it reasons")

    @pytest.mark.parametrize("variant", SCORING_VARIANTS)
    def test_every_variant_scores_the_tasks_own_dimensions(self, task, variant):
        """A variant must not quietly rescore on a different rubric."""
        schema = evaluation_schema_for(task, variant)
        scores = schema.model_fields["scores"].annotation
        assert set(scores.model_fields) == set(task.dimensions)

    @pytest.mark.parametrize("variant", SCORING_VARIANTS)
    def test_total_is_bounded_by_the_rubric(self, task, variant):
        schema = evaluation_schema_for(task, variant)
        md = schema.model_fields["total_score"].metadata
        assert any(getattr(m, "le", None) == task.max_score for m in md)


class TestAnchoredCapsAreEnforceable:
    """The anchored variant's claim is that a named defect caps the score.
    That is only worth anything if a reader — or this function — can check it.
    """

    DIMS = ["coherence", "completeness"]

    def test_no_defect_with_a_low_score_is_a_violation(self):
        v = anchored_violations(
            {"coherence": "NO DEFECT FOUND", "completeness": "missing scope"},
            {"coherence": 6, "completeness": 5})
        assert any("coherence" in x for x in v)
        assert not any("completeness" in x for x in v)

    def test_a_named_defect_with_a_high_score_is_a_violation(self):
        v = anchored_violations(
            {"coherence": "section 3 contradicts section 1"},
            {"coherence": 9})
        assert v and "coherence" in v[0]

    def test_a_consistent_pair_is_clean(self):
        assert anchored_violations(
            {"coherence": "NO DEFECT FOUND", "completeness": "no limitations section"},
            {"coherence": 10, "completeness": 6}) == []

    def test_case_and_whitespace_do_not_defeat_the_check(self):
        assert anchored_violations(
            {"coherence": "  no defect found  "}, {"coherence": 9}) == []

    def test_an_empty_defect_is_not_treated_as_a_named_one(self):
        """A model that returns "" has told us nothing; do not invent a cap
        from it and report a violation that is really a parse problem."""
        assert anchored_violations({"coherence": ""}, {"coherence": 10}) == []

    def test_the_boundaries_are_where_the_prompt_says(self):
        assert anchored_violations(
            {"c": "NO DEFECT FOUND"}, {"c": NO_DEFECT_MIN}) == []
        assert anchored_violations({"c": "real defect"}, {"c": MINOR_DEFECT_MAX}) == []
        assert anchored_violations({"c": "real defect"}, {"c": MINOR_DEFECT_MAX + 1}) != []


class TestPrompts:
    @pytest.mark.parametrize("variant", VARIANTS)
    def test_every_variant_has_a_system_prompt_naming_the_judge_role(self, variant):
        p = load_judge_system_prompt(variant)
        assert "Evaluator" in p and "fresh context" in p

    def test_undisclosed_framing_does_not_name_the_experiment(self):
        """The judge's prompt is not shown to agents, but the leak-scanning
        habit is cheap and the last framing leak was in a file nobody scanned.
        """
        assert "GENESIS" not in load_judge_system_prompt("anchored", framing="undisclosed")

    def test_anchored_prompt_states_the_caps_it_will_be_checked_against(self):
        p = load_judge_system_prompt("anchored")
        assert "NO DEFECT FOUND" in p
        assert f"{NO_DEFECT_MIN}-10" in p

    @pytest.mark.parametrize("variant", SCORING_VARIANTS)
    def test_the_prompt_contains_the_artifact_and_the_rubric(self, task, variant):
        doc = {"title": "T", "protocol_id": "p1", "content": "UNIQUE-BODY-TEXT"}
        prompt = build_evaluation_prompt(task, doc, variant)
        assert "UNIQUE-BODY-TEXT" in prompt
        for dim in task.dimensions:
            assert dim in prompt

    def test_anchored_asks_for_defects_and_current_does_not(self, task):
        doc = {"title": "T", "protocol_id": "p", "content": "x" * 50}
        assert "NO DEFECT FOUND" in build_evaluation_prompt(task, doc, "anchored")
        assert "NO DEFECT FOUND" not in build_evaluation_prompt(task, doc, "current")


class TestTheVariantReachesTheLiveEvaluationPhase:
    """Wiring nobody exercises is wiring that quietly does not work.

    The variant has to change two things together — the tail of the prompt and
    the schema the answer is parsed against. If only one moves, the judge is
    asked for defects and parsed for justifications, or the reverse, and the
    phase fails or silently drops the reasoning.
    """

    def _prompt_for(self, variant, task_name="protocol"):
        from controller.config import RunConfig
        from controller.judge import variant_instruction
        from controller.tasks import create_task
        from controller.world.artifacts import ProtocolDocument

        cfg = RunConfig(run_id="T", condition="BASELINE", task=task_name)
        task = create_task(cfg)

        class _World:
            protocols = {"p1": ProtocolDocument(
                protocol_id="p1", title="T", content="body", version=1)}
            doctrine = {}

        class _Cycle:
            cycle_id = 3
            proposed_protocol = {"protocol_id": "p1", "title": "T",
                                 "content": "body text", "action": "create",
                                 "artifact_id": "p1", "language": "python",
                                 "tests": "t"}
            previous_artifact_content = ""
            execution_result = None

        return task.evaluation_prompt(
            _World(), _Cycle(), "doctrine here",
            instruction=variant_instruction(variant))

    @pytest.mark.parametrize("task_name", ["protocol", "code"])
    def test_anchored_replaces_the_default_tail_rather_than_appending(self, task_name):
        """Both instructions present at once would ask for two output shapes."""
        from controller.tasks.protocol import DEFAULT_EVALUATION_INSTRUCTION

        prompt = self._prompt_for("anchored", task_name)
        assert "NO DEFECT FOUND" in prompt
        assert DEFAULT_EVALUATION_INSTRUCTION not in prompt

    @pytest.mark.parametrize("task_name", ["protocol", "code"])
    def test_current_still_asks_exactly_what_it_used_to(self, task_name):
        from controller.tasks.protocol import DEFAULT_EVALUATION_INSTRUCTION

        prompt = self._prompt_for("current", task_name)
        assert DEFAULT_EVALUATION_INSTRUCTION in prompt
        assert "NO DEFECT FOUND" not in prompt

    @pytest.mark.parametrize("task_name", ["protocol", "code"])
    def test_the_artifact_and_rubric_survive_the_substitution(self, task_name):
        """The slot must not have eaten the rest of the prompt."""
        from controller.config import RunConfig
        from controller.tasks import create_task

        prompt = self._prompt_for("anchored", task_name)
        assert "body text" in prompt and "doctrine here" in prompt
        # Each task's OWN dimensions: `code` does not score coherence.
        task = create_task(RunConfig(run_id="T", condition="BASELINE", task=task_name))
        for dim in task.dimensions:
            assert dim in prompt, f"{task_name} prompt lost {dim}"

    def test_the_phase_asks_for_the_shape_it_parses(self):
        """The prompt's requested field and the schema's field must agree."""
        from controller.config import RunConfig
        from controller.judge import evaluation_schema_for, reasoning_field
        from controller.tasks import create_task

        task = create_task(RunConfig(run_id="T", condition="BASELINE"))
        for variant in SCORING_VARIANTS:
            field = reasoning_field(variant)
            schema = evaluation_schema_for(task, variant)
            assert field in schema.model_fields, variant
            prompt = self._prompt_for(variant)
            # `current` predates the field names and asks in prose
            # ("a one-sentence justification per dimension"), so match the
            # stem rather than the identifier.
            stem = field.rstrip("s")
            assert stem in prompt.lower(), (
                f"{variant} asks for nothing resembling {field}")

    def test_config_rejects_a_variant_that_does_not_exist(self):
        from pydantic import ValidationError
        from controller.config import RunConfig

        with pytest.raises(ValidationError):
            RunConfig(run_id="T", condition="BASELINE", judge_variant="wishful")

    def test_config_accepts_every_variant_the_module_offers(self):
        from controller.config import RunConfig

        for variant in SCORING_VARIANTS:
            cfg = RunConfig(run_id="T", condition="BASELINE", judge_variant=variant)
            assert cfg.judge_variant == variant

    def test_the_default_is_the_benchmarked_baseline(self):
        """Nothing should change scoring until the bench says which wins."""
        from controller.config import RunConfig
        assert RunConfig(run_id="T", condition="BASELINE").judge_variant == "current"


class TestPairwiseShipsWithItsOwnMetric:
    """Pairwise writes improvement/quality_index, never scores/total_score.

    Eleven files read total_score and treat it as an absolute [0, 50] level.
    A pairwise comparison produces a [-5, +5] delta. Putting the delta in that
    field would corrupt every existing comparison without raising once, so the
    two never share a name.
    """

    def test_config_accepts_it(self):
        assert RunConfig(run_id="T", condition="BASELINE",
                         judge_variant="pairwise").judge_variant == "pairwise"

    def test_it_is_not_a_scoring_variant(self):
        assert "pairwise" in VARIANTS and "pairwise" not in SCORING_VARIANTS

    def test_its_schema_reasons_before_it_chooses(self):
        from controller.judge import pairwise_schema
        t = create_task(RunConfig(run_id="T", condition="BASELINE"))
        fields = list(pairwise_schema(t).model_fields)
        assert fields.index("justifications") < fields.index("choices")

    def test_it_chooses_on_every_dimension_the_task_declares(self):
        from controller.judge import pairwise_schema
        t = create_task(RunConfig(run_id="T", condition="BASELINE"))
        choices = pairwise_schema(t).model_fields["choices"].annotation
        assert set(choices.model_fields) == set(t.dimensions)

    def test_the_prompt_shows_both_versions_and_warns_about_position(self):
        from controller.judge import build_pairwise_prompt, load_judge_system_prompt
        t = create_task(RunConfig(run_id="T", condition="BASELINE"))
        p = build_pairwise_prompt(t, "FIRST-DOC", "SECOND-DOC")
        assert "FIRST-DOC" in p and "SECOND-DOC" in p
        assert "Position carries no information" in load_judge_system_prompt("pairwise")


class TestImprovementRespectsTheRandomisedSlot:
    """The new artifact's slot is randomised to cancel position bias, so the
    mapping must be told which slot it was. A judge that always answers "A"
    would otherwise read as steady improvement for a whole run."""

    from controller.judge import improvement_from_choices as _f

    def test_new_in_a_wins_when_a_is_chosen(self):
        from controller.judge import improvement_from_choices
        assert improvement_from_choices({"d": "A"}, new_is="A") == {"d": 1}

    def test_new_in_b_loses_when_a_is_chosen(self):
        from controller.judge import improvement_from_choices
        assert improvement_from_choices({"d": "A"}, new_is="B") == {"d": -1}

    def test_a_tie_is_zero_in_either_slot(self):
        from controller.judge import improvement_from_choices
        for slot in ("A", "B"):
            assert improvement_from_choices({"d": "tie"}, new_is=slot) == {"d": 0}

    def test_a_constant_answer_nets_zero_across_balanced_slots(self):
        """The property the randomisation exists for: a judge that always
        says "A" must average to no improvement, not to improvement."""
        from controller.judge import improvement_from_choices
        dims = ["a", "b", "c", "d", "e"]
        always_a = {d: "A" for d in dims}
        in_a = sum(improvement_from_choices(always_a, "A").values())
        in_b = sum(improvement_from_choices(always_a, "B").values())
        assert in_a + in_b == 0

    def test_net_is_bounded_by_the_dimension_count(self):
        from controller.judge import improvement_from_choices
        t = create_task(RunConfig(run_id="T", condition="BASELINE"))
        best = {d: "A" for d in t.dimensions}
        assert sum(improvement_from_choices(best, "A").values()) == len(t.dimensions)
