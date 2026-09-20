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

    @pytest.mark.parametrize("variant", VARIANTS)
    def test_every_variant_scores_the_tasks_own_dimensions(self, task, variant):
        """A variant must not quietly rescore on a different rubric."""
        schema = evaluation_schema_for(task, variant)
        scores = schema.model_fields["scores"].annotation
        assert set(scores.model_fields) == set(task.dimensions)

    @pytest.mark.parametrize("variant", VARIANTS)
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

    @pytest.mark.parametrize("variant", VARIANTS)
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
