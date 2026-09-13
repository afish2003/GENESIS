"""Tests for doctrine revision apply semantics.

Before 2026-09-13 an approved revision appended a *description* of the change
instead of the change. Doctrine never actually changed, agents re-proposed the
same edit every cycle, and doctrine_diffs.jsonl logged approvals that made it
look like evolution was happening. Doctrine evolution is the primary dependent
variable, so this destroyed the measurement without ever failing.

These tests pin the replacement behaviour and, just as importantly, the guards
against a model silently truncating a document.
"""

import pytest

from controller.config import RunConfig
from controller.phases.doctrine_revision import MIN_RETAINED_FRACTION, apply_revision
from controller.phases.schemas import DoctrineRevisionProposal

CURRENT = """# Doctrine

## Working Principles

1. Evidence before assertion.
2. Name disagreements explicitly.
3. Prefer reversible decisions.
4. Record the reasoning, not only the conclusion.
"""


def proposal(**kw) -> DoctrineRevisionProposal:
    base = dict(
        target_document="doctrine.md",
        proposed_diff="Add a fifth principle on quality.",
        rationale="Quality was raised in evaluation feedback.",
        revised_content=CURRENT.rstrip() + "\n5. Consistent quality over quantity.\n",
    )
    base.update(kw)
    return DoctrineRevisionProposal(**base)


class TestReplaceMode:
    def test_writes_the_agents_document(self):
        text, how, why = apply_revision(CURRENT, proposal(), 1, "axiom")
        assert how == "replace"
        assert "5. Consistent quality over quantity." in text
        # The change is IN the document, not described beside it.
        assert "*Revision (cycle" not in text

    def test_preserves_untouched_content(self):
        text, _, _ = apply_revision(CURRENT, proposal(), 1, "axiom")
        for line in ["1. Evidence before assertion.",
                     "2. Name disagreements explicitly.",
                     "4. Record the reasoning, not only the conclusion."]:
            assert line in text

    def test_document_does_not_accumulate_changelog(self):
        """The old behaviour grew the file with notes on every cycle."""
        text = CURRENT
        for cycle in range(5):
            p = proposal(revised_content=text.rstrip() + f"\n{cycle+5}. Principle {cycle+5}.\n")
            text, _, _ = apply_revision(text, p, cycle, "axiom")
        assert "*Revision (cycle" not in text
        assert text.count("## Working Principles") == 1


class TestTruncationGuards:
    def test_empty_revised_content_refused(self):
        text, _, why = apply_revision(CURRENT, proposal(revised_content=""), 1, "axiom")
        assert text is None and "no revised_content" in why

    def test_description_instead_of_document_refused(self):
        """The exact failure mode that caused the original bug."""
        text, _, why = apply_revision(
            CURRENT,
            proposal(revised_content="Add a new working principle about quality."),
            1, "axiom",
        )
        assert text is None
        assert "truncation" in why

    def test_truncated_document_refused(self):
        half = CURRENT[: int(len(CURRENT) * 0.4)]
        text, _, why = apply_revision(CURRENT, proposal(revised_content=half), 1, "axiom")
        assert text is None and "retention floor" in why

    def test_at_the_floor_is_accepted(self):
        """A deliberate condensation just above the floor is a legitimate edit."""
        keep = CURRENT[: int(len(CURRENT) * (MIN_RETAINED_FRACTION + 0.2))] + "\nend."
        text, how, _ = apply_revision(CURRENT, proposal(revised_content=keep), 1, "axiom")
        assert text is not None and how == "replace"

    def test_no_op_refused(self):
        text, _, why = apply_revision(CURRENT, proposal(revised_content=CURRENT), 1, "axiom")
        assert text is None and "identical" in why

    def test_growth_is_always_allowed(self):
        big = CURRENT + "\n" + ("6. Another principle.\n" * 50)
        text, how, _ = apply_revision(CURRENT, proposal(revised_content=big), 1, "axiom")
        assert text is not None and how == "replace"


class TestLegacyAppendMode:
    def test_reproduces_old_behaviour(self):
        text, how, _ = apply_revision(CURRENT, proposal(), 3, "flux", mode="append")
        assert how == "append(legacy)"
        assert "*Revision (cycle 3, proposed by flux)*" in text
        assert text.startswith(CURRENT)

    def test_legacy_mode_ignores_revised_content(self):
        """Which is precisely why doctrine never changed."""
        text, _, _ = apply_revision(
            CURRENT, proposal(revised_content="TOTALLY NEW DOCUMENT"), 1, "axiom",
            mode="append",
        )
        assert "TOTALLY NEW DOCUMENT" not in text


class TestConfig:
    def test_replace_is_the_default(self):
        assert RunConfig(run_id="R", condition="BASELINE").doctrine_apply_mode == "replace"

    def test_append_still_selectable(self):
        cfg = RunConfig(run_id="R", condition="BASELINE", doctrine_apply_mode="append")
        assert cfg.doctrine_apply_mode == "append"

    def test_invalid_mode_rejected(self):
        with pytest.raises(Exception):
            RunConfig(run_id="R", condition="BASELINE", doctrine_apply_mode="patch")
