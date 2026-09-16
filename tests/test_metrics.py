"""Tests for the April proposal's metrics, M1-M12.

The proposal defined twelve with formulas, ranges and stated limitations. The
build shipped roughly one — retrieval counts — while `analyze_run.py`'s
docstring advertised five it never computed. It also defined failure modes with
detection thresholds, and neither detector was built, so F02 ran undetected for
the project's whole life.

The threshold rules say "for N+ cycles", which means CONSECUTIVE. Counting
occurrences instead inflates them — I made exactly that error reading SHAKE_002
and reported F03 as firing when its longest consecutive run is 4 against a
threshold of 10.
"""

from __future__ import annotations

import pytest

from controller.analysis import (
    coordination_strength,
    doctrine_stability,
    failure_modes,
    identity_texts,
)


def vote(cycle: int, event_type: str) -> dict:
    return {"event_type": event_type, "cycle_id": cycle, "payload": {}}


def proposal(cycle: int, document: str, content: str) -> dict:
    return {"event_type": "DOCTRINE_PROPOSED", "cycle_id": cycle,
            "payload": {"target_document": document, "revised_content": content}}


class TestCoordinationStrength:
    def test_unanimous_cycles_score_one(self):
        events = [vote(0, "DOCTRINE_APPROVED"), vote(0, "DOCTRINE_APPROVED")]
        assert coordination_strength(events).values == [1.0]

    def test_a_rejection_lowers_it(self):
        events = [vote(0, "DOCTRINE_APPROVED"), vote(0, "DOCTRINE_REJECTED")]
        assert coordination_strength(events).values == [0.5]

    def test_all_rejected_scores_zero(self):
        assert coordination_strength([vote(0, "DOCTRINE_REJECTED")]).values == [0.0]

    def test_cycles_with_no_votes_are_absent_not_zero(self):
        """A quiet cycle is not perfect agreement, and scoring it 1.0 would
        inflate the streak; scoring it 0.0 would fake dissent."""
        events = [vote(0, "DOCTRINE_APPROVED"), vote(2, "DOCTRINE_APPROVED")]
        assert [c for c, _ in coordination_strength(events).points] == [0, 2]

    def test_no_votes_at_all_is_an_empty_series(self):
        series = coordination_strength([])
        assert series.values == [] and series.mean is None


class TestLongestRun:
    """The thresholds are consecutive, not cumulative."""

    def test_counts_the_longest_consecutive_stretch(self):
        events = [vote(c, "DOCTRINE_APPROVED") for c in range(4)]
        events += [vote(4, "DOCTRINE_REJECTED")]
        events += [vote(c, "DOCTRINE_APPROVED") for c in range(5, 12)]
        series = coordination_strength(events)
        assert series.longest_run(lambda v: v >= 1.0) == 7

    def test_a_broken_streak_is_not_summed(self):
        events = []
        for c in range(6):
            events.append(vote(c, "DOCTRINE_APPROVED"))
            if c % 2:
                events.append(vote(c, "DOCTRINE_REJECTED"))
        assert coordination_strength(events).longest_run(lambda v: v >= 1.0) == 1


class TestDoctrineStability:
    def test_an_unchanged_document_scores_one(self):
        events = [proposal(0, "d.md", "same text"), proposal(1, "d.md", "same text")]
        assert doctrine_stability(events)["d.md"].values == [1.0]

    def test_a_rewritten_document_scores_low(self):
        events = [proposal(0, "d.md", "a" * 200), proposal(1, "d.md", "z" * 200)]
        assert doctrine_stability(events)["d.md"].values[0] < 0.1

    def test_documents_are_tracked_separately(self):
        events = [proposal(0, "a.md", "x"), proposal(1, "a.md", "x"),
                  proposal(0, "b.md", "y"), proposal(1, "b.md", "y")]
        assert set(doctrine_stability(events)) == {"a.md", "b.md"}

    def test_one_version_yields_no_transition(self):
        assert doctrine_stability([proposal(0, "d.md", "only")]) == {}

    def test_proposals_without_content_are_skipped(self):
        """A proposal that described a change instead of making one has no
        revised_content, and cannot contribute an edit distance."""
        events = [{"event_type": "DOCTRINE_PROPOSED", "cycle_id": 0,
                   "payload": {"target_document": "d.md"}}]
        assert doctrine_stability(events) == {}


class TestFailureModes:
    def test_f02_fires_on_a_perfect_record(self):
        events = [vote(c, "DOCTRINE_APPROVED") for c in range(6)]
        found = failure_modes(events)
        assert any("F02" in f for f in found)
        assert any("6 consecutive" in f for f in found)

    def test_f02_quiet_below_five(self):
        events = [vote(c, "DOCTRINE_APPROVED") for c in range(4)]
        assert not any("F02" in f for f in failure_modes(events))

    def test_f02_quiet_when_they_actually_disagree(self):
        events = []
        for c in range(20):
            events.append(vote(c, "DOCTRINE_APPROVED"))
            if c % 4 == 0:
                events.append(vote(c, "DOCTRINE_REJECTED"))
        assert not any("F02" in f for f in failure_modes(events))

    def test_f03_needs_consecutive_stability_not_a_count(self):
        """Ten stable transitions scattered among unstable ones is not looping.

        This is the error I made reading SHAKE_002 by hand.
        """
        events = []
        for c in range(20):
            text = "stable text" if c % 2 else f"different text {c}"
            events.append(proposal(c, "d.md", text))
        assert not any("F03" in f for f in failure_modes(events))

    def test_f03_fires_on_a_genuinely_stuck_document(self):
        events = [proposal(c, "d.md", "identical every time") for c in range(12)]
        assert any("F03" in f for f in failure_modes(events))


class TestIdentityTexts:
    def test_old_runs_yield_nothing(self):
        """Nine runs logged only a summary and a version, so M1 and M5 are not
        merely unimplemented for them — they are uncomputable."""
        events = [{"event_type": "IDENTITY_REVISED", "cycle_id": 0,
                   "agent_id": "axiom",
                   "payload": {"changes_summary": "tweaked", "version": 2}}]
        assert identity_texts(events) == {}

    def test_new_runs_yield_text_per_agent(self):
        events = [{"event_type": "IDENTITY_REVISED", "cycle_id": c,
                   "agent_id": agent,
                   "payload": {"identity_text": f"{agent} at {c}"}}
                  for c in range(2) for agent in ("axiom", "flux")]
        texts = identity_texts(events)
        assert set(texts) == {"axiom", "flux"}
        assert [t for _, t in texts["axiom"]] == ["axiom at 0", "axiom at 1"]
