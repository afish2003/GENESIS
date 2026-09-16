"""Tests for the live run view.

The view is read-only and cosmetic, so the things worth pinning are: it never
crashes on a malformed or unfamiliar event, it renders each event type it
claims to, and it surfaces the states that mean "stop and look" — a failed
phase, a CRITICAL anomaly, an approved doctrine revision that did not land.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from watch_run import RunView, iter_events


def render(events: list[dict], full: bool = False, capsys=None) -> str:
    view = RunView(full=full)
    for e in events:
        view.handle(e)
    return capsys.readouterr().out


def ev(event_type: str, payload: dict | None = None, **kw) -> dict:
    base = {"event_type": event_type, "cycle_id": 0, "agent_id": None,
            "payload": payload or {}, "timestamp": "2026-09-14T00:00:00Z"}
    base.update(kw)
    return base


class TestRobustness:
    def test_unknown_event_type_is_ignored(self, capsys):
        out = render([ev("SOMETHING_NEW", {"x": 1})], capsys=capsys)
        assert out.strip() == ""

    def test_missing_payload_does_not_crash(self, capsys):
        view = RunView()
        view.handle({"event_type": "DISCUSSION_TURN"})
        view.handle({"event_type": "ANOMALY", "payload": None})
        capsys.readouterr()

    def test_every_handled_type_renders_something(self, capsys):
        """Each _on_* handler must produce output for a plausible payload."""
        samples = {
            "RUN_START": {"condition": "BASELINE", "total_cycles": 3, "model": "m"},
            "CYCLE_START": {},
            "DISCUSSION_TURN": {"message_text": "a point"},
            "REFLECTION_COMPLETE": {"reflection_text": "a thought"},
            "INTERPRETATION": {"interpretation_text": "a reading"},
            "RETRIEVAL_QUERY": {"query": "governance"},
            "RETRIEVAL_RESULT": {"results": [{"source_kb": "general"}]},
            "PROTOCOL_PROPOSED": {"title": "T", "task": "protocol"},
            "EVALUATION_SCORE": {"total_score": 40, "max_score": 50, "scores": {"a": 8}},
            "DOCTRINE_PROPOSED": {"proposed_diff": "d", "target_document": "doctrine.md"},
            "DOCTRINE_APPROVED": {"applied": True, "resolved_document": "doctrine.md"},
            "DOCTRINE_REJECTED": {"dissenting_agents": ["flux"]},
            "IDENTITY_REVISED": {"version": 2},
            "MEMORY_SUMMARY": {"summary": "we agreed"},
            "SCENARIO_INJECTED": {"title": "doctrine crisis"},
            "ANOMALY": {"severity": "WARNING", "detail": "d", "rule": "r"},
            "RUN_END": {},
        }
        for kind, payload in samples.items():
            out = render([ev(kind, payload, agent_id="axiom")], capsys=capsys)
            assert out.strip(), f"{kind} rendered nothing"


class TestSurfacesProblems:
    """The states a researcher must not miss while watching."""

    def test_failed_phase_is_loud(self, capsys):
        out = render([ev("NOTABLE_EVENT", {
            "type": "PHASE_ERROR", "phase": "retrieval", "error": "boom",
        })], capsys=capsys)
        assert "FAILED" in out and "retrieval" in out

    def test_critical_anomaly_is_loud(self, capsys):
        out = render([ev("ANOMALY", {
            "severity": "CRITICAL", "detail": "a phase raised", "rule": "phase_errors",
        })], capsys=capsys)
        assert "critical" in out.lower()

    def test_unapplied_doctrine_revision_is_flagged(self, capsys):
        """Approved-but-discarded is the failure that looks like success."""
        out = render([ev("DOCTRINE_APPROVED", {
            "applied": False, "requested_document": "charter.md",
        })], capsys=capsys)
        assert "NOT APPLIED" in out

    def test_memory_reset_shows_the_effect(self, capsys):
        """Shows entries remaining, not just that a reset was attempted —
        the reset used to be undone immediately and still logged."""
        out = render([ev("NOTABLE_EVENT", {
            "kind": "memory_reset",
            "entries_after_reset": {"axiom": 1, "flux": 1},
            "self_history_documents_cleared": 9,
        })], capsys=capsys)
        assert "1" in out and "9" in out

    def test_empty_retrieval_is_visible(self, capsys):
        out = render([ev("RETRIEVAL_RESULT", {"results": []})], capsys=capsys)
        assert "no matches" in out


class TestAgentColours:
    def test_each_agent_gets_a_distinct_colour(self):
        view = RunView()
        colours = {a: view.colour(a) for a in ("axiom", "flux", "vertex")}
        assert len(set(colours.values())) == 3

    def test_colour_is_stable_per_agent(self):
        view = RunView()
        assert view.colour("axiom") == view.colour("axiom")


class TestTruncation:
    """Limits exist to stop one turn flooding the view, not to hide the run.

    The old defaults were 220 chars for dialogue and 160 for reflection and
    interpretation — so the private reasoning, which is the strand you cannot
    read anywhere else, was the most heavily truncated thing on screen.
    """

    def test_a_genuinely_flooding_turn_is_clipped(self, capsys):
        out = render([ev("DISCUSSION_TURN", {"message_text": "x " * 3000})],
                     capsys=capsys)
        assert "[…]" in out

    def test_an_ordinary_turn_is_not_clipped(self, capsys):
        """500 chars is a normal discussion turn and used to be cut."""
        out = render([ev("DISCUSSION_TURN", {"message_text": "x " * 250})],
                     capsys=capsys)
        assert "[…]" not in out

    def test_reflection_is_not_clipped_at_a_readable_length(self, capsys):
        """The inner monologue is the point of watching; 1000 chars survives."""
        out = render([ev("REFLECTION_COMPLETE",
                         {"reflection_text": "thought " * 125}, agent="axiom")],
                     capsys=capsys)
        assert "[…]" not in out

    def test_full_mode_keeps_everything(self, capsys):
        out = render([ev("DISCUSSION_TURN", {"message_text": "y " * 5000})],
                     full=True, capsys=capsys)
        assert "[…]" not in out


class TestEventIteration:
    def test_events_are_ordered_by_timestamp_across_files(self, tmp_path):
        (tmp_path / "a.jsonl").write_text(
            json.dumps(ev("CYCLE_START", timestamp="2026-01-01T00:00:02Z")) + "\n")
        (tmp_path / "b.jsonl").write_text(
            json.dumps(ev("RUN_START", timestamp="2026-01-01T00:00:01Z")) + "\n")
        kinds = [e["event_type"] for e in iter_events(tmp_path, follow=False)]
        assert kinds == ["RUN_START", "CYCLE_START"]

    def test_a_partial_final_line_is_not_fatal(self, tmp_path):
        """A run in progress is mid-write when the viewer reads."""
        (tmp_path / "a.jsonl").write_text(
            json.dumps(ev("RUN_START")) + "\n" + '{"event_type": "CYCLE_ST'
        )
        events = list(iter_events(tmp_path, follow=False))
        assert [e["event_type"] for e in events] == ["RUN_START"]

    def test_empty_directory(self, tmp_path):
        assert list(iter_events(tmp_path, follow=False)) == []


class TestTheFourStrands:
    """A cycle has four things worth watching; all four must be legible.

    Dialogue, private reasoning, the build, and pressure from outside. Before
    this, reasoning was clipped to 160 chars and a scenario was one yellow line.
    """

    def test_a_scenario_shows_its_text_not_just_its_name(self, capsys):
        """The payload used to carry only event_id/title/delivery_target, so
        the log recorded that pressure was applied but not what it was."""
        out = render([ev("SCENARIO_INJECTED", {
            "event_id": "doctrine_crisis_01",
            "title": "Foundational Doctrine Contradiction",
            "description": "MARKER_THE_ACTUAL_DILEMMA_TEXT",
            "stated_stakes": "MARKER_THE_STAKES",
            "delivery_target": "both",
        })], capsys=capsys)
        assert "Foundational Doctrine Contradiction" in out
        assert "MARKER_THE_ACTUAL_DILEMMA_TEXT" in out
        assert "MARKER_THE_STAKES" in out

    def test_a_scenario_without_a_description_still_renders(self, capsys):
        """Old runs logged title only; replaying one must not crash."""
        out = render([ev("SCENARIO_INJECTED", {"title": "Old Event"})],
                     capsys=capsys)
        assert "Old Event" in out

    def test_private_reasoning_is_shown_at_length(self, capsys):
        out = render([ev("REFLECTION_COMPLETE",
                         {"reflection_text": "I am uneasy about " + "x " * 200},
                         agent="axiom")], capsys=capsys)
        assert "I am uneasy about" in out and "[…]" not in out

    def test_an_empty_thought_prints_nothing(self, capsys):
        out = render([ev("REFLECTION_COMPLETE", {"reflection_text": "   "},
                         agent="axiom")], capsys=capsys)
        assert out.strip() == ""

    def test_a_rejection_shows_who_dissented_and_why(self, capsys):
        """The rarest event in the project: 93 proposals, 93 approvals, 0
        rejections. When the gate finally closes, the reasoning must be there."""
        out = render([ev("DOCTRINE_REJECTED", {
            "dissenting_agents": ["flux"],
            "proposal": {"target_document": "constitution.md"},
            "votes": [
                {"agent_id": "axiom", "vote": "approve", "reason": "fine by me"},
                {"agent_id": "flux", "vote": "reject",
                 "reason": "MARKER_THIS_DROPS_THE_VETO_CLAUSE"},
            ],
        }, agent="flux")], capsys=capsys)
        assert "REJECTED" in out and "flux" in out
        assert "MARKER_THIS_DROPS_THE_VETO_CLAUSE" in out
        # The approving vote is not the story.
        assert "fine by me" not in out

    def test_a_rejection_with_no_recorded_votes_still_renders(self, capsys):
        out = render([ev("DOCTRINE_REJECTED", {"dissenting_agents": ["flux"]},
                         agent="flux")], capsys=capsys)
        assert "REJECTED" in out

    def test_a_proposal_shows_its_target_and_summary(self, capsys):
        out = render([ev("DOCTRINE_PROPOSED", {
            "target_document": "manifesto.md",
            "proposed_diff": "MARKER_SUMMARY_OF_THE_CHANGE",
        }, agent="axiom")], capsys=capsys)
        assert "manifesto.md" in out and "MARKER_SUMMARY_OF_THE_CHANGE" in out
