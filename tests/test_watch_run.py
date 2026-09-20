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


#: Keys an event envelope actually has. `base.update(kw)` used to accept
#: anything, so `ev(..., agent="axiom")` — a plausible typo for `agent_id` —
#: added a junk key and left agent_id None. Several tests looked like they
#: exercised per-agent rendering and silently did not.
_EVENT_KEYS = {"event_type", "cycle_id", "agent_id", "payload", "timestamp"}


def ev(event_type: str, payload: dict | None = None, **kw) -> dict:
    if "agent" in kw:
        kw["agent_id"] = kw.pop("agent")
    unknown = set(kw) - _EVENT_KEYS
    if unknown:
        raise TypeError(
            f"ev() got unexpected event field(s) {sorted(unknown)}; an event "
            f"envelope has {sorted(_EVENT_KEYS)}. A misspelled field would "
            f"otherwise be silently ignored and the test would assert nothing."
        )
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


class TestLiveTail:
    """Following the generation feed.

    The controller truncates this file at the start of every cycle, so the
    reader must notice it shrinking. A reader that only ever seeks forward goes
    permanently silent after cycle 0 — and silently, which is the worst kind.
    """

    def _tail(self, tmp_path):
        from watch_run import LiveTail

        path = tmp_path / "live.jsonl"
        path.write_text("", encoding="utf-8")
        return LiveTail(path), path

    def _write(self, path, *deltas, phase="discussion"):
        with path.open("a", encoding="utf-8") as f:
            for d in deltas:
                f.write(json.dumps({"phase": phase, "cycle_id": 0, "delta": d}) + "\n")

    def test_nothing_to_read_is_not_an_error(self, tmp_path):
        tail, _ = self._tail(tmp_path)
        assert tail.poll() is False
        assert str(tail.render()) == ""

    def test_a_missing_file_is_not_an_error(self, tmp_path):
        from watch_run import LiveTail

        assert LiveTail(tmp_path / "absent.jsonl").poll() is False

    def test_deltas_accumulate_into_readable_text(self, tmp_path):
        tail, path = self._tail(tmp_path)
        self._write(path, "Hello", " there", ", partner")
        assert tail.poll() is True
        assert "Hello there, partner" in str(tail.render())

    def test_the_phase_label_is_carried(self, tmp_path):
        tail, path = self._tail(tmp_path)
        self._write(path, "x", phase="reflection")
        tail.poll()
        assert "reflection" in str(tail.render())

    def test_truncation_restarts_the_reader(self, tmp_path):
        """The bug this guards: after cycle 0 the file shrinks, and a
        forward-only reader sits past EOF showing nothing for the whole run."""
        tail, path = self._tail(tmp_path)
        self._write(path, "first cycle text")
        tail.poll()
        assert "first cycle text" in str(tail.render())

        path.write_text("", encoding="utf-8")          # new cycle
        self._write(path, "second cycle text")
        assert tail.poll() is True
        rendered = str(tail.render())
        assert "second cycle text" in rendered
        assert "first cycle" not in rendered

    def test_a_half_written_line_is_skipped_not_fatal(self, tmp_path):
        tail, path = self._tail(tmp_path)
        with path.open("a", encoding="utf-8") as f:
            f.write('{"phase": "x", "delta": "good"}\n{"phase": "x", "del')
        assert tail.poll() is True
        assert "good" in str(tail.render())

    def test_the_buffer_does_not_grow_without_bound(self, tmp_path):
        tail, path = self._tail(tmp_path)
        self._write(path, *["word " * 50 for _ in range(200)])
        tail.poll()
        assert len(tail.buffer) <= tail.keep * 4 + 16

    def test_clear_drops_the_streamed_copy(self, tmp_path):
        """Called when the finished event renders, so nothing prints twice."""
        tail, path = self._tail(tmp_path)
        self._write(path, "some text")
        tail.poll()
        tail.clear()
        assert str(tail.render()) == ""


class TestLiveFeedIsNotResearchData:
    def test_the_event_iterator_ignores_it(self, tmp_path):
        """live.jsonl rows have no event_type and no timestamp. Treating them
        as events would reorder the transcript around rows that sort as ''."""
        from watch_run import LIVE_FILE, iter_events

        (tmp_path / "transcripts.jsonl").write_text(
            json.dumps(ev("DISCUSSION_TURN", {"message_text": "real"})) + "\n",
            encoding="utf-8")
        (tmp_path / LIVE_FILE).write_text(
            json.dumps({"phase": "discussion", "delta": "partial"}) + "\n",
            encoding="utf-8")

        events = list(iter_events(tmp_path, follow=False))
        assert len(events) == 1
        assert events[0]["event_type"] == "DISCUSSION_TURN"


class TestReadableStream:
    """Most phases emit structured output, so the raw stream is JSON.

    Watching `{"message_text": "` arrive character by character is not watching
    an agent think. The preview lifts out the prose.
    """

    def readable(self, raw):
        from watch_run import LiveTail

        return LiveTail.readable(raw)

    def test_plain_prose_is_untouched(self):
        text = "Plain prose streaming in normally, no JSON anywhere."
        assert self.readable(text) == text

    def test_a_completed_field_yields_its_value(self):
        raw = ('{"message_text": "I am not convinced the evaluation feedback '
               'supports that change", "references": []}')
        assert self.readable(raw) == (
            "I am not convinced the evaluation feedback supports that change")

    def test_a_field_still_being_written_is_shown(self):
        """The most interesting part is the part not finished yet."""
        raw = '{"message_text": "We should refine the protocol docum'
        assert self.readable(raw) == "We should refine the protocol docum"

    def test_a_complete_object_does_not_leak_punctuation(self):
        """The first version appended the text after the last quote
        unconditionally, so a finished object showed a trailing ': []}'."""
        raw = ('{"message_text": "A sufficiently long value to be shown here", '
               '"refs": []}')
        assert "]}" not in self.readable(raw)
        assert ":" not in self.readable(raw)

    def test_escaped_quotes_survive(self):
        raw = ('{"t": "He said \\"no\\" to the revision, which I think was '
               'the right call"}')
        assert self.readable(raw) == (
            'He said "no" to the revision, which I think was the right call')

    def test_short_values_are_not_shown(self):
        """Ids, enums and field names are not reading material."""
        assert self.readable('{"action": "create", "id": "x1", "n": 3}') == ""

    def test_empty_input_is_empty_output(self):
        assert self.readable("") == ""

    def test_several_long_fields_are_joined(self):
        raw = ('{"a": "the first sufficiently long passage of writing here", '
               '"b": "the second sufficiently long passage of writing here"}')
        out = self.readable(raw)
        assert "first sufficiently long" in out and "second sufficiently long" in out
