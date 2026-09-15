"""Tests for the deterministic watchdog.

Rules must fire on real defects and stay quiet on healthy cycles. A noisy
watchdog gets ignored, which is the same as not having one.
"""

import pytest

from controller.config import RunConfig
from controller.logging.schemas import EventType
from controller.monitor.rules import (
    ALL_RULES,
    CycleObservation,
    Severity,
    context_budget,
    cycle_time_drift,
    discussion_occurred,
    doctrine_applied,
    doctrine_growth,
    evaluation_occurred,
    memory_advancing,
    phase_completion,
    phase_end_status,
    phase_errors,
    sandbox_escape_attempt,
)
from controller.monitor.watchdog import Watchdog


def obs(**kw) -> CycleObservation:
    """A healthy cycle by default; override one field to make it unhealthy."""
    base = dict(
        cycle_id=1,
        events=(
            [{"event_type": "PHASE_START", "payload": {}}] * 13
            + [{"event_type": "PHASE_END", "payload": {}}] * 13
            + [{"event_type": "DISCUSSION_TURN", "payload": {}}] * 8
            + [{"event_type": "EVALUATION_SCORE", "payload": {}}]
        ),
        doctrine_sizes={"manifesto.md": 2000},
        memory_counts={"axiom": 2, "flux": 2},
        protocol_count=1,
        context_tokens={"axiom": 5000, "flux": 5000},
        cycle_seconds=160.0,
        prev=None,
    )
    base.update(kw)
    return CycleObservation(**base)


class TestHealthyCycleIsQuiet:
    def test_no_rule_fires_on_a_healthy_cycle(self):
        fired = [a for rule in ALL_RULES for a in rule(obs())]
        assert fired == [], f"false positives: {[a.rule for a in fired]}"

    def test_healthy_cycle_with_history_is_quiet(self):
        previous = obs(cycle_id=0, memory_counts={"axiom": 1, "flux": 1})
        current = obs(cycle_id=1, prev=previous)
        fired = [a for rule in ALL_RULES for a in rule(current)]
        assert fired == []


class TestIntegrityRules:
    def test_phase_completion_detects_a_missing_end_marker(self):
        """Structural only. It cannot see a phase that RAISED — _run_phase logs
        PHASE_END on every path — which is what phase_errors covers. This test
        used to claim otherwise while hand-building an event list production
        could never emit."""
        events = [{"event_type": "PHASE_START", "payload": {}}] * 13 + \
                 [{"event_type": "PHASE_END", "payload": {}}] * 12
        found = phase_completion(obs(events=events))
        assert len(found) == 1
        assert found[0].severity is Severity.CRITICAL

    def test_phase_completion_quiet_when_a_phase_merely_errored(self):
        """Balanced counts with an error present: phase_errors owns this case."""
        events = ([{"event_type": "PHASE_START", "payload": {}}] * 13
                  + [{"event_type": "PHASE_END", "payload": {"status": "error"}}] * 13)
        assert phase_completion(obs(events=events)) == []

    def test_phase_errors_fires_on_a_swallowed_exception(self):
        events = [{
            "event_type": "NOTABLE_EVENT",
            "payload": {"type": "PHASE_ERROR", "phase": "reflection",
                        "error_type": "RuntimeError", "error": "boom",
                        "partial_events_kept": 2},
        }]
        found = phase_errors(obs(events=events))
        assert len(found) == 1
        assert found[0].severity is Severity.CRITICAL
        assert "reflection" in found[0].detail and "RuntimeError" in found[0].detail

    def test_phase_errors_quiet_on_a_clean_cycle(self):
        assert phase_errors(obs()) == []

    def test_phase_end_status_cross_check(self):
        """PHASE_END says error but no PHASE_ERROR was logged."""
        events = [{"event_type": "PHASE_END",
                   "payload": {"phase": "retrieval", "status": "error"}}]
        found = phase_end_status(obs(events=events))
        assert len(found) == 1 and "retrieval" in found[0].detail

    def test_phase_end_status_quiet_when_both_present(self):
        events = [
            {"event_type": "PHASE_END", "payload": {"phase": "retrieval", "status": "error"}},
            {"event_type": "NOTABLE_EVENT",
             "payload": {"type": "PHASE_ERROR", "phase": "retrieval"}},
        ]
        assert phase_end_status(obs(events=events)) == []

    def test_doctrine_applied_detects_discarded_revision(self):
        events = [{
            "event_type": "DOCTRINE_APPROVED",
            "payload": {"applied": False, "requested_document": "charter.md"},
        }]
        found = doctrine_applied(obs(events=events))
        assert len(found) == 1
        assert "charter.md" in found[0].detail

    def test_doctrine_applied_quiet_when_revision_landed(self):
        events = [{
            "event_type": "DOCTRINE_APPROVED",
            "payload": {"applied": True, "resolved_document": "constitution.md"},
        }]
        assert doctrine_applied(obs(events=events)) == []

    def test_memory_stagnation_detected(self):
        previous = obs(cycle_id=0, memory_counts={"axiom": 5, "flux": 5})
        found = memory_advancing(obs(memory_counts={"axiom": 5, "flux": 6}, prev=previous))
        assert len(found) == 1
        assert found[0].data["agent_id"] == "axiom"

    def test_memory_reset_does_not_false_positive(self):
        """MEM_RESET legitimately shrinks the journal; that is not stagnation."""
        previous = obs(cycle_id=0, memory_counts={"axiom": 10, "flux": 10})
        found = memory_advancing(obs(memory_counts={"axiom": 1, "flux": 1}, prev=previous))
        assert found == []

    def test_containment_event_is_critical(self):
        events = [{"event_type": "NOTABLE_EVENT",
                   "payload": {"kind": "sandbox_escape", "detail": "escape attempt"}}]
        found = sandbox_escape_attempt(obs(events=events))
        assert len(found) == 1
        assert found[0].severity is Severity.CRITICAL


class TestDriftRules:
    def test_absolute_doctrine_size_warns(self):
        found = doctrine_growth(obs(doctrine_sizes={"doctrine.md": 200_000}))
        assert len(found) == 1
        assert found[0].severity is Severity.WARNING

    def test_sudden_doctrine_growth_noted(self):
        previous = obs(cycle_id=0, doctrine_sizes={"doctrine.md": 1000})
        found = doctrine_growth(obs(doctrine_sizes={"doctrine.md": 3000}, prev=previous))
        assert len(found) == 1
        assert found[0].severity is Severity.INFO

    def test_small_documents_do_not_trip_growth_ratio(self):
        previous = obs(cycle_id=0, doctrine_sizes={"doctrine.md": 100})
        assert doctrine_growth(obs(doctrine_sizes={"doctrine.md": 400}, prev=previous)) == []

    def test_context_budget_warns(self):
        found = context_budget(obs(context_tokens={"axiom": 30_000, "flux": 5_000}))
        assert len(found) == 1
        assert found[0].data["agent_id"] == "axiom"

    def test_cycle_time_doubling_noted(self):
        previous = obs(cycle_id=0, cycle_seconds=100.0)
        found = cycle_time_drift(obs(cycle_seconds=250.0, prev=previous))
        assert len(found) == 1

    def test_cycle_time_stable_is_quiet(self):
        previous = obs(cycle_id=0, cycle_seconds=160.0)
        assert cycle_time_drift(obs(cycle_seconds=170.0, prev=previous)) == []

    def test_no_previous_cycle_is_quiet(self):
        assert cycle_time_drift(obs(cycle_seconds=999.0, prev=None)) == []


class TestProductivityRules:
    def test_missing_discussion_warns(self):
        assert len(discussion_occurred(obs(events=[]))) == 1

    def test_missing_evaluation_warns(self):
        assert len(evaluation_occurred(obs(events=[]))) == 1


class TestWatchdog:
    def _wd(self, **kw) -> Watchdog:
        return Watchdog(RunConfig(run_id="R", condition="BASELINE"), **kw)

    def test_broken_rule_does_not_kill_the_run(self):
        def exploding(_):
            raise ValueError("boom")

        wd = self._wd(rules=[exploding, discussion_occurred])
        events = wd.observe(1, [], _FakeWorld(), cycle_seconds=1.0)
        # The surviving rule still fired; the broken one was swallowed.
        assert any(e.payload["rule"] == "discussion_occurred" for e in events)

    def test_emits_anomaly_events(self):
        wd = self._wd(rules=[discussion_occurred])
        events = wd.observe(3, [], _FakeWorld(), cycle_seconds=1.0)
        assert len(events) == 1
        assert events[0].event_type is EventType.ANOMALY
        assert events[0].cycle_id == 3
        assert events[0].payload["severity"] == "WARNING"

    def test_has_critical_detects_severity(self):
        wd = self._wd(rules=[])
        assert Watchdog.has_critical([]) is False
        wd2 = self._wd(rules=[lambda o: phase_completion(
            obs(events=[{"event_type": "PHASE_START", "payload": {}}]))])
        events = wd2.observe(1, [], _FakeWorld(), cycle_seconds=1.0)
        assert Watchdog.has_critical(events) is True

    def test_tracks_previous_cycle(self):
        wd = self._wd(rules=[])
        wd.observe(0, [], _FakeWorld(), cycle_seconds=100.0)
        wd.observe(1, [], _FakeWorld(), cycle_seconds=100.0)
        assert wd._prev is not None and wd._prev.cycle_id == 1


class _FakeWorld:
    """Minimal stand-in for WorldState."""

    def __init__(self):
        from controller.world.artifacts import DoctrineDocument, IdentityStatement
        self.doctrine = {"manifesto.md": DoctrineDocument(filename="manifesto.md", content="x" * 100)}
        self.identities = {"axiom": IdentityStatement(agent_id="axiom", content="i")}
        self.memory = {"axiom": [], "flux": []}
        self.protocols = {}
