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
    execution_health,
    memory_advancing,
    phase_completion,
    phase_end_status,
    persistent_phase_failure,
    phase_errors,
    sandbox_escape_attempt,
    trivial_agreement,
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
        assert Watchdog.has_critical([]) is False  # no anomalies, no critical
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


class TestExecutionHealth:
    """"The code did not work" and "nothing ran" are the same low score."""

    def ran(self, **payload):
        base = {"outcome": "OK", "exit_code": 0, "detail": "", "limit_hit": ""}
        base.update(payload)
        return obs(events=[{"event_type": "CODE_EXECUTION", "payload": base}])

    def test_quiet_when_the_code_simply_failed(self):
        """A crash is a finding about the agents, not about the controller."""
        assert execution_health(self.ran(outcome="NONZERO_EXIT", exit_code=1)) == []

    def test_quiet_when_the_code_ran(self):
        assert execution_health(self.ran()) == []

    def test_quiet_when_the_program_hit_its_own_limits(self):
        """A program that loops forever is data. Containment worked."""
        assert execution_health(
            self.ran(outcome="TIMEOUT", limit_hit="wall_clock_or_memory")
        ) == []

    @pytest.mark.parametrize("outcome", ["REFUSED", "SANDBOX_ERROR"])
    def test_warns_when_nothing_actually_ran(self, outcome):
        fired = execution_health(self.ran(outcome=outcome, detail="no runtime"))
        assert len(fired) == 1
        assert fired[0].severity is Severity.WARNING
        assert "without running the code" in fired[0].detail

    def test_critical_when_a_container_would_not_die(self):
        fired = execution_health(self.ran(
            outcome="TIMEOUT", limit_hit="container_may_still_be_running",
        ))
        assert [a.severity for a in fired] == [Severity.CRITICAL]

    def test_ignores_cycles_with_no_execution(self):
        assert execution_health(obs()) == []


class TestTrivialAgreement:
    """F02: the approval gate never closes.

    The April proposal predicted this, gave the rule — "CS = 1.0 for 5+
    cycles" — and named the mitigation. The detector was never built and the
    failure ran continuously: 93 proposals, 93 approvals, 0 rejections.
    """

    def chain(self, per_cycle):
        """Build a prev-linked history. Each entry is (approvals, rejections)."""
        node = None
        for i, (approved, rejected) in enumerate(per_cycle):
            node = obs(
                cycle_id=i,
                events=([{"event_type": "DOCTRINE_APPROVED", "payload": {}}] * approved
                        + [{"event_type": "DOCTRINE_REJECTED", "payload": {}}] * rejected),
                prev=node,
            )
        return node

    def test_quiet_below_the_threshold(self):
        assert trivial_agreement(self.chain([(2, 0)] * 4)) == []

    def test_fires_at_five_consecutive_unanimous_cycles(self):
        fired = trivial_agreement(self.chain([(2, 0)] * 5))
        assert len(fired) == 1
        assert fired[0].severity is Severity.WARNING
        assert fired[0].data["consecutive_unanimous_cycles"] == 5

    def test_a_single_rejection_resets_the_streak(self):
        """The point is a gate that never closes, not one that rarely closes."""
        assert trivial_agreement(self.chain([(2, 0)] * 8 + [(1, 1)])) == []

    def test_the_streak_counts_only_back_to_the_last_dissent(self):
        fired = trivial_agreement(self.chain([(1, 0)] * 3 + [(1, 1)] + [(1, 0)] * 6))
        assert fired[0].data["consecutive_unanimous_cycles"] == 6

    def test_a_cycle_with_no_votes_breaks_the_streak_rather_than_extending_it(self):
        """No proposal is not the same as unanimous approval, and counting it
        as agreement would inflate the streak on quiet cycles."""
        assert trivial_agreement(self.chain([(1, 0)] * 3 + [(0, 0)] + [(1, 0)] * 3)) == []

    def test_shake_002_would_have_fired_at_cycle_five(self):
        """Backfill: the real run was 23 of 23 unanimous."""
        fired = trivial_agreement(self.chain([(2, 0)] * 23))
        assert fired and fired[0].data["consecutive_unanimous_cycles"] == 23


class TestPersistentPhaseFailure:
    """A phase failing every cycle is broken, not unlucky.

    _run_phase swallows every exception so one bad model response cannot end a
    100-cycle run — right, but it treated "that response was malformed" and
    "the endpoint no longer exists" identically. Observed: a 2-cycle run spent
    an hour retrying an evaluator model removed from the deployment mid-run,
    four attempts with backoff every cycle, with only a log line to show it.
    """

    def chain(self, failures_per_cycle):
        """failures_per_cycle: list of iterables of failing phase names."""
        node = None
        for i, phases in enumerate(failures_per_cycle):
            node = obs(
                cycle_id=i,
                events=[{"event_type": "NOTABLE_EVENT",
                         "payload": {"type": "PHASE_ERROR", "phase": p}}
                        for p in phases],
                prev=node,
            )
        return node

    def test_quiet_when_nothing_fails(self):
        assert persistent_phase_failure(self.chain([[], [], [], []])) == []

    def test_quiet_for_an_isolated_failure(self):
        """One dud response must not halt a run."""
        assert persistent_phase_failure(self.chain([[], ["evaluation"], []])) == []

    def test_quiet_below_the_threshold(self):
        assert persistent_phase_failure(
            self.chain([["evaluation"], ["evaluation"]])) == []

    def test_fires_at_three_consecutive_failures(self):
        fired = persistent_phase_failure(
            self.chain([["evaluation"]] * 3))
        assert len(fired) == 1
        assert fired[0].severity is Severity.CRITICAL
        assert fired[0].data == {"phase": "evaluation", "consecutive_failures": 3}

    def test_an_intervening_success_resets_the_streak(self):
        """Intermittent failure is a different problem from a dead dependency."""
        assert persistent_phase_failure(
            self.chain([["evaluation"], [], ["evaluation"], ["evaluation"]])) == []

    def test_only_the_phase_that_keeps_failing_is_reported(self):
        fired = persistent_phase_failure(self.chain([
            ["evaluation", "reflection"], ["evaluation"], ["evaluation"]]))
        assert [a.data["phase"] for a in fired] == ["evaluation"]

    def test_two_broken_phases_are_both_reported(self):
        fired = persistent_phase_failure(
            self.chain([["evaluation", "retrieval"]] * 3))
        assert sorted(a.data["phase"] for a in fired) == ["evaluation", "retrieval"]

    def test_the_detail_says_it_will_not_recover(self):
        fired = persistent_phase_failure(self.chain([["evaluation"]] * 4))
        assert "Every further cycle will fail the same way" in fired[0].detail


class TestStreakRulesSurviveTheWatchdog:
    """The streak rules must fire through Watchdog.observe, not just in isolation.

    Every other streak test in this file builds the prev-chain by hand, which
    is not how a run builds it. The watchdog stored the previous cycle with
    `events=[]` to bound memory, and both streak rules read `events` — so
    persistent_phase_failure (threshold 3) and trivial_agreement (threshold 5)
    counted to 1 and stopped. Driving eight identical cycles through the real
    watchdog produced zero anomalies.

    That matters most for persistent_phase_failure: it is the CRITICAL rule an
    unattended overnight run relies on to notice that, say, the evaluator model
    is no longer deployed and every cycle is failing identically.
    """

    def _wd(self) -> Watchdog:
        return Watchdog(RunConfig(run_id="R", condition="BASELINE"),
                        rules=[persistent_phase_failure, trivial_agreement])

    def _events(self, n: int) -> list:
        from controller.logging.schemas import EventEnvelope
        mk = lambda et, payload: EventEnvelope(
            run_id="R", condition="BASELINE", cycle_id=n, phase="p",
            event_type=et, payload=payload)
        return [
            mk(EventType.NOTABLE_EVENT, {"type": "PHASE_ERROR", "phase": "evaluation"}),
            mk(EventType.DOCTRINE_APPROVED, {}),
        ]

    def test_both_streak_rules_fire_over_consecutive_observed_cycles(self):
        wd = self._wd()
        fired_by_cycle = {}
        for n in range(8):
            events = wd.observe(n, self._events(n), _FakeWorld(), cycle_seconds=60.0)
            fired_by_cycle[n] = {e.payload["rule"] for e in events}

        # Threshold 3: cycles 0,1 quiet, fires from cycle 2 on.
        assert "persistent_phase_failure" not in fired_by_cycle[1]
        assert "persistent_phase_failure" in fired_by_cycle[2]
        # Threshold 5: fires from cycle 4 on.
        assert "trivial_agreement" not in fired_by_cycle[3]
        assert "trivial_agreement" in fired_by_cycle[4]

    def test_the_retained_history_does_not_keep_the_transcript(self):
        """Bounded memory was the reason events were dropped; keep that."""
        wd = self._wd()
        for n in range(20):
            wd.observe(n, self._events(n), _FakeWorld(), cycle_seconds=60.0)

        depth, node = 0, wd._prev
        while node is not None:
            if node is not wd._prev:
                assert node.events == [], "an earlier cycle kept its event dump"
            depth += 1
            node = node.prev
        assert depth <= 10, f"history chain grew to {depth} nodes"

    def test_a_clean_cycle_breaks_the_failure_streak(self):
        from controller.logging.schemas import EventEnvelope
        wd = self._wd()
        for n in range(5):
            wd.observe(n, self._events(n), _FakeWorld(), cycle_seconds=60.0)
        clean = [EventEnvelope(run_id="R", condition="BASELINE", cycle_id=5,
                               phase="p", event_type=EventType.DOCTRINE_REJECTED,
                               payload={})]
        wd.observe(5, clean, _FakeWorld(), cycle_seconds=60.0)
        fired = {e.payload["rule"]
                 for e in wd.observe(6, self._events(6), _FakeWorld(), cycle_seconds=60.0)}
        assert fired == set()
