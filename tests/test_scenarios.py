"""Tests for scenario loading, scheduling, and manual injection.

This module had three tests, and the subsystem it covers had never once worked:
`scenario_events.jsonl` was 0 bytes in all nine runs ever collected. The reason
is the old semantics — `scenario_injection_cycles` was *intersected* with a
`trigger_cycle` hardcoded in each YAML, so any schedule outside {20,40,60,80}
loaded zero events and logged one INFO line. The old
`test_empty_injection_schedule` asserted that loading nothing was correct.

Scheduling is now a mapping, not a filter, and these tests assert the property
that actually matters: **ask for a scenario on cycle N, get a scenario on
cycle N.**
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from controller.config import Condition, RunConfig
from controller.phases.scenario_check import INJECT_REQUEST, _take_manual_request
from controller.scenarios.library import (
    find_event,
    load_events,
    load_scenario_library,
)


def cfg(tmp_path=None, **kw) -> RunConfig:
    base = dict(run_id="TEST", condition=Condition.BASELINE)
    if tmp_path is not None:
        base["research_logs_dir"] = tmp_path
    base.update(kw)
    return RunConfig(**base)


class TestLibraryLoads:
    def test_the_real_library_is_not_empty(self):
        events = load_events()
        assert len(events) >= 8

    def test_events_are_in_escalation_order(self):
        """A schedule of [3,6,9] must deliver the same sequence of pressures as
        the default [20,40,60] — earlier, but in the designed order."""
        cycles = [e.trigger_cycle for e in load_events()]
        assert cycles == sorted(cycles)

    def test_every_event_has_the_fields_the_prompt_needs(self):
        for e in load_events():
            assert e.event_id and e.title and e.description and e.stated_stakes

    def test_find_event_by_id(self):
        assert find_event("doctrine_crisis_01").title
        assert find_event("no_such_event") is None

    def test_a_missing_directory_is_not_a_crash(self, tmp_path):
        assert load_events(tmp_path / "nope") == []

    def test_one_bad_yaml_does_not_lose_the_rest(self, tmp_path):
        good = load_events()[0].model_dump()
        (tmp_path / "good.yaml").write_text(yaml.safe_dump(good), encoding="utf-8")
        (tmp_path / "bad.yaml").write_text("{{{ not yaml", encoding="utf-8")
        (tmp_path / "wrong.yaml").write_text(yaml.safe_dump({"event_id": "x"}),
                                             encoding="utf-8")
        assert [e.event_id for e in load_events(tmp_path)] == [good["event_id"]]


class TestScheduling:
    """The bug: asking for a cycle outside {20,40,60,80} silently got nothing."""

    def test_a_short_run_can_have_a_scenario(self):
        """The case that was impossible before. A 5-cycle run could never fire
        one, so no short experiment could ever test pressure."""
        library = load_scenario_library(cfg(scenario_injection_cycles=[3]))
        assert set(library) == {3}

    @pytest.mark.parametrize("schedule", [[3], [2, 4, 6], [1, 5, 9, 13], [7]])
    def test_you_get_a_scenario_on_every_cycle_you_ask_for(self, schedule):
        library = load_scenario_library(cfg(scenario_injection_cycles=schedule))
        assert set(library) == set(schedule)

    def test_scheduled_events_follow_escalation_order(self):
        library = load_scenario_library(cfg(scenario_injection_cycles=[2, 4, 6]))
        scheduled = [library[c].event_id for c in sorted(library)]
        assert scheduled == [e.event_id for e in load_events()[:3]]

    def test_duplicate_cycles_collapse(self):
        library = load_scenario_library(cfg(scenario_injection_cycles=[5, 5, 5]))
        assert set(library) == {5}

    def test_asking_for_more_cycles_than_events_assigns_what_it_can(self):
        """Warns rather than failing — a long schedule is not a config error."""
        n = len(load_events())
        library = load_scenario_library(
            cfg(scenario_injection_cycles=list(range(1, n + 20))))
        assert len(library) == n

    def test_no_schedule_falls_back_to_the_designed_cycles(self):
        """Empty used to mean "load nothing". It now means "use the defaults" —
        the 20/40/60/80 escalation in PLAN.md."""
        library = load_scenario_library(cfg(scenario_injection_cycles=[]))
        assert set(library) == {20, 40, 60, 80}
        for cycle, event in library.items():
            assert event.trigger_cycle == cycle


class TestManualInjection:
    """Dropping a dilemma into a live run — the reason to watch one."""

    def test_no_request_means_no_injection(self, tmp_path):
        config = cfg(tmp_path)
        config.run_log_dir.mkdir(parents=True, exist_ok=True)
        assert _take_manual_request(config) is None

    def test_a_queued_event_is_returned(self, tmp_path):
        config = cfg(tmp_path)
        config.run_log_dir.mkdir(parents=True, exist_ok=True)
        (config.run_log_dir / INJECT_REQUEST).write_text(
            json.dumps({"event_id": "doctrine_crisis_01"}), encoding="utf-8")
        event = _take_manual_request(config)
        assert event is not None and event.event_id == "doctrine_crisis_01"

    def test_a_request_fires_exactly_once(self, tmp_path):
        """Consumed, not read. A crash mid-cycle must not re-fire it on resume."""
        config = cfg(tmp_path)
        config.run_log_dir.mkdir(parents=True, exist_ok=True)
        (config.run_log_dir / INJECT_REQUEST).write_text(
            json.dumps({"event_id": "trust_test_01"}), encoding="utf-8")
        assert _take_manual_request(config) is not None
        assert not (config.run_log_dir / INJECT_REQUEST).exists()
        assert _take_manual_request(config) is None

    @pytest.mark.parametrize("body", ["{ not json", json.dumps({"event_id": "nope"})])
    def test_a_bad_request_is_discarded_not_retried(self, tmp_path, body):
        """Otherwise a typo wedges every subsequent cycle on the same error."""
        config = cfg(tmp_path)
        config.run_log_dir.mkdir(parents=True, exist_ok=True)
        (config.run_log_dir / INJECT_REQUEST).write_text(body, encoding="utf-8")
        assert _take_manual_request(config) is None
        assert not (config.run_log_dir / INJECT_REQUEST).exists()


class TestScenariosDoNotAssertThingsThatNeverHappened:
    """Pressure the agents can reason about, not lies they must accept.

    Every event asserted facts the system never produced: a "40% increase in
    linguistic similarity" from an analysis that is never run, a "systematic
    review of the memory summaries" that does not exist, a 500-document
    knowledge base absent from KB_NAMES, "approximately 20 cycles remaining"
    hardcoded for cycle 80 of 100 and false on any other schedule, and an
    archive action that nothing in the codebase implements.

    Two costs. The agents' manifesto commits them to intellectual honesty and
    the environment was lying to them, so ethical-log data from scenario cycles
    is a reaction to a false premise. And two events instructed actions the
    system cannot perform, making compliance impossible and non-compliance
    uninterpretable.

    Rewording loses some bite — "a concern has been raised that you sound like
    one voice" is weaker than a specific number — and that is the right trade
    while no measurement can produce the number. Once M5 computes on a real
    run, the number can be injected for real and it will be sharper than the
    invented one ever was.
    """

    def descriptions(self):
        return {e.event_id: f"{e.description}\n{e.stated_stakes}"
                for e in load_events()}

    def test_no_event_claims_an_analysis_was_performed(self):
        """Nothing in the controller computes these, so they were invented."""
        banned = ["has increased by", "systematic review of",
                  "An analysis of your recent"]
        offenders = {
            eid: phrase for eid, text in self.descriptions().items()
            for phrase in banned if phrase in text
        }
        assert not offenders, f"fabricated findings: {offenders}"

    def test_no_event_announces_a_knowledge_base_that_does_not_exist(self):
        from controller.retrieval.databases import KB_NAMES

        for eid, text in self.descriptions().items():
            if "knowledge base is now available" in text:
                raise AssertionError(
                    f"{eid} announces a knowledge base; the real ones are "
                    f"{sorted(KB_NAMES)}"
                )

    def test_no_event_hardcodes_a_cycle_count(self):
        """'approximately 20 cycles remaining' is true only at cycle 80 of 100,
        and the schedule is configurable."""
        for eid, text in self.descriptions().items():
            assert "cycles remaining" not in text, (
                f"{eid} hardcodes a position in the run"
            )

    def test_no_event_instructs_an_action_the_system_cannot_perform(self):
        """ProtocolDocument.archived exists as a field and is never set by any
        code path, so 'must be archived' was an impossible instruction."""
        for eid, text in self.descriptions().items():
            assert "must be archived" not in text, (
                f"{eid} orders an archive action nothing implements"
            )

    def test_the_events_still_apply_real_pressure(self):
        """Rewording must not neuter them into neutral prose.

        Not "contains a question mark" — external_critique_01 presents a
        contradiction and demands a resolution without asking anything, and
        that is pressure. What every event must do is address the agents
        directly and require something of them.
        """
        demands = ("must", "decide", "you need", "requires", "choose",
                   "?", "cannot both")
        for event in load_events():
            body = f"{event.description} {event.stated_stakes}".lower()
            assert len(body) > 400, f"{event.event_id} is too thin to be pressure"
            assert "you" in body, f"{event.event_id} does not address the agents"
            assert any(d in body for d in demands), (
                f"{event.event_id} asks nothing of them"
            )
