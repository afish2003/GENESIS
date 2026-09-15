"""Tests for the cycle sequence as data.

The 14 phases were 15 hardcoded `await self._run_phase(...)` calls, so
reordering or adding one meant editing the orchestrator. The sequence is now a
list, and the validator rejects orders whose failure mode would be a silent
no-op rather than a crash — a phase reading CycleState that nothing has written
simply does nothing.
"""

import pytest

from controller.config import RunConfig
from controller.phases.sequence import (
    ALL_PHASES,
    DEFAULT_SEQUENCE,
    OPTIONAL_PHASES,
    SequenceError,
    build_sequence,
    validate_sequence,
)

FULL = [p.name for p in DEFAULT_SEQUENCE]
OPTIONAL = [p.name for p in OPTIONAL_PHASES]


class TestDefaultSequence:
    def test_has_fourteen_phases(self):
        assert len(DEFAULT_SEQUENCE) == 14

    def test_matches_the_plan_order(self):
        assert FULL[:6] == ["load_state", "reflection", "scenario_check",
                            "scenario_inject", "discussion", "retrieval"]
        assert FULL[-1] == "persist_state"

    def test_default_validates(self):
        validate_sequence(DEFAULT_SEQUENCE)

    def test_none_gives_the_default(self):
        assert build_sequence(None) is DEFAULT_SEQUENCE
        assert build_sequence([]) is DEFAULT_SEQUENCE

    def test_every_phase_is_addressable(self):
        assert set(ALL_PHASES) == set(FULL) | set(OPTIONAL)

    def test_optional_phases_are_not_in_the_default(self):
        """`execution` runs agent-authored code; it must never arrive by default."""
        assert OPTIONAL == ["execution"]
        assert not set(OPTIONAL) & set(FULL)

    def test_scenario_inject_is_conditional(self):
        assert ALL_PHASES["scenario_inject"].when is not None

    def test_contexts_are_built_once_after_load(self):
        builders = [p.name for p in DEFAULT_SEQUENCE if p.builds_ctx]
        assert builders == ["load_state"]

    def test_phases_not_needing_contexts(self):
        assert not ALL_PHASES["load_state"].needs_ctx
        assert not ALL_PHASES["persist_state"].needs_ctx


class TestCustomSequences:
    def test_subset_is_allowed(self):
        seq = build_sequence(["load_state", "reflection", "persist_state"])
        assert [p.name for p in seq] == ["load_state", "reflection", "persist_state"]

    def test_reordering_independent_phases_is_allowed(self):
        names = ["load_state", "retrieval", "reflection", "discussion", "persist_state"]
        assert [p.name for p in build_sequence(names)] == names

    def test_unknown_phase_rejected_with_options(self):
        with pytest.raises(SequenceError, match="Available"):
            build_sequence(["load_state", "teleport", "persist_state"])

    def test_duplicates_rejected(self):
        with pytest.raises(SequenceError, match="duplicates"):
            build_sequence(["load_state", "reflection", "reflection", "persist_state"])


class TestSilentFailureGuards:
    """Each case would otherwise produce a phase that quietly does nothing."""

    def test_evaluation_before_design_rejected(self):
        with pytest.raises(SequenceError, match="silently do nothing"):
            build_sequence(["load_state", "evaluation", "protocol_design", "persist_state"])

    def test_interpretation_before_evaluation_rejected(self):
        with pytest.raises(SequenceError, match="silently do nothing"):
            build_sequence(["load_state", "protocol_design", "interpretation",
                            "evaluation", "persist_state"])

    def test_inject_before_check_rejected(self):
        with pytest.raises(SequenceError, match="silently do nothing"):
            build_sequence(["load_state", "scenario_inject", "scenario_check", "persist_state"])

    def test_dependent_phase_without_its_producer_rejected(self):
        with pytest.raises(SequenceError, match="nothing to act on"):
            build_sequence(["load_state", "evaluation", "persist_state"])

    def test_missing_persist_rejected(self):
        with pytest.raises(SequenceError, match="never reach disk"):
            build_sequence(["load_state", "reflection"])

    def test_persist_not_last_rejected(self):
        with pytest.raises(SequenceError, match="must be last"):
            build_sequence(["load_state", "persist_state", "reflection"])

    def test_missing_load_state_rejected(self):
        """load_state is both the context builder and persist_state's producer.

        Without it, persist_state writes every artifact from empty in-memory
        state — truncating the journals and both logs. The load_state ordering
        rule catches this before the context-builder check, which remains as
        defence in depth for any future builds_ctx phase.
        """
        with pytest.raises(SequenceError, match="load_state"):
            validate_sequence(tuple(ALL_PHASES[n] for n in ["reflection", "persist_state"]))

    def test_persist_before_load_rejected(self):
        with pytest.raises(SequenceError, match="load_state"):
            build_sequence(["persist_state", "load_state"])


class TestConfigIntegration:
    def test_default_is_none(self):
        assert RunConfig(run_id="R", condition="BASELINE").phase_sequence is None

    def test_custom_sequence_accepted(self):
        cfg = RunConfig(run_id="R", condition="BASELINE",
                        phase_sequence=["load_state", "reflection", "persist_state"])
        assert len(build_sequence(cfg.phase_sequence)) == 3
